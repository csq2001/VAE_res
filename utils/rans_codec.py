from __future__ import annotations

import struct

import numpy as np
import torch


RANS_BYTE_L = 1 << 23


def _validate_precision(precision: int) -> None:
    if precision < 1 or precision > 16:
        raise ValueError("rANS precision must be between 1 and 16 bits")


def _probabilities_to_cdf(probabilities: np.ndarray, precision: int) -> np.ndarray:
    _validate_precision(precision)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    total = 1 << precision
    if probabilities.size > total:
        raise ValueError("CDF precision is too small for the symbol alphabet")

    probabilities = probabilities.astype(np.float64, copy=False)
    probabilities = np.maximum(probabilities, 0.0)
    probability_sum = float(probabilities.sum())
    if not np.isfinite(probability_sum) or probability_sum <= 0.0:
        probabilities = np.full(probabilities.shape, 1.0 / probabilities.size)
    else:
        probabilities = probabilities / probability_sum

    remaining = total - probabilities.size
    scaled = probabilities * remaining
    frequencies = np.floor(scaled).astype(np.int64) + 1
    missing = total - int(frequencies.sum())
    if missing:
        fractions = scaled - np.floor(scaled)
        order = np.argsort(-fractions, kind="stable")
        frequencies[order[:missing]] += 1

    cdf = np.empty(probabilities.size + 1, dtype=np.int32)
    cdf[0] = 0
    np.cumsum(frequencies, out=cdf[1:])
    if int(cdf[-1]) != total or np.any(np.diff(cdf) <= 0):
        raise RuntimeError("failed to construct a valid quantized CDF")
    return cdf


def channel_cdfs_from_logits(
    logits: torch.Tensor,
    precision: int = 16,
    checkerboard_context: bool = False,
) -> np.ndarray:
    if logits.ndim != 5:
        raise ValueError("logits must have shape [batch, channels, symbols, height, width]")
    probabilities = torch.softmax(logits.detach().float(), dim=2)
    if checkerboard_context:
        height, width = logits.shape[-2:]
        rows_grid = torch.arange(height, device=logits.device).view(height, 1)
        columns_grid = torch.arange(width, device=logits.device).view(1, width)
        anchor_mask = (rows_grid + columns_grid) % 2 == 0
        anchor_probabilities = probabilities[..., anchor_mask].mean(dim=-1)
        nonanchor_probabilities = probabilities[..., ~anchor_mask].mean(dim=-1)
        grouped = torch.stack([anchor_probabilities, nonanchor_probabilities], dim=2)
        rows = grouped.cpu().numpy().reshape(-1, probabilities.shape[2])
    else:
        channel_probabilities = probabilities.mean(dim=(3, 4)).cpu().numpy()
        rows = channel_probabilities.reshape(-1, channel_probabilities.shape[2])
    return np.stack([_probabilities_to_cdf(row, precision) for row in rows])


def reshape_residual_logits(logits: torch.Tensor, channels: int, num_symbols: int) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("residual logits must have shape [batch, channels * symbols, height, width]")
    batch, packed_channels, height, width = logits.shape
    if packed_channels != channels * num_symbols:
        raise ValueError(
            f"expected {channels * num_symbols} logit channels, received {packed_channels}"
        )
    return logits.reshape(batch, channels, num_symbols, height, width)


def encode_residual(
    q: torch.Tensor,
    residual_logits: torch.Tensor,
    max_q: int,
    precision: int = 16,
    checkerboard_context: bool = False,
) -> tuple[bytes, np.ndarray]:
    q_array = torch.round(q.detach()).to(torch.int64).cpu().numpy()
    if q_array.ndim != 4:
        raise ValueError("q must have shape [batch, channels, height, width]")
    if q_array.min() < -max_q or q_array.max() > max_q:
        raise ValueError(f"residual symbols must be within [-{max_q}, {max_q}]")

    batch, channels, height, width = q_array.shape
    num_symbols = 2 * max_q + 1
    logits = reshape_residual_logits(residual_logits, channels, num_symbols)
    if tuple(logits.shape[:2]) != (batch, channels) or tuple(logits.shape[-2:]) != (height, width):
        raise ValueError("q and residual logits have incompatible shapes")
    cdfs = channel_cdfs_from_logits(logits, precision, checkerboard_context)

    symbols = (q_array + max_q).reshape(-1)
    symbols_per_cdf = height * width
    state = RANS_BYTE_L
    emitted = bytearray()
    scale = 1 << precision

    for position in range(symbols.size - 1, -1, -1):
        symbol = int(symbols[position])
        cdf_row = position // symbols_per_cdf
        if checkerboard_context:
            pixel_position = position % symbols_per_cdf
            row = pixel_position // width
            column = pixel_position % width
            cdf_row = cdf_row * 2 + (row + column) % 2
        cdf = cdfs[cdf_row]
        start = int(cdf[symbol])
        frequency = int(cdf[symbol + 1]) - start
        state_limit = ((RANS_BYTE_L >> precision) << 8) * frequency
        while state >= state_limit:
            emitted.append(state & 0xFF)
            state >>= 8
        state = (state // frequency) * scale + (state % frequency) + start

    return struct.pack("<I", state) + bytes(reversed(emitted)), cdfs


def decode_residual(
    payload: bytes,
    cdfs: np.ndarray,
    shape: tuple[int, int, int, int],
    max_q: int,
    precision: int = 16,
    checkerboard_context: bool = False,
) -> torch.Tensor:
    _validate_precision(precision)
    if len(payload) < 4:
        raise ValueError("rANS payload is truncated")
    batch, channels, height, width = shape
    expected_rows = batch * channels * (2 if checkerboard_context else 1)
    num_symbols = 2 * max_q + 1
    cdfs = np.asarray(cdfs, dtype=np.int32)
    if cdfs.shape != (expected_rows, num_symbols + 1):
        raise ValueError(
            f"expected CDF shape {(expected_rows, num_symbols + 1)}, received {cdfs.shape}"
        )
    if np.any(cdfs[:, 0] != 0) or np.any(cdfs[:, -1] != 1 << precision):
        raise ValueError("rANS CDF has invalid bounds")
    if np.any(np.diff(cdfs, axis=1) <= 0):
        raise ValueError("rANS CDF contains a zero-frequency symbol")

    state = struct.unpack("<I", payload[:4])[0]
    cursor = 4
    mask = (1 << precision) - 1
    symbol_count = batch * channels * height * width
    symbols_per_cdf = height * width
    decoded = np.empty(symbol_count, dtype=np.int16)

    for position in range(symbol_count):
        cdf_row = position // symbols_per_cdf
        if checkerboard_context:
            pixel_position = position % symbols_per_cdf
            row = pixel_position // width
            column = pixel_position % width
            cdf_row = cdf_row * 2 + (row + column) % 2
        cdf = cdfs[cdf_row]
        cumulative = state & mask
        symbol = int(np.searchsorted(cdf, cumulative, side="right") - 1)
        if symbol < 0 or symbol >= num_symbols:
            raise ValueError("rANS stream decoded an invalid symbol")
        start = int(cdf[symbol])
        frequency = int(cdf[symbol + 1]) - start
        state = frequency * (state >> precision) + cumulative - start
        while state < RANS_BYTE_L:
            if cursor >= len(payload):
                raise ValueError("rANS payload ended before all symbols were decoded")
            state = (state << 8) | payload[cursor]
            cursor += 1
        decoded[position] = symbol - max_q

    if cursor != len(payload):
        raise ValueError(f"rANS payload has {len(payload) - cursor} trailing bytes")
    return torch.from_numpy(decoded.reshape(shape).astype(np.float32))
