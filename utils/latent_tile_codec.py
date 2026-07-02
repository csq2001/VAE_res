from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, replace

import numpy as np
import torch

from .ecc_rs import ReedSolomonError, encode_parity_shards, recover_data_shards
from .packet_format import FLAG_PARITY
from .packet_format import StreamType
from .residual_tile_codec import (
    decode_edit_byte_candidates,
    encode_edit_bytes,
)


_MAGIC = b"LTY1"
_VERSION = 1
_HEADER = struct.Struct("<4sBIIBBBBHHHIIBBBBH")


class LatentTileError(ValueError):
    pass


@dataclass(frozen=True)
class LatentTileConfig:
    spatial_size: int = 4
    channel_group: int = 32
    compression_level: int = 9
    decode_beam: int = 96
    rs_data_tiles: int = 8
    rs_parity_tiles: int = 4

    def __post_init__(self) -> None:
        if self.spatial_size < 1:
            raise ValueError("spatial_size must be positive")
        if self.channel_group < 1:
            raise ValueError("channel_group must be positive")
        if not 0 <= self.compression_level <= 9:
            raise ValueError("compression_level must be between 0 and 9")
        if self.decode_beam < 4:
            raise ValueError("decode_beam must be at least 4")
        if self.rs_data_tiles < 1 or self.rs_parity_tiles < 1:
            raise ValueError("latent RS tile counts must be positive")
        if self.rs_data_tiles + self.rs_parity_tiles > 255:
            raise ValueError("latent RS group cannot exceed 255 tiles")


@dataclass(frozen=True)
class LatentTileAddress:
    image_id: int
    block_id: int
    tile_row: int
    tile_col: int
    tile_height: int
    tile_width: int
    channel_start: int
    channel_count: int
    payload_length: int
    crc32: int = 0
    rs_group_id: int = 0
    rs_index: int = 0
    rs_data_shards: int = 1
    rs_parity_shards: int = 0
    flags: int = 0
    shard_bytes: int = 0

    @property
    def stream_type(self) -> StreamType:
        return StreamType.LATENT

    @property
    def is_parity(self) -> bool:
        return bool(self.flags & FLAG_PARITY)

    @property
    def packet_index(self) -> int:
        return (
            self.channel_start * 65536
            + self.tile_row * 256
            + self.tile_col
        )

    def pack(self, *, zero_crc: bool = False) -> bytes:
        return _HEADER.pack(
            _MAGIC,
            _VERSION,
            self.image_id,
            self.block_id,
            self.tile_row,
            self.tile_col,
            self.tile_height,
            self.tile_width,
            self.channel_start,
            self.channel_count,
            self.payload_length,
            0 if zero_crc else self.crc32,
            self.rs_group_id,
            self.rs_index,
            self.rs_data_shards,
            self.rs_parity_shards,
            self.flags,
            self.shard_bytes,
        )

    @classmethod
    def unpack(cls, raw: bytes) -> "LatentTileAddress":
        if len(raw) != _HEADER.size:
            raise LatentTileError("invalid latent tile address length")
        values = _HEADER.unpack(raw)
        if values[0] != _MAGIC or values[1] != _VERSION:
            raise LatentTileError("latent tile address magic/version mismatch")
        address = cls(
            image_id=values[2],
            block_id=values[3],
            tile_row=values[4],
            tile_col=values[5],
            tile_height=values[6],
            tile_width=values[7],
            channel_start=values[8],
            channel_count=values[9],
            payload_length=values[10],
            crc32=values[11],
            rs_group_id=values[12],
            rs_index=values[13],
            rs_data_shards=values[14],
            rs_parity_shards=values[15],
            flags=values[16],
            shard_bytes=values[17],
        )
        if (
            not address.tile_height
            or not address.tile_width
            or not address.channel_count
        ):
            raise LatentTileError("invalid latent tile shape")
        return address


@dataclass
class LatentTilePacket:
    header: LatentTileAddress
    payload: bytes
    dna_sequence: str | None
    is_erasure: bool = False
    error: str | None = None


@dataclass(frozen=True)
class LatentTileDecodeResult:
    y_hat: torch.Tensor
    corrected_codewords: int


@dataclass(frozen=True)
class LatentTileRecoveryResult:
    tiles: dict[int, LatentTileDecodeResult]
    valid_packets: int
    erasures: int
    corrected_tiles: int
    corrected_codewords: int
    rs_recovered_tiles: int
    errors: tuple[str, ...]


def _address_matches(
    actual: LatentTileAddress,
    expected: LatentTileAddress,
) -> bool:
    return (
        actual.image_id == expected.image_id
        and actual.block_id == expected.block_id
        and actual.tile_row == expected.tile_row
        and actual.tile_col == expected.tile_col
        and actual.tile_height == expected.tile_height
        and actual.tile_width == expected.tile_width
        and actual.channel_start == expected.channel_start
        and actual.channel_count == expected.channel_count
        and actual.rs_group_id == expected.rs_group_id
        and actual.rs_index == expected.rs_index
        and actual.is_parity == expected.is_parity
    )


def _frame_packet(packet: LatentTilePacket) -> LatentTilePacket:
    header = replace(packet.header, crc32=0)
    checksum = zlib.crc32(header.pack(zero_crc=True) + packet.payload) & 0xFFFFFFFF
    packet.header = replace(header, crc32=checksum)
    packet.dna_sequence = encode_edit_bytes(packet.header.pack() + packet.payload)
    return packet


def encode_latent_tile(
    y_hat: torch.Tensor,
    address: LatentTileAddress,
    config: LatentTileConfig | None = None,
) -> LatentTilePacket:
    config = config or LatentTileConfig()
    if y_hat.ndim != 4 or y_hat.shape[0] != 1:
        raise ValueError("latent tile must have shape [1, C, H, W]")
    expected_shape = (
        address.channel_count,
        address.tile_height,
        address.tile_width,
    )
    if tuple(y_hat.shape[1:]) != expected_shape:
        raise ValueError(f"latent tile shape does not match address: {expected_shape}")
    rounded = torch.round(y_hat).to(torch.int16).cpu()
    if rounded.min().item() < -128 or rounded.max().item() > 127:
        dtype = np.int16
        dtype_flag = b"\x02"
    else:
        dtype = np.int8
        dtype_flag = b"\x01"
    payload = dtype_flag + zlib.compress(
        rounded.numpy().astype(dtype, copy=False).tobytes(),
        level=config.compression_level,
    )
    address = replace(address, payload_length=len(payload), crc32=0)
    return _frame_packet(LatentTilePacket(
        header=address,
        payload=payload,
        dna_sequence=None,
    ))


def add_latent_rs_parity(
    data_packets: list[LatentTilePacket],
    config: LatentTileConfig | None = None,
) -> list[LatentTilePacket]:
    config = config or LatentTileConfig()
    output: list[LatentTilePacket] = []
    for group_id, start in enumerate(
        range(0, len(data_packets), config.rs_data_tiles)
    ):
        group = data_packets[start : start + config.rs_data_tiles]
        data_count = len(group)
        shard_bytes = max(len(packet.payload) for packet in group)
        padded = [
            packet.payload.ljust(shard_bytes, b"\x00") for packet in group
        ]
        parity = encode_parity_shards(padded, config.rs_parity_tiles)
        for index, packet in enumerate(group):
            packet.header = replace(
                packet.header,
                rs_group_id=group_id,
                rs_index=index,
                rs_data_shards=data_count,
                rs_parity_shards=config.rs_parity_tiles,
                shard_bytes=shard_bytes,
            )
            output.append(_frame_packet(packet))
        template = group[0].header
        for parity_index, payload in enumerate(parity):
            header = replace(
                template,
                tile_row=0,
                tile_col=0,
                tile_height=1,
                tile_width=1,
                channel_start=0,
                channel_count=1,
                payload_length=len(payload),
                crc32=0,
                rs_index=data_count + parity_index,
                flags=FLAG_PARITY,
            )
            output.append(
                _frame_packet(
                    LatentTilePacket(header, payload, dna_sequence=None)
                )
            )
    return output


def _decode_packet_payload(
    packet: LatentTilePacket,
    config: LatentTileConfig,
) -> tuple[LatentTileAddress, bytes, int]:
    if packet.dna_sequence is None:
        raise LatentTileError("latent tile sequence is missing")
    expected_bytes = _HEADER.size + packet.header.payload_length
    candidates = decode_edit_byte_candidates(
        packet.dna_sequence,
        expected_bytes,
        beam=config.decode_beam,
    )
    errors: list[str] = []
    for raw, edits in candidates:
        try:
            address = LatentTileAddress.unpack(raw[: _HEADER.size])
            if not _address_matches(address, packet.header):
                raise LatentTileError("decoded latent tile address does not match")
            payload = raw[_HEADER.size :]
            expected_crc = zlib.crc32(
                address.pack(zero_crc=True) + payload
            ) & 0xFFFFFFFF
            if expected_crc != address.crc32:
                raise LatentTileError("latent tile CRC mismatch")
            return address, payload, edits
        except (LatentTileError, struct.error) as exc:
            errors.append(str(exc))
    detail = errors[0] if errors else "no CRC-valid edit-code path"
    raise LatentTileError(f"latent tile cannot be corrected: {detail}")


def _payload_to_result(
    address: LatentTileAddress,
    payload: bytes,
    corrected_codewords: int,
) -> LatentTileDecodeResult:
    if not payload or payload[0] not in (1, 2):
        raise LatentTileError("invalid latent tile dtype")
    dtype = np.int8 if payload[0] == 1 else np.int16
    try:
        values = np.frombuffer(zlib.decompress(payload[1:]), dtype=dtype)
    except zlib.error as exc:
        raise LatentTileError(f"latent tile zlib failure: {exc}") from exc
    expected_values = (
        address.channel_count * address.tile_height * address.tile_width
    )
    if values.size != expected_values:
        raise LatentTileError("latent tile payload shape mismatch")
    y_hat = torch.from_numpy(values.copy()).to(torch.float32).reshape(
        1,
        address.channel_count,
        address.tile_height,
        address.tile_width,
    )
    return LatentTileDecodeResult(y_hat, corrected_codewords)


def decode_latent_tile(
    packet: LatentTilePacket,
    config: LatentTileConfig | None = None,
) -> LatentTileDecodeResult:
    config = config or LatentTileConfig()
    address, payload, edits = _decode_packet_payload(packet, config)
    if address.is_parity:
        raise LatentTileError("parity shard is not a latent data tile")
    return _payload_to_result(address, payload, edits)


def recover_latent_rs_tiles(
    packets: list[LatentTilePacket],
    config: LatentTileConfig | None = None,
) -> LatentTileRecoveryResult:
    config = config or LatentTileConfig()
    valid: dict[int, tuple[LatentTileAddress, bytes, int]] = {}
    errors: list[str] = []
    for packet in packets:
        try:
            valid[id(packet)] = _decode_packet_payload(packet, config)
            packet.is_erasure = False
            packet.error = None
        except LatentTileError as exc:
            packet.is_erasure = True
            packet.error = str(exc)
            errors.append(
                f"latent group {packet.header.rs_group_id} "
                f"shard {packet.header.rs_index}: {exc}"
            )

    groups: dict[int, list[LatentTilePacket]] = {}
    for packet in packets:
        groups.setdefault(packet.header.rs_group_id, []).append(packet)

    results: dict[int, LatentTileDecodeResult] = {}
    corrected_tiles = corrected_codewords = rs_recovered_tiles = 0
    for group in groups.values():
        group.sort(key=lambda packet: packet.header.rs_index)
        template = group[0].header
        available = {
            packet.header.rs_index: decoded[1].ljust(
                template.shard_bytes, b"\x00"
            )
            for packet in group
            if (decoded := valid.get(id(packet))) is not None
        }
        recovered: list[bytes] | None
        try:
            recovered = recover_data_shards(
                available,
                template.rs_data_shards,
                template.rs_parity_shards,
            )
        except (ReedSolomonError, ValueError):
            recovered = None

        for packet in group:
            if packet.header.is_parity:
                continue
            decoded = valid.get(id(packet))
            try:
                if decoded is not None:
                    address, payload, edits = decoded
                    corrected_tiles += int(edits > 0)
                    corrected_codewords += edits
                elif recovered is not None:
                    address = packet.header
                    payload = recovered[address.rs_index][
                        : address.payload_length
                    ]
                    edits = 0
                    rs_recovered_tiles += 1
                else:
                    continue
                results[address.packet_index] = _payload_to_result(
                    address, payload, edits
                )
            except LatentTileError as exc:
                errors.append(str(exc))

    return LatentTileRecoveryResult(
        tiles=results,
        valid_packets=len(valid),
        erasures=len(packets) - len(valid),
        corrected_tiles=corrected_tiles,
        corrected_codewords=corrected_codewords,
        rs_recovered_tiles=rs_recovered_tiles,
        errors=tuple(errors),
    )
