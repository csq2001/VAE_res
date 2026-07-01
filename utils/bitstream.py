from __future__ import annotations

import io
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dna_constraints import ConstrainedDNAEncoder, dna_stats
from .rans_codec import decode_residual, encode_residual


@dataclass
class PackedCodecStream:
    payload: bytes
    dna: str
    metadata: dict


def pack_tensors(
    y_hat: torch.Tensor,
    q: torch.Tensor,
    metadata: dict,
    residual_codec: str = "zlib",
    residual_logits: torch.Tensor | None = None,
    max_q: int = 64,
    rans_precision: int = 16,
    checkerboard_context: bool = False,
) -> bytes:
    buffer = io.BytesIO()
    y_int = torch.round(y_hat).cpu()
    if y_int.min().item() >= -128 and y_int.max().item() <= 127:
        y_array = y_int.to(torch.int8).numpy()
        metadata = {**metadata, "y_dtype": "int8"}
    else:
        y_array = y_int.to(torch.int16).numpy()
        metadata = {**metadata, "y_dtype": "int16"}
    metadata = {**metadata, "stream_version": 2, "residual_codec": residual_codec}
    if residual_codec == "zlib":
        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        np.savez_compressed(
            buffer,
            metadata=np.frombuffer(meta_bytes, dtype=np.uint8),
            y_hat=y_array,
            q=torch.round(q).cpu().clamp(-128, 127).to(torch.int8).numpy(),
        )
    elif residual_codec == "rans":
        if residual_logits is None:
            raise ValueError("residual_logits are required for rANS encoding")
        rans_payload, cdfs = encode_residual(
            q,
            residual_logits,
            max_q,
            rans_precision,
            checkerboard_context,
        )
        metadata = {
            **metadata,
            "max_q": max_q,
            "rans_precision": rans_precision,
            "rans_payload_bytes": len(rans_payload),
            "checkerboard_context": checkerboard_context,
        }
        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        np.savez_compressed(
            buffer,
            metadata=np.frombuffer(meta_bytes, dtype=np.uint8),
            y_hat=y_array,
            residual_rans=np.frombuffer(rans_payload, dtype=np.uint8),
            residual_cdfs=cdfs,
            q_shape=np.asarray(q.shape, dtype=np.int32),
        )
    else:
        raise ValueError(f"unsupported residual codec: {residual_codec}")
    return zlib.compress(buffer.getvalue(), level=9)


def unpack_tensors(payload: bytes) -> tuple[torch.Tensor, torch.Tensor, dict]:
    raw = zlib.decompress(payload)
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        metadata = json.loads(bytes(data["metadata"].tolist()).decode("utf-8"))
        y_hat = torch.from_numpy(data["y_hat"].astype(np.float32))
        residual_codec = metadata.get("residual_codec", "zlib")
        if residual_codec == "zlib":
            q = torch.from_numpy(data["q"].astype(np.float32))
        elif residual_codec == "rans":
            rans_payload = data["residual_rans"].astype(np.uint8, copy=False).tobytes()
            cdfs = data["residual_cdfs"].astype(np.int32, copy=False)
            q_shape = tuple(int(value) for value in data["q_shape"].tolist())
            q = decode_residual(
                rans_payload,
                cdfs,
                q_shape,
                int(metadata["max_q"]),
                int(metadata["rans_precision"]),
                bool(metadata.get("checkerboard_context", False)),
            )
        else:
            raise ValueError(f"unsupported residual codec in stream: {residual_codec}")
    return y_hat, q, metadata


def payload_to_dna(payload: bytes, max_homopolymer: int = 3) -> PackedCodecStream:
    encoder = ConstrainedDNAEncoder(max_homopolymer=max_homopolymer)
    dna = encoder.encode(payload)
    stats = dna_stats(dna)
    return PackedCodecStream(
        payload=payload,
        dna=dna,
        metadata={
            "payload_bytes": len(payload),
            "payload_bits": len(payload) * 8,
            "dna_nt": len(dna),
            "bits_per_nt": (len(payload) * 8) / max(len(dna), 1),
            "gc": stats.gc,
            "max_homopolymer": stats.max_homopolymer,
        },
    )


def write_dna_fasta(sequence: str, path: str | Path, line_width: int = 120) -> None:
    path = Path(path)
    with path.open("w", encoding="ascii") as handle:
        handle.write(">vae_residual_codec\n")
        for i in range(0, len(sequence), line_width):
            handle.write(sequence[i : i + line_width] + "\n")
