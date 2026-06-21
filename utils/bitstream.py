from __future__ import annotations

import io
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dna_constraints import ConstrainedDNAEncoder, dna_stats


@dataclass
class PackedCodecStream:
    payload: bytes
    dna: str
    metadata: dict


def pack_tensors(y_hat: torch.Tensor, q: torch.Tensor, metadata: dict) -> bytes:
    buffer = io.BytesIO()
    y_int = torch.round(y_hat).cpu()
    if y_int.min().item() >= -128 and y_int.max().item() <= 127:
        y_array = y_int.to(torch.int8).numpy()
        metadata = {**metadata, "y_dtype": "int8"}
    else:
        y_array = y_int.to(torch.int16).numpy()
        metadata = {**metadata, "y_dtype": "int16"}
    meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    np.savez_compressed(
        buffer,
        metadata=np.frombuffer(meta_bytes, dtype=np.uint8),
        y_hat=y_array,
        q=torch.round(q).cpu().clamp(-128, 127).to(torch.int8).numpy(),
    )
    return zlib.compress(buffer.getvalue(), level=9)


def unpack_tensors(payload: bytes) -> tuple[torch.Tensor, torch.Tensor, dict]:
    raw = zlib.decompress(payload)
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        metadata = json.loads(bytes(data["metadata"].tolist()).decode("utf-8"))
        y_hat = torch.from_numpy(data["y_hat"].astype(np.float32))
        q = torch.from_numpy(data["q"].astype(np.float32))
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
