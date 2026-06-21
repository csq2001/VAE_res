from __future__ import annotations

from dataclasses import dataclass


BASES = ("A", "C", "G", "T")
BITS_TO_BASE = {"00": "A", "01": "C", "10": "G", "11": "T"}
BASE_TO_BITS = {v: k for k, v in BITS_TO_BASE.items()}


@dataclass
class DNAStats:
    gc: float
    max_homopolymer: int
    length: int


def bytes_to_bits(payload: bytes) -> str:
    return "".join(f"{byte:08b}" for byte in payload)


def bits_to_bytes(bits: str) -> bytes:
    if len(bits) % 8:
        bits = bits[: len(bits) - (len(bits) % 8)]
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def dna_stats(sequence: str) -> DNAStats:
    if not sequence:
        return DNAStats(0.0, 0, 0)
    gc = sum(1 for base in sequence if base in "GC") / len(sequence)
    max_run = 1
    run = 1
    for prev, cur in zip(sequence, sequence[1:]):
        run = run + 1 if prev == cur else 1
        max_run = max(max_run, run)
    return DNAStats(gc=gc, max_homopolymer=max_run, length=len(sequence))


class ConstrainedDNAEncoder:
    """Reversible stateful DNA mapper with a homopolymer constraint."""

    def __init__(self, max_homopolymer: int = 3) -> None:
        self.max_homopolymer = max_homopolymer

    def encode(self, payload: bytes) -> str:
        bits = bytes_to_bits(payload)
        sequence = []
        cursor = 0
        while cursor < len(bits):
            legal = self._legal_bases(sequence)
            if len(legal) >= 4:
                code = bits[cursor : cursor + 2].ljust(2, "0")
                sequence.append(legal[int(code, 2)])
                cursor += 2
            elif len(legal) == 3:
                if bits[cursor] == "0":
                    sequence.append(legal[0])
                    cursor += 1
                else:
                    code = bits[cursor : cursor + 2].ljust(2, "0")
                    sequence.append(legal[1] if code == "10" else legal[2])
                    cursor += 2
            elif len(legal) == 2:
                sequence.append(legal[int(bits[cursor])])
                cursor += 1
            else:
                sequence.append(legal[0])
        return "".join(sequence)

    def decode(self, sequence: str, bit_length: int) -> bytes:
        bits = []
        prefix = []
        for base in sequence:
            legal = self._legal_bases(prefix)
            if base not in legal:
                raise ValueError(f"Illegal base {base!r} for current DNA state")
            index = legal.index(base)
            if len(legal) >= 4:
                bits.append(f"{index:02b}")
            elif len(legal) == 3:
                bits.append("0" if index == 0 else ("10" if index == 1 else "11"))
            elif len(legal) == 2:
                bits.append(str(index))
            prefix.append(base)
            if sum(len(part) for part in bits) >= bit_length:
                break
        bitstream = "".join(bits)[:bit_length]
        return bits_to_bytes(bitstream)

    def decode_lossy_mapping(self, sequence: str) -> bytes:
        bits = "".join(BASE_TO_BITS[base] for base in sequence if base in BASE_TO_BITS)
        return bits_to_bytes(bits)

    def _would_exceed(self, sequence: list[str], base: str) -> bool:
        if len(sequence) < self.max_homopolymer:
            return False
        return all(prev == base for prev in sequence[-self.max_homopolymer :])

    def _legal_bases(self, sequence: list[str]) -> list[str]:
        return [base for base in BASES if not self._would_exceed(sequence, base)]
