from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, replace

import numpy as np
import torch

from .packet_format import StreamType


# A greedy length-7 quaternary codebook with minimum Levenshtein distance 3.
# Consequently, every codeword with one substitution, insertion, or deletion
# has a unique nearest codeword.  It is deliberately stored as a constant so
# importing this module does not spend several seconds rebuilding the codebook.
_CODEWORDS = (
    "AACAACA AACACGC AACAGAG AACCAAT AACCGGA AACCTCC AACGATC AACGCCG AACGTAA "
    "AACTCTA AACTTGG AAGAAGG AAGACAA AAGAGCC AAGCCTT AAGCTAG AAGGACT AAGGTGC "
    "AAGTGGT AAGTTCA AATACCT AATCATG AATCCAC AATGAGA AATTGTC ACAACCG ACAAGAA "
    "ACAATGC ACACACT ACACGTG ACAGCGA ACAGTAT ACATAAC ACATCTT ACCAGGT ACCATTA "
    "ACCGAAG ACCTAGA ACCTGCG ACGAATT ACGCAGC ACGGCTC ACGTCGG ACGTGTA ACTATAG "
    "ACTCCGT ACTCGCA ACTGACC ACTGGTT ACTTCAA AGAACAC AGAAGCT AGACGGC AGACTCG "
    "AGAGATA AGAGGAG AGATTGA AGCGAGT AGCGTTG AGCTCAT AGGATTC AGGCACA AGGCTGT "
    "AGGTAAG AGTACGA AGTAGTG AGTCTAA AGTGTCT AGTTAGC AGTTCCG ATAACGT ATACCAG "
    "ATACTTC ATAGGCA ATCAGTC ATCGCTT ATCTACT ATGCCGA ATGGAAC ATGGTCG ATTATCC "
    "ATTCGAT ATTCTGG ATTGCGC CAACAGT CAACGCA CAACTAC CAAGACG CAAGGTT CAAGTGA "
    "CAATCAA CAATTCT CACATCG CACCGTC CACTATT CAGAATC CAGATGT CAGCTTA CAGGCAG "
    "CAGTCGC CATACGG CATAGAT CATCACC CATGCTC CATTAAG CATTGGA CCAAGCC CCAATTG "
    "CCACAAG CCACCTA CCATACA CCGACAC CCGAGGA CCGCGAT CCGGTAA CCGTAGT CCGTTCC "
    "CCTAACG CCTCTCT CCTGAAT CCTGCCA CCTGTGC CCTTATC CGACCAT CGAGAGC CGAGCTG "
    "CGCAACT CGCAGAA CGCCTAG CGCGTCA CGCTGGT CGGACTA CGGCATT CGGTCCT CGGTGAC "
    "CGGTTGG CGTATAC CTAGCAC CTATCCG CTATGGC CTCAAGA CTCACAT CTCCAAC CTCGATG "
    "CTCGGCT CTCTGTA CTGATCA CTGCAGG CTGTTAT CTTCCTT GAACATA GAAGAAC GAAGCCA "
    "GAAGTTG GAATAGG GAATCTC GAATGAT GACAAGT GACCACG GACGTCT GACTGTG GACTTAC "
    "GAGCGAA GAGTACC GATAGTA GATATGC GATCGGT GATGGCC GCACGAC GCACTCA GCAGGCT "
    "GCATTGT GCCAATC GCCAGCA GCCATGG GCGATAT GCGCTTC GCGGAGA GCTACTG GCTTCGC "
    "GCTTGAG GGAACGG GGAAGTC GGAATAA GGACACC GGATGCA GGCCGAT GGCCTTA GGTAACA "
    "GGTCAGG GGTGATT GGTGGAA GGTTCTA GTAAGAG GTACCGC GTACTAT GTATTCC GTCCATT "
    "GTCGACA GTCGTAG GTCTCGA GTGAACT GTGAGGC GTGCCTG GTGGCAT GTGGTTA GTGTGCG "
    "GTTACAA GTTGGTG TAACCTG TAAGCAT TAAGTCC TAATGCG TACATGA TACGAGG TACGGAC "
    "TAGAGTT TAGATAC TATCTCA TATGTAG TCACGGT TCAGTGG TCATATG TCCATCC TCCGACT "
    "TCCGCAA TCCTCGT TCCTTAG TCGCACG TCGCTGA TCGTCCA TCTAAGC TCTCATA TCTGGCG "
    "TGAAGGA TGATCGC TGCAATA TGCGGTT TGCGTGC TGCTACC TGGACAG TGGATCT TGGCCTC "
    "TGGTAGA TGGTGTG TGTATGG TGTCAAC TGTTGCT TTAACCA TTACGCC TTAGAAG TTCACTG "
    "TTCCGAG TTGCGTA TTGGAGT TTGTATC"
).split()
_BASES = "ACGT"
_BITS_TO_BASE = ("A", "C", "G", "T")
_BASE_TO_VALUE = {base: value for value, base in enumerate(_BITS_TO_BASE)}
_MAGIC = b"RTQ1"
_VERSION = 1
_HEADER = struct.Struct("<4sBIIBBBBBBHBI")


class ResidualTileError(ValueError):
    pass


def _build_edit_lookup() -> dict[str, tuple[int, int] | None]:
    lookup: dict[str, tuple[int, int] | None] = {}

    def add(sequence: str, byte: int, edits: int) -> None:
        value = (byte, edits)
        previous = lookup.get(sequence)
        if previous is None and sequence in lookup:
            return
        if previous is not None and previous[0] != byte:
            lookup[sequence] = None
        else:
            lookup[sequence] = value

    for byte, word in enumerate(_CODEWORDS):
        add(word, byte, 0)
        for index in range(len(word)):
            add(word[:index] + word[index + 1 :], byte, 1)
            for base in _BASES:
                if base != word[index]:
                    add(word[:index] + base + word[index + 1 :], byte, 1)
        for index in range(len(word) + 1):
            for base in _BASES:
                add(word[:index] + base + word[index:], byte, 1)
    return lookup


_EDIT_LOOKUP = _build_edit_lookup()


@dataclass(frozen=True)
class ResidualTileConfig:
    tile_size: int = 16
    compression_level: int = 9
    decode_beam: int = 96
    protect_mean_abs: float = 0.3

    def __post_init__(self) -> None:
        if self.tile_size < 1:
            raise ValueError("tile_size must be positive")
        if not 0 <= self.compression_level <= 9:
            raise ValueError("compression_level must be between 0 and 9")
        if self.decode_beam < 4:
            raise ValueError("decode_beam must be at least 4")
        if self.protect_mean_abs < 0:
            raise ValueError("protect_mean_abs cannot be negative")


@dataclass(frozen=True)
class ResidualTileAddress:
    image_id: int
    block_id: int
    tile_row: int
    tile_col: int
    tile_height: int
    tile_width: int
    channels: int
    tau: int
    payload_length: int
    crc32: int = 0
    payload_mode: int = 0

    @property
    def stream_type(self) -> StreamType:
        return StreamType.RESIDUAL

    @property
    def flags(self) -> int:
        return 0

    @property
    def is_parity(self) -> bool:
        return False

    @property
    def packet_index(self) -> int:
        return self.tile_row * 256 + self.tile_col

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
            self.channels,
            self.tau,
            self.payload_length,
            self.payload_mode,
            0 if zero_crc else self.crc32,
        )

    @classmethod
    def unpack(cls, raw: bytes) -> "ResidualTileAddress":
        if len(raw) != _HEADER.size:
            raise ResidualTileError("invalid residual tile address length")
        (
            magic,
            version,
            image_id,
            block_id,
            tile_row,
            tile_col,
            tile_height,
            tile_width,
            channels,
            tau,
            payload_length,
            payload_mode,
            crc32,
        ) = _HEADER.unpack(raw)
        if magic != _MAGIC or version != _VERSION:
            raise ResidualTileError("residual tile address magic/version mismatch")
        if not channels or not tile_height or not tile_width:
            raise ResidualTileError("invalid residual tile shape")
        return cls(
            image_id=image_id,
            block_id=block_id,
            tile_row=tile_row,
            tile_col=tile_col,
            tile_height=tile_height,
            tile_width=tile_width,
            channels=channels,
            tau=tau,
            payload_length=payload_length,
            crc32=crc32,
            payload_mode=payload_mode,
        )


@dataclass
class ResidualTilePacket:
    header: ResidualTileAddress
    payload: bytes
    dna_sequence: str | None
    is_erasure: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ResidualTileDecodeResult:
    q: torch.Tensor
    corrected_codewords: int


@dataclass(frozen=True)
class _DecodeState:
    position: int
    edits: int
    raw: bytes


def _encode_bytes(raw: bytes) -> str:
    return "".join(_CODEWORDS[value] for value in raw)


def encode_edit_bytes(raw: bytes) -> str:
    """Encode bytes as independent single-edit-correcting DNA codewords."""
    return _encode_bytes(raw)


def _encode_compact_bytes(raw: bytes) -> str:
    sequence: list[str] = []
    for value in raw:
        sequence.extend(
            (
                _BITS_TO_BASE[(value >> 6) & 3],
                _BITS_TO_BASE[(value >> 4) & 3],
                _BITS_TO_BASE[(value >> 2) & 3],
                _BITS_TO_BASE[value & 3],
            )
        )
    return "".join(sequence)


def _decode_compact_bytes(sequence: str) -> bytes:
    if len(sequence) % 4:
        raise ResidualTileError("compact residual payload length is not byte aligned")
    try:
        values = [_BASE_TO_VALUE[base] for base in sequence]
    except KeyError as exc:
        raise ResidualTileError("compact residual payload contains invalid base") from exc
    return bytes(
        (values[index] << 6)
        | (values[index + 1] << 4)
        | (values[index + 2] << 2)
        | values[index + 3]
        for index in range(0, len(values), 4)
    )


def _advance_states(
    sequence: str,
    states: list[_DecodeState],
    count: int,
    beam: int,
) -> list[_DecodeState]:
    current = states
    for _ in range(count):
        candidates: dict[int, _DecodeState] = {}
        for state in current:
            for length in (6, 7, 8):
                end = state.position + length
                if end > len(sequence):
                    continue
                decoded = _EDIT_LOOKUP.get(sequence[state.position:end])
                if decoded is None:
                    continue
                value, edits = decoded
                candidate = _DecodeState(
                    position=end,
                    edits=state.edits + edits,
                    raw=state.raw + bytes((value,)),
                )
                previous = candidates.get(end)
                if previous is None or candidate.edits < previous.edits:
                    candidates[end] = candidate
        if not candidates:
            return []
        current = sorted(
            candidates.values(),
            key=lambda item: (item.edits, abs(item.position - len(item.raw) * 7)),
        )[:beam]
    return current


def decode_edit_byte_candidates(
    sequence: str,
    expected_bytes: int,
    *,
    beam: int = 96,
) -> list[tuple[bytes, int]]:
    """Return complete fixed-length byte decodings ordered by edit count."""
    states = _advance_states(
        sequence,
        [_DecodeState(position=0, edits=0, raw=b"")],
        expected_bytes,
        beam,
    )
    complete = [state for state in states if state.position == len(sequence)]
    complete.sort(key=lambda state: state.edits)
    return [(state.raw, state.edits) for state in complete]


def _address_matches(
    actual: ResidualTileAddress,
    expected: ResidualTileAddress,
) -> bool:
    return (
        actual.image_id == expected.image_id
        and actual.block_id == expected.block_id
        and actual.tile_row == expected.tile_row
        and actual.tile_col == expected.tile_col
        and actual.tile_height == expected.tile_height
        and actual.tile_width == expected.tile_width
        and actual.channels == expected.channels
        and actual.tau == expected.tau
    )


def encode_residual_tile(
    q: torch.Tensor,
    address: ResidualTileAddress,
    config: ResidualTileConfig | None = None,
) -> ResidualTilePacket:
    config = config or ResidualTileConfig()
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("residual tile must have shape [1, C, H, W]")
    expected_shape = (
        address.channels,
        address.tile_height,
        address.tile_width,
    )
    if tuple(q.shape[1:]) != expected_shape:
        raise ValueError(f"residual tile shape does not match address: {expected_shape}")
    rounded = torch.round(q).to(torch.int16).cpu()
    if rounded.min().item() < -128 or rounded.max().item() > 127:
        raise ValueError("residual q values must fit int8")
    payload = zlib.compress(
        rounded.to(torch.int8).numpy().tobytes(),
        level=config.compression_level,
    )
    payload_mode = int(
        float(rounded.abs().to(torch.float32).mean().item())
        >= config.protect_mean_abs
    )
    address = replace(
        address,
        payload_length=len(payload),
        crc32=0,
        payload_mode=payload_mode,
    )
    checksum = zlib.crc32(address.pack(zero_crc=True) + payload) & 0xFFFFFFFF
    address = replace(address, crc32=checksum)
    return ResidualTilePacket(
        header=address,
        payload=payload,
        dna_sequence=(
            encode_edit_bytes(address.pack())
            + (
                encode_edit_bytes(payload)
                if payload_mode
                else _encode_compact_bytes(payload)
            )
        ),
    )


def decode_residual_tile(
    packet: ResidualTilePacket,
    config: ResidualTileConfig | None = None,
) -> ResidualTileDecodeResult:
    config = config or ResidualTileConfig()
    if packet.dna_sequence is None:
        raise ResidualTileError("residual tile sequence is missing")
    sequence = packet.dna_sequence
    header_states = _advance_states(
        sequence,
        [_DecodeState(position=0, edits=0, raw=b"")],
        _HEADER.size,
        config.decode_beam,
    )
    errors: list[str] = []
    for state in header_states:
        try:
            address = ResidualTileAddress.unpack(state.raw)
            if not _address_matches(address, packet.header):
                raise ResidualTileError("decoded residual tile address does not match")
            payload_dna = sequence[state.position :]
            if address.payload_mode:
                payload_candidates = decode_edit_byte_candidates(
                    payload_dna,
                    address.payload_length,
                    beam=config.decode_beam,
                )
                if not payload_candidates:
                    raise ResidualTileError(
                        "protected residual payload cannot be corrected"
                    )
                payload, payload_edits = payload_candidates[0]
            else:
                if len(payload_dna) != address.payload_length * 4:
                    raise ResidualTileError(
                        "compact residual payload has an insertion/deletion"
                    )
                payload = _decode_compact_bytes(payload_dna)
                payload_edits = 0
            expected_crc = zlib.crc32(
                address.pack(zero_crc=True) + payload
            ) & 0xFFFFFFFF
            if expected_crc != address.crc32:
                raise ResidualTileError("compact residual payload CRC mismatch")
            raw_q = zlib.decompress(payload)
            expected_values = (
                address.channels * address.tile_height * address.tile_width
            )
            values = np.frombuffer(raw_q, dtype=np.int8)
            if values.size != expected_values:
                raise ResidualTileError("residual tile payload shape mismatch")
            q = torch.from_numpy(values.copy()).to(torch.float32).reshape(
                1,
                address.channels,
                address.tile_height,
                address.tile_width,
            )
            return ResidualTileDecodeResult(
                q=q,
                corrected_codewords=state.edits + payload_edits,
            )
        except (ResidualTileError, zlib.error, struct.error) as exc:
            errors.append(str(exc))
    detail = errors[0] if errors else "no CRC-valid edit-code path"
    raise ResidualTileError(f"residual tile cannot be corrected: {detail}")
