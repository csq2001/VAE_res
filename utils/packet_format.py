from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, replace
from enum import IntEnum


PACKET_MAGIC = b"VDP1"
PACKET_VERSION = 1
FLAG_PARITY = 1
HEADER_STRUCT = struct.Struct(">4sBBBBIIHHHHIIIIIIIHHHHI")


class PacketFormatError(ValueError):
    pass


class PacketCRCError(PacketFormatError):
    pass


class StreamType(IntEnum):
    LATENT = 0
    RESIDUAL = 1

    @property
    def label(self) -> str:
        return "latent" if self is StreamType.LATENT else "residual"

    @classmethod
    def from_label(cls, label: str) -> "StreamType":
        if label == "latent":
            return cls.LATENT
        if label == "residual":
            return cls.RESIDUAL
        raise ValueError(f"unsupported stream type: {label}")


@dataclass(frozen=True)
class PacketHeader:
    stream_type: StreamType
    image_id: int
    block_id: int
    row_id: int
    col_id: int
    block_size: int
    tau: int
    image_height: int
    image_width: int
    payload_length: int
    stream_length: int
    packet_index: int
    total_packets: int
    rs_group_id: int
    rs_index: int
    rs_data_shards: int
    rs_parity_shards: int
    crc32: int = 0
    flags: int = 0

    @property
    def is_parity(self) -> bool:
        return bool(self.flags & FLAG_PARITY)

    def pack(self, zero_crc: bool = False) -> bytes:
        return HEADER_STRUCT.pack(
            PACKET_MAGIC,
            PACKET_VERSION,
            int(self.stream_type),
            self.flags,
            0,
            self.image_id,
            self.block_id,
            self.row_id,
            self.col_id,
            self.block_size,
            self.tau,
            self.image_height,
            self.image_width,
            self.payload_length,
            self.stream_length,
            self.packet_index,
            self.total_packets,
            self.rs_group_id,
            self.rs_index,
            self.rs_data_shards,
            self.rs_parity_shards,
            0,
            0 if zero_crc else self.crc32,
        )

    def with_crc(self, payload: bytes) -> "PacketHeader":
        checksum = zlib.crc32(self.pack(zero_crc=True) + payload) & 0xFFFFFFFF
        return replace(self, crc32=checksum)

    @classmethod
    def unpack(cls, data: bytes) -> "PacketHeader":
        if len(data) < HEADER_STRUCT.size:
            raise PacketFormatError("packet header is truncated")
        values = HEADER_STRUCT.unpack(data[: HEADER_STRUCT.size])
        (
            magic,
            version,
            stream_type,
            flags,
            _reserved,
            image_id,
            block_id,
            row_id,
            col_id,
            block_size,
            tau,
            image_height,
            image_width,
            payload_length,
            stream_length,
            packet_index,
            total_packets,
            rs_group_id,
            rs_index,
            rs_data_shards,
            rs_parity_shards,
            _reserved2,
            crc32,
        ) = values
        if magic != PACKET_MAGIC:
            raise PacketFormatError("invalid packet magic")
        if version != PACKET_VERSION:
            raise PacketFormatError(f"unsupported packet version: {version}")
        try:
            decoded_stream_type = StreamType(stream_type)
        except ValueError as exc:
            raise PacketFormatError(f"invalid stream type: {stream_type}") from exc
        return cls(
            stream_type=decoded_stream_type,
            image_id=image_id,
            block_id=block_id,
            row_id=row_id,
            col_id=col_id,
            block_size=block_size,
            tau=tau,
            image_height=image_height,
            image_width=image_width,
            payload_length=payload_length,
            stream_length=stream_length,
            packet_index=packet_index,
            total_packets=total_packets,
            rs_group_id=rs_group_id,
            rs_index=rs_index,
            rs_data_shards=rs_data_shards,
            rs_parity_shards=rs_parity_shards,
            crc32=crc32,
            flags=flags,
        )


@dataclass
class DNAPacket:
    header: PacketHeader
    payload: bytes
    dna_sequence: str | None = None
    is_erasure: bool = False
    error: str | None = None

    def serialize(self) -> bytes:
        header = self.header.with_crc(self.payload)
        self.header = header
        return header.pack() + self.payload


@dataclass(frozen=True)
class PacketConfig:
    payload_bytes: int = 512

    def __post_init__(self) -> None:
        if self.payload_bytes < 16:
            raise ValueError("payload_bytes must be at least 16")


def parse_packet(data: bytes) -> DNAPacket:
    header = PacketHeader.unpack(data)
    end = HEADER_STRUCT.size + header.payload_length
    if len(data) < end:
        raise PacketFormatError("packet payload is truncated")
    payload = data[HEADER_STRUCT.size:end]
    expected = header.with_crc(payload).crc32
    if expected != header.crc32:
        raise PacketCRCError(
            f"CRC mismatch: expected {expected:08x}, received {header.crc32:08x}"
        )
    return DNAPacket(header=header, payload=payload)


def normalize_image_id(image_id: int | str) -> int:
    if isinstance(image_id, int):
        if not 0 <= image_id <= 0xFFFFFFFF:
            raise ValueError("integer image_id must fit uint32")
        return image_id
    return zlib.crc32(image_id.encode("utf-8")) & 0xFFFFFFFF
