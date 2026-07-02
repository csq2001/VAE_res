from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch

from .dna_constraints import ConstrainedDNAEncoder
from .ecc_rs import RSConfig, encode_parity_shards, recover_data_shards
from .marker_code import (
    MarkerConfig,
    MarkerDecodeError,
    deframe_payload_dna,
    frame_payload_dna,
)
from .packet_format import (
    FLAG_PARITY,
    DNAPacket,
    PacketConfig,
    PacketFormatError,
    PacketHeader,
    StreamType,
    parse_packet,
)
from .rans_codec import decode_residual, encode_residual


@dataclass(frozen=True)
class StreamRecoveryReport:
    total_packets: int
    valid_packets: int
    erasures: int
    recovered_data_packets: int
    errors: tuple[str, ...]


def pack_latent_stream(y_hat: torch.Tensor) -> bytes:
    rounded = torch.round(y_hat).cpu()
    dtype = torch.int8 if rounded.min().item() >= -128 and rounded.max().item() <= 127 else torch.int16
    buffer = io.BytesIO()
    np.savez_compressed(buffer, y_hat=rounded.to(dtype).numpy())
    return buffer.getvalue()


def unpack_latent_stream(payload: bytes) -> torch.Tensor:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return torch.from_numpy(data["y_hat"].astype(np.float32))


def pack_residual_stream(
    q: torch.Tensor,
    residual_logits: torch.Tensor | None = None,
    codec: str = "rans",
    max_q: int = 64,
    rans_precision: int = 16,
    checkerboard_context: bool = False,
) -> bytes:
    buffer = io.BytesIO()
    if codec == "zlib":
        np.savez_compressed(
            buffer,
            codec=np.asarray([0], dtype=np.uint8),
            q=torch.round(q).cpu().clamp(-128, 127).to(torch.int8).numpy(),
        )
    elif codec == "rans":
        if residual_logits is None:
            raise ValueError("residual_logits are required for rANS residual streams")
        rans_payload, cdfs = encode_residual(
            q,
            residual_logits,
            max_q,
            rans_precision,
            checkerboard_context,
        )
        np.savez_compressed(
            buffer,
            codec=np.asarray([1], dtype=np.uint8),
            residual_rans=np.frombuffer(rans_payload, dtype=np.uint8),
            residual_cdfs=cdfs,
            q_shape=np.asarray(q.shape, dtype=np.int32),
            max_q=np.asarray([max_q], dtype=np.int32),
            rans_precision=np.asarray([rans_precision], dtype=np.int32),
            checkerboard_context=np.asarray([int(checkerboard_context)], dtype=np.uint8),
        )
    else:
        raise ValueError(f"unsupported residual codec: {codec}")
    return buffer.getvalue()


def unpack_residual_stream(payload: bytes) -> torch.Tensor:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        codec = int(data["codec"][0])
        if codec == 0:
            return torch.from_numpy(data["q"].astype(np.float32))
        if codec != 1:
            raise ValueError(f"unsupported residual stream codec id: {codec}")
        return decode_residual(
            data["residual_rans"].astype(np.uint8, copy=False).tobytes(),
            data["residual_cdfs"].astype(np.int32, copy=False),
            tuple(int(value) for value in data["q_shape"].tolist()),
            int(data["max_q"][0]),
            int(data["rans_precision"][0]),
            bool(data["checkerboard_context"][0]),
        )


def frame_packet(
    packet: DNAPacket,
    marker_config: MarkerConfig | None = None,
    max_homopolymer: int = 3,
) -> str:
    marker_config = marker_config or MarkerConfig()
    raw = packet.serialize()
    payload_dna = ConstrainedDNAEncoder(max_homopolymer=max_homopolymer).encode(raw)
    sequence = frame_payload_dna(payload_dna, packet.header.stream_type.label, marker_config)
    packet.dna_sequence = sequence
    return sequence


def decode_framed_packet(
    sequence: str,
    marker_config: MarkerConfig | None = None,
    max_homopolymer: int = 3,
) -> DNAPacket:
    marker_config = marker_config or MarkerConfig()
    marker_stream_type, payload_dna = deframe_payload_dna(sequence, marker_config)
    if not payload_dna:
        raise PacketFormatError("framed packet contains no payload DNA")
    raw = ConstrainedDNAEncoder(max_homopolymer=max_homopolymer).decode(
        payload_dna,
        bit_length=len(payload_dna) * 2,
    )
    packet = parse_packet(raw)
    if packet.header.stream_type.label != marker_stream_type:
        raise PacketFormatError("stream marker and packet header disagree")
    packet.dna_sequence = sequence
    return packet


def _packet_identity_matches(actual: PacketHeader, expected: PacketHeader) -> bool:
    return (
        actual.stream_type == expected.stream_type
        and actual.image_id == expected.image_id
        and actual.block_id == expected.block_id
        and actual.row_id == expected.row_id
        and actual.col_id == expected.col_id
        and actual.packet_index == expected.packet_index
        and actual.rs_group_id == expected.rs_group_id
        and actual.rs_index == expected.rs_index
        and actual.is_parity == expected.is_parity
    )


def create_stream_packets(
    payload: bytes,
    *,
    stream_type: StreamType,
    image_id: int,
    block_id: int,
    row_id: int,
    col_id: int,
    block_size: int,
    tau: int,
    image_height: int,
    image_width: int,
    first_rs_group_id: int,
    packet_config: PacketConfig | None = None,
    rs_config: RSConfig | None = None,
    marker_config: MarkerConfig | None = None,
    max_homopolymer: int = 3,
) -> tuple[list[DNAPacket], int]:
    packet_config = packet_config or PacketConfig()
    rs_config = rs_config or RSConfig()
    marker_config = marker_config or MarkerConfig()
    chunks = [
        payload[index : index + packet_config.payload_bytes]
        for index in range(0, len(payload), packet_config.payload_bytes)
    ] or [b""]

    entries: list[tuple[int, int, bool, bytes, int, int]] = []
    group_id = first_rs_group_id
    for group_start in range(0, len(chunks), rs_config.data_shards):
        group_chunks = chunks[group_start : group_start + rs_config.data_shards]
        data_count = len(group_chunks)
        padded = [
            chunk.ljust(packet_config.payload_bytes, b"\x00")
            for chunk in group_chunks
        ]
        parity = encode_parity_shards(padded, rs_config.parity_shards)
        for rs_index, chunk in enumerate(group_chunks):
            entries.append((group_id, rs_index, False, chunk, len(chunk), data_count))
        for parity_index, shard in enumerate(parity):
            entries.append(
                (
                    group_id,
                    data_count + parity_index,
                    True,
                    shard,
                    len(shard),
                    data_count,
                )
            )
        group_id += 1

    total_packets = len(entries)
    packets: list[DNAPacket] = []
    for packet_index, (
        rs_group_id,
        rs_index,
        is_parity,
        packet_payload,
        payload_length,
        data_count,
    ) in enumerate(entries):
        header = PacketHeader(
            stream_type=stream_type,
            image_id=image_id,
            block_id=block_id,
            row_id=row_id,
            col_id=col_id,
            block_size=block_size,
            tau=tau,
            image_height=image_height,
            image_width=image_width,
            payload_length=payload_length,
            stream_length=len(payload),
            packet_index=packet_index,
            total_packets=total_packets,
            rs_group_id=rs_group_id,
            rs_index=rs_index,
            rs_data_shards=data_count,
            rs_parity_shards=rs_config.parity_shards,
            flags=FLAG_PARITY if is_parity else 0,
        )
        packet = DNAPacket(header=header, payload=packet_payload)
        frame_packet(packet, marker_config, max_homopolymer)
        packets.append(packet)
    return packets, group_id


def recover_stream_payload(
    expected_packets: list[DNAPacket],
    *,
    packet_config: PacketConfig | None = None,
    marker_config: MarkerConfig | None = None,
    max_homopolymer: int = 3,
) -> tuple[bytes, StreamRecoveryReport]:
    if not expected_packets:
        raise ValueError("no packets were supplied")
    packet_config = packet_config or PacketConfig()
    marker_config = marker_config or MarkerConfig()
    valid: dict[int, DNAPacket] = {}
    errors: list[str] = []
    for expected in expected_packets:
        if expected.dna_sequence is None:
            errors.append(f"packet {expected.header.packet_index}: missing")
            continue
        try:
            decoded = decode_framed_packet(
                expected.dna_sequence,
                marker_config,
                max_homopolymer,
            )
            if not _packet_identity_matches(decoded.header, expected.header):
                raise PacketFormatError("decoded packet identity does not match manifest")
            valid[expected.header.packet_index] = decoded
        except (MarkerDecodeError, PacketFormatError, ValueError) as exc:
            errors.append(f"packet {expected.header.packet_index}: {exc}")

    groups: dict[int, list[DNAPacket]] = {}
    for packet in expected_packets:
        groups.setdefault(packet.header.rs_group_id, []).append(packet)

    recovered_by_packet_index: dict[int, bytes] = {}
    recovered_count = 0
    for group_packets in groups.values():
        group_packets.sort(key=lambda packet: packet.header.rs_index)
        example = group_packets[0].header
        available: dict[int, bytes] = {}
        for expected in group_packets:
            decoded = valid.get(expected.header.packet_index)
            if decoded is not None:
                available[expected.header.rs_index] = decoded.payload.ljust(
                    packet_config.payload_bytes,
                    b"\x00",
                )
        recovered = recover_data_shards(
            available,
            example.rs_data_shards,
            example.rs_parity_shards,
        )
        for expected in group_packets:
            if expected.header.is_parity:
                continue
            recovered_payload = recovered[expected.header.rs_index][
                : expected.header.payload_length
            ]
            recovered_by_packet_index[expected.header.packet_index] = recovered_payload
            if expected.header.packet_index not in valid:
                recovered_count += 1

    data_packets = sorted(
        (packet for packet in expected_packets if not packet.header.is_parity),
        key=lambda packet: packet.header.packet_index,
    )
    payload = b"".join(
        recovered_by_packet_index[packet.header.packet_index]
        for packet in data_packets
    )
    stream_length = data_packets[0].header.stream_length
    payload = payload[:stream_length]
    report = StreamRecoveryReport(
        total_packets=len(expected_packets),
        valid_packets=len(valid),
        erasures=len(expected_packets) - len(valid),
        recovered_data_packets=recovered_count,
        errors=tuple(errors),
    )
    return payload, report

