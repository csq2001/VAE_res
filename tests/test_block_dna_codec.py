from __future__ import annotations

import pytest
import torch

from models import LatentInpainter, VaeResidualCodec
from utils.block_bitstream import (
    decode_framed_packet,
    pack_residual_stream,
    unpack_residual_stream,
)
from utils.dna_channel import DNAChannelConfig, simulate_dna_channel
from utils.ecc_rs import (
    RSConfig,
    ReedSolomonCodec,
    encode_parity_shards,
    recover_data_shards,
)
from utils.marker_code import MarkerConfig, classify_stream_marker
from utils.packet_format import (
    DNAPacket,
    PacketCRCError,
    PacketConfig,
    PacketHeader,
    StreamType,
    parse_packet,
)
from utils.patch_codec import BlockDNACodec
from utils.latent_tile_codec import LatentTileConfig
from utils.residual_tile_codec import (
    ResidualTileAddress,
    ResidualTileConfig,
    ResidualTileError,
    decode_residual_tile,
    encode_residual_tile,
)


def make_model() -> VaeResidualCodec:
    torch.manual_seed(9)
    return VaeResidualCodec(
        in_channels=1,
        latent_channels=4,
        base_channels=8,
        residual_hidden=8,
        residual_condition_channels=2,
        residual_extra_blocks=0,
        max_q=4,
        checkerboard_context=False,
    ).eval()


def make_codec(residual_codec: str = "zlib") -> BlockDNACodec:
    return BlockDNACodec(
        make_model(),
        block_size=64,
        tau=2,
        residual_codec=residual_codec,
        packet_config=PacketConfig(payload_bytes=256),
        rs_config=RSConfig(data_shards=3, parity_shards=2),
    )


def test_latent_and_residual_marker_classification():
    config = MarkerConfig()
    assert classify_stream_marker(config.latent_marker, config) == "latent"
    assert classify_stream_marker(config.residual_marker, config) == "residual"


def test_crc_detects_corrupted_packet():
    header = PacketHeader(
        stream_type=StreamType.LATENT,
        image_id=1,
        block_id=0,
        row_id=0,
        col_id=0,
        block_size=64,
        tau=2,
        image_height=64,
        image_width=64,
        payload_length=5,
        stream_length=5,
        packet_index=0,
        total_packets=1,
        rs_group_id=0,
        rs_index=0,
        rs_data_shards=1,
        rs_parity_shards=1,
    )
    packet = DNAPacket(header, b"hello")
    raw = bytearray(packet.serialize())
    raw[-1] ^= 0x01
    with pytest.raises(PacketCRCError):
        parse_packet(bytes(raw))


def test_rs_recovers_partial_erasure_packets():
    data = [
        bytes((offset + index) % 256 for index in range(64))
        for offset in range(4)
    ]
    parity = encode_parity_shards(data, parity_shards=2)
    all_shards = data + parity
    available = {
        index: shard
        for index, shard in enumerate(all_shards)
        if index not in {1, 4}
    }
    assert recover_data_shards(available, 4, 2) == data
    codec = ReedSolomonCodec(RSConfig(4, 2))
    encoded = codec.encode(data)
    encoded[1] = None
    assert codec.decode(encoded, erasure_positions=[4]) == data


def test_rans_residual_stream_round_trip():
    model = make_model()
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        output = model(x, tau=2, deterministic=True)
    payload = pack_residual_stream(
        output.q,
        output.residual_logits,
        codec="rans",
        max_q=model.residual_entropy.max_q,
        checkerboard_context=model.residual_entropy.checkerboard_context,
    )
    assert torch.equal(unpack_residual_stream(payload), output.q)


def test_64x64_block_complete_round_trip_without_errors():
    codec = make_codec()
    image = torch.rand(1, 1, 64, 64)
    encoded = codec.encode_image(image, image_id="round-trip")
    decoded = codec.decode_image(encoded)
    assert len(encoded.blocks) == 1
    assert encoded.blocks[0].block_size == 64
    assert torch.equal(decoded.image, encoded.reference_image())
    assert all(report.latent.erasures == 0 for report in decoded.blocks)
    assert all(report.residual.erasures == 0 for report in decoded.blocks)


def test_full_image_round_trip_with_multiple_of_eight_padding():
    codec = make_codec()
    image = torch.rand(1, 1, 73, 81)
    encoded = codec.encode_image(image, image_id="full-image")
    decoded = codec.decode_image(encoded)
    assert len(encoded.blocks) == 1
    assert encoded.blocks[0].encoded_image_size == (80, 88)
    assert decoded.image.shape == image.shape
    assert torch.equal(decoded.image, encoded.reference_image())


def test_missing_latent_tile_is_recovered_by_outer_rs():
    codec = make_codec()
    encoded = codec.encode_image(torch.rand(1, 1, 64, 64), image_id=3)
    expected = encoded.reference_image()
    first_data_packet = next(
        packet for packet in encoded.packets() if not packet.header.is_parity
    )
    first_data_packet.dna_sequence = None
    decoded = codec.decode_image(encoded)
    assert torch.equal(decoded.image, expected)
    assert sum(report.latent.erasures for report in decoded.blocks) == 0
    assert sum(
        report.latent_rs_recovered_tiles for report in decoded.blocks
    ) == 1


def test_latent_rs_recovers_four_missing_tiles():
    codec = make_codec()
    encoded = codec.encode_image(torch.rand(1, 1, 64, 64), image_id=30)
    expected = encoded.reference_image()
    group = [
        packet
        for packet in encoded.blocks[0].latent_tiles
        if packet.header.rs_group_id == 0 and not packet.header.is_parity
    ]
    for packet in group[:4]:
        packet.dna_sequence = None
    decoded = codec.decode_image(encoded)
    assert torch.equal(decoded.image, expected)
    assert decoded.blocks[0].latent.erasures == 0
    assert decoded.blocks[0].latent_rs_recovered_tiles == 4


def test_latent_rs_overflow_uses_local_prior_fallback():
    codec = make_codec()
    encoded = codec.encode_image(torch.rand(1, 1, 64, 64), image_id=31)
    group_packets = [
        packet
        for packet in encoded.blocks[0].latent_tiles
        if packet.header.rs_group_id == 0
    ]
    data_packets = [
        packet for packet in group_packets if not packet.header.is_parity
    ]
    parity_packet = next(
        packet for packet in group_packets if packet.header.is_parity
    )
    for packet in data_packets:
        packet.dna_sequence = None
    parity_packet.dna_sequence = None
    decoded = codec.decode_image(encoded)
    assert decoded.blocks[0].latent.erasures == len(data_packets)
    assert decoded.blocks[0].latent_rs_recovered_tiles == 0
    assert torch.isfinite(decoded.image).all()


@pytest.mark.parametrize("edit_type", ["substitution", "insertion", "deletion"])
def test_latent_ids_error_is_directly_corrected(edit_type: str):
    codec = make_codec()
    encoded = codec.encode_image(torch.rand(1, 1, 64, 64), image_id=4)
    expected = encoded.reference_image()
    packet = next(packet for packet in encoded.packets() if not packet.header.is_parity)
    sequence = packet.dna_sequence
    assert sequence is not None
    edit_position = len(codec.marker_config.block_start_marker) + len(
        codec.marker_config.latent_marker
    ) + 10
    if edit_type == "substitution":
        replacement = "A" if sequence[edit_position] != "A" else "C"
        packet.dna_sequence = (
            sequence[:edit_position] + replacement + sequence[edit_position + 1 :]
        )
    elif edit_type == "insertion":
        packet.dna_sequence = sequence[:edit_position] + "A" + sequence[edit_position:]
    else:
        packet.dna_sequence = sequence[:edit_position] + sequence[edit_position + 1 :]

    with pytest.raises(ValueError):
        decode_framed_packet(packet.dna_sequence, codec.marker_config)
    decoded = codec.decode_image(encoded)
    assert torch.equal(decoded.image, expected)
    assert sum(report.latent.erasures for report in decoded.blocks) == 0
    assert sum(
        report.latent.recovered_data_packets for report in decoded.blocks
    ) == 1
    assert sum(
        report.latent_corrected_codewords for report in decoded.blocks
    ) == 1


def test_dna_channel_changes_length_for_indels():
    sequence = "ACGT" * 200
    result = simulate_dna_channel(
        sequence,
        DNAChannelConfig(insertion_rate=0.05, deletion_rate=0.02),
        seed=13,
    )
    assert len(result.sequence) == len(sequence) + result.insertions - result.deletions
    assert result.insertions > 0
    assert result.deletions > 0


@pytest.mark.parametrize("edit_type", ["substitution", "insertion", "deletion"])
def test_residual_tile_address_code_corrects_one_ids_edit(edit_type: str):
    q = torch.randint(-4, 5, (1, 1, 16, 16), dtype=torch.int64).float()
    packet = encode_residual_tile(
        q,
        ResidualTileAddress(
            image_id=1,
            block_id=2,
            tile_row=1,
            tile_col=3,
            tile_height=16,
            tile_width=16,
            channels=1,
            tau=2,
            payload_length=0,
        ),
    )
    sequence = packet.dna_sequence
    assert sequence is not None
    position = 7 * 5 + 3
    if edit_type == "substitution":
        replacement = "A" if sequence[position] != "A" else "C"
        packet.dna_sequence = (
            sequence[:position] + replacement + sequence[position + 1 :]
        )
    elif edit_type == "insertion":
        packet.dna_sequence = sequence[:position] + "A" + sequence[position:]
    else:
        packet.dna_sequence = sequence[:position] + sequence[position + 1 :]
    result = decode_residual_tile(packet)
    assert torch.equal(result.q, q)
    assert result.corrected_codewords == 1


@pytest.mark.parametrize("edit_type", ["substitution", "insertion", "deletion"])
def test_residual_compact_payload_detects_ids_edit(edit_type: str):
    q = torch.zeros(1, 1, 16, 16)
    packet = encode_residual_tile(
        q,
        ResidualTileAddress(1, 2, 1, 3, 16, 16, 1, 2, 0),
    )
    sequence = packet.dna_sequence
    assert sequence is not None
    position = len(packet.header.pack()) * 7 + 10
    if edit_type == "substitution":
        replacement = "A" if sequence[position] != "A" else "C"
        packet.dna_sequence = (
            sequence[:position] + replacement + sequence[position + 1 :]
        )
    elif edit_type == "insertion":
        packet.dna_sequence = sequence[:position] + "A" + sequence[position:]
    else:
        packet.dna_sequence = sequence[:position] + sequence[position + 1 :]
    with pytest.raises(ResidualTileError):
        decode_residual_tile(packet)


def test_high_energy_residual_payload_keeps_edit_correction():
    q = torch.full((1, 1, 16, 16), 2.0)
    packet = encode_residual_tile(
        q,
        ResidualTileAddress(1, 2, 1, 3, 16, 16, 1, 2, 0),
    )
    assert packet.header.payload_mode == 1
    sequence = packet.dna_sequence
    assert sequence is not None
    position = len(packet.header.pack()) * 7 + 10
    replacement = "A" if sequence[position] != "A" else "C"
    packet.dna_sequence = (
        sequence[:position] + replacement + sequence[position + 1 :]
    )
    result = decode_residual_tile(packet)
    assert torch.equal(result.q, q)
    assert result.corrected_codewords == 1


def test_uncorrectable_residual_tile_is_locally_dropped():
    codec = make_codec()
    encoded = codec.encode_image(torch.rand(1, 1, 64, 64), image_id=7)
    encoded.blocks[0].residual_tiles[0].dna_sequence = None
    decoded = codec.decode_image(encoded)
    report = decoded.blocks[0].residual
    assert report.erasures == 1
    assert report.valid_packets == len(encoded.blocks[0].residual_tiles) - 1
    assert torch.isfinite(decoded.image).all()


def test_residual_tile_has_no_periodic_marker_framing():
    q = torch.zeros(1, 1, 8, 8)
    packet = encode_residual_tile(
        q,
        ResidualTileAddress(1, 0, 0, 0, 8, 8, 1, 2, 0),
        ResidualTileConfig(tile_size=8),
    )
    assert packet.dna_sequence is not None
    assert len(packet.dna_sequence) == (
        len(packet.header.pack()) * 7 + len(packet.payload) * 4
    )


def test_latent_inpainter_preserves_every_valid_value():
    torch.manual_seed(17)
    model = LatentInpainter(
        latent_channels=8,
        channel_group=4,
        hidden_channels=16,
        context_channels=24,
    )
    latent = torch.randn(2, 8, 16, 16)
    mask = torch.ones_like(latent)
    mask[:, :4, 4:8, 4:8] = 0
    output = model(
        latent * mask,
        mask,
        torch.zeros(8),
        torch.ones(8),
    )
    assert output.repaired.shape == latent.shape
    assert output.uncertainty.shape == (2, 2, 16, 16)
    assert torch.equal(output.repaired[mask.bool()], latent[mask.bool()])


def test_rs_overflow_is_sent_to_latent_inpainter():
    vae = make_model()
    inpainter = LatentInpainter(
        latent_channels=4,
        channel_group=4,
        hidden_channels=16,
        context_channels=24,
    )
    codec = BlockDNACodec(
        vae,
        block_size=64,
        tau=2,
        latent_inpainter=inpainter,
        latent_tile_config=LatentTileConfig(
            spatial_size=4,
            channel_group=4,
            rs_data_tiles=8,
            rs_parity_tiles=4,
        ),
    )
    encoded = codec.encode_image(torch.rand(1, 1, 64, 64), image_id=99)
    group = [
        packet
        for packet in encoded.blocks[0].latent_tiles
        if packet.header.rs_group_id == 0
    ]
    for packet in group:
        if not packet.header.is_parity:
            packet.dna_sequence = None
    next(packet for packet in group if packet.header.is_parity).dna_sequence = None
    decoded = codec.decode_image(encoded)
    report = decoded.blocks[0]
    assert report.latent.erasures > 0
    assert report.latent_predicted_tiles == report.latent.erasures
    assert torch.isfinite(decoded.image).all()
