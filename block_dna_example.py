from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from utils.dna_channel import DNAChannelConfig, simulate_dna_channel
from utils.ecc_rs import RSConfig
from utils.marker_code import MarkerConfig
from utils.metrics import max_abs_error_pixels, psnr
from utils.packet_format import PacketConfig
from utils.patch_codec import BlockDNACodec
from utils.latent_tile_codec import LatentTileConfig
from utils.residual_tile_codec import ResidualTileConfig
from viewer_server import image_to_tensor, model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Full-image VAE residual-tile DNA example.")
    parser.add_argument("--image", default="data/test/000003.png")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/checkpoint_stage1.pth",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--residual-codec", choices=["zlib", "rans"], default="zlib")
    parser.add_argument("--packet-bytes", type=int, default=128)
    parser.add_argument("--rs-data", type=int, default=8)
    parser.add_argument("--rs-parity", type=int, default=8)
    parser.add_argument("--sync-interval", type=int, default=100)
    parser.add_argument("--residual-tile-size", type=int, default=16)
    parser.add_argument("--latent-spatial-size", type=int, default=4)
    parser.add_argument("--latent-channel-group", type=int, default=32)
    parser.add_argument("--substitution-rate", type=float, default=0.0)
    parser.add_argument("--insertion-rate", type=float, default=0.0)
    parser.add_argument("--deletion-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoint(Path(args.checkpoint), device)
    image = image_to_tensor(Path(args.image), model.in_channels).to(device)
    codec = BlockDNACodec(
        model,
        block_size=args.block_size,
        tau=args.tau,
        residual_codec=args.residual_codec,
        packet_config=PacketConfig(args.packet_bytes),
        rs_config=RSConfig(args.rs_data, args.rs_parity),
        marker_config=MarkerConfig(sync_interval=args.sync_interval),
        latent_tile_config=LatentTileConfig(
            spatial_size=args.latent_spatial_size,
            channel_group=args.latent_channel_group,
        ),
        residual_tile_config=ResidualTileConfig(tile_size=args.residual_tile_size),
    )
    encoded = codec.encode_image(image, image_id=Path(args.image).name)
    channel = DNAChannelConfig(
        substitution_rate=args.substitution_rate,
        insertion_rate=args.insertion_rate,
        deletion_rate=args.deletion_rate,
    )
    rng = random.Random(args.seed)
    total_errors = 0
    for packet in encoded.packets():
        result = simulate_dna_channel(packet.dna_sequence or "", channel, rng=rng)
        packet.dna_sequence = result.sequence
        total_errors += result.total_errors

    decoded = codec.decode_image(encoded)
    reference = image.detach().cpu()
    dna_nt = sum(len(packet.dna_sequence or "") for packet in encoded.packets())
    recovered_packets = sum(
        report.latent.recovered_data_packets
        for report in decoded.blocks
    )
    corrected_residual_tiles = sum(
        report.residual.recovered_data_packets for report in decoded.blocks
    )
    dropped_residual_tiles = sum(
        report.residual.erasures for report in decoded.blocks
    )
    rs_recovered_latent_tiles = sum(
        report.latent_rs_recovered_tiles for report in decoded.blocks
    )
    print(
        {
            "device": str(device),
            "blocks": len(encoded.blocks),
            "packets": len(encoded.packets()),
            "dna_nt": dna_nt,
            "simulated_base_errors": total_errors,
            "corrected_latent_tiles": recovered_packets,
            "rs_recovered_latent_tiles": rs_recovered_latent_tiles,
            "corrected_residual_tiles": corrected_residual_tiles,
            "dropped_residual_tiles": dropped_residual_tiles,
            "psnr": psnr(reference, decoded.image),
            "maxerr": max_abs_error_pixels(reference, decoded.image),
        }
    )


if __name__ == "__main__":
    main()
