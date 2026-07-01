from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from utils.dna_channel import DNAChannelConfig, simulate_dna_channel
from utils.ecc_rs import RSConfig
from utils.marker_code import MarkerConfig
from utils.metrics import max_abs_error_pixels, psnr
from utils.packet_format import FLAG_PARITY, PacketConfig, StreamType
from utils.patch_codec import BlockDNACodec, BlockDecodeError
from utils.latent_tile_codec import LatentTileConfig
from utils.residual_tile_codec import ResidualTileConfig
from viewer_server import (
    ExclusiveThreadingHTTPServer,
    image_to_tensor,
    json_response,
    list_checkpoints,
    list_images,
    model_from_checkpoint,
    resolve_under,
    tensor_to_data_url,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
VIEWER_HTML = ROOT / "block_dna_viewer.html"
MODEL_LOCK = threading.Lock()
MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def _number(value: float) -> float | None:
    return round(value, 5) if math.isfinite(value) else None


def _bounded_int(
    params: dict,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = int(params.get(name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_float(
    params: dict,
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = float(params.get(name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    key = (str(checkpoint), str(device))
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = model_from_checkpoint(checkpoint, device)
    return MODEL_CACHE[key]


def _error_visual(reference: torch.Tensor, decoded: torch.Tensor) -> str:
    error = (reference.detach().cpu() - decoded.detach().cpu()).abs()
    intensity = error.max(dim=1, keepdim=True).values
    peak = max(float(intensity.max().item()), 1.0 / 255.0)
    intensity = (intensity / peak).clamp(0.0, 1.0)
    heat = torch.cat(
        (intensity, (1.0 - (intensity - 0.5).abs() * 2.0).clamp(0.0, 1.0), 1.0 - intensity),
        dim=1,
    )
    return tensor_to_data_url(heat)


def _metric_pair(reference: torch.Tensor, result: torch.Tensor) -> dict:
    return {
        "psnr": _number(psnr(reference, result)),
        "maxerr": max_abs_error_pixels(reference, result),
    }


def evaluate(params: dict) -> dict:
    started = time.perf_counter()
    image_name = str(params.get("image", ""))
    checkpoint_name = str(params.get("checkpoint", ""))
    if image_name not in {entry["path"] for entry in list_images()}:
        raise ValueError("Unknown image")
    if checkpoint_name not in set(list_checkpoints()):
        raise ValueError("Unknown checkpoint")

    block_size = _bounded_int(params, "block_size", 64, 16, 512)
    if block_size % 8:
        raise ValueError("block_size must be a multiple of 8")
    tau = _bounded_int(params, "tau", 5, 0, 64)
    packet_bytes = _bounded_int(params, "packet_bytes", 128, 16, 8192)
    rs_data = _bounded_int(params, "rs_data", 8, 1, 254)
    rs_parity = _bounded_int(params, "rs_parity", 8, 1, 254)
    if rs_data + rs_parity > 255:
        raise ValueError("rs_data + rs_parity cannot exceed 255")
    sync_interval = _bounded_int(params, "sync_interval", 100, 16, 1024)
    residual_tile_size = _bounded_int(
        params, "residual_tile_size", 16, 4, block_size
    )
    latent_spatial_size = _bounded_int(
        params, "latent_spatial_size", 4, 1, 64
    )
    latent_channel_group = _bounded_int(
        params, "latent_channel_group", 32, 1, 1024
    )
    residual_codec = str(params.get("residual_codec", "zlib"))
    if residual_codec not in {"zlib", "rans"}:
        raise ValueError("residual_codec must be zlib or rans")
    seed = _bounded_int(params, "seed", 42, 0, 2**31 - 1)
    channel = DNAChannelConfig(
        substitution_rate=_bounded_float(params, "substitution_rate", 0.0, 0.0, 0.25),
        insertion_rate=_bounded_float(params, "insertion_rate", 0.0, 0.0, 0.25),
        deletion_rate=_bounded_float(params, "deletion_rate", 0.0, 0.0, 0.25),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = resolve_under(DATA_ROOT, image_name)
    checkpoint_path = resolve_under(ROOT, checkpoint_name)

    with MODEL_LOCK:
        model = _load_model(checkpoint_path, device)
        image = image_to_tensor(image_path, model.in_channels).to(device)
        codec = BlockDNACodec(
            model,
            block_size=block_size,
            tau=tau,
            residual_codec=residual_codec,
            packet_config=PacketConfig(packet_bytes),
            rs_config=RSConfig(rs_data, rs_parity),
            marker_config=MarkerConfig(sync_interval=sync_interval),
            latent_tile_config=LatentTileConfig(
                spatial_size=latent_spatial_size,
                channel_group=latent_channel_group,
            ),
            residual_tile_config=ResidualTileConfig(tile_size=residual_tile_size),
        )

        encode_started = time.perf_counter()
        encoded = codec.encode_image(image, image_id=image_name)
        encode_seconds = time.perf_counter() - encode_started
        clean_reference = encoded.reference_image()
        packets = encoded.packets()
        original_sequences = [packet.dna_sequence or "" for packet in packets]
        original_dna_nt = sum(map(len, original_sequences))

        rng = random.Random(seed)
        substitutions = insertions = deletions = 0
        damaged_packets = 0
        for packet in packets:
            result = simulate_dna_channel(packet.dna_sequence or "", channel, rng=rng)
            packet.dna_sequence = result.sequence
            substitutions += result.substitutions
            insertions += result.insertions
            deletions += result.deletions
            damaged_packets += int(result.total_errors > 0)

        decode_started = time.perf_counter()
        decoded = None
        decode_error = None
        try:
            decoded = codec.decode_image(encoded)
        except BlockDecodeError as exc:
            decode_error = str(exc)
        decode_seconds = time.perf_counter() - decode_started

    reference = image.detach().cpu()
    pixels = encoded.original_height * encoded.original_width
    raw_bits = pixels * encoded.channels * 8
    raw_dna_nt = raw_bits / 2.0
    latent_bytes = sum(len(block.latent_stream) for block in encoded.blocks)
    residual_bytes = sum(len(block.residual_stream) for block in encoded.blocks)
    data_packets = sum(not bool(packet.header.flags & FLAG_PARITY) for packet in packets)
    parity_packets = len(packets) - data_packets
    mutated_dna_nt = sum(len(packet.dna_sequence or "") for packet in packets)

    reports = [] if decoded is None else [
        report_stream
        for block in decoded.blocks
        for report_stream in (block.latent, block.residual)
    ]
    latent_reports = [] if decoded is None else [
        block.latent for block in decoded.blocks
    ]
    residual_reports = [] if decoded is None else [
        block.residual for block in decoded.blocks
    ]
    valid_packets = sum(report.valid_packets for report in reports)
    erasures = sum(report.erasures for report in reports)
    recovered_packets = sum(
        report.recovered_data_packets for report in latent_reports
    )
    latent_rs_recovered = (
        0
        if decoded is None
        else sum(block.latent_rs_recovered_tiles for block in decoded.blocks)
    )
    corrected_latent_codewords = (
        0
        if decoded is None
        else sum(block.latent_corrected_codewords for block in decoded.blocks)
    )
    dropped_latent_tiles = sum(report.erasures for report in latent_reports)
    corrected_residual_tiles = sum(
        report.recovered_data_packets for report in residual_reports
    )
    corrected_residual_codewords = (
        0
        if decoded is None
        else sum(block.residual_corrected_codewords for block in decoded.blocks)
    )
    dropped_residual_tiles = sum(report.erasures for report in residual_reports)
    recovery_errors = [error for report in reports for error in report.errors]

    result = {
        "ok": decoded is not None,
        "decode_error": decode_error,
        "device": str(device),
        "image": {
            "name": image_name,
            "width": encoded.original_width,
            "height": encoded.original_height,
            "encoded_width": encoded.blocks[0].encoded_image_size[1],
            "encoded_height": encoded.blocks[0].encoded_image_size[0],
            "channels": encoded.channels,
            "blocks": len(encoded.blocks),
            "rows": encoded.rows,
            "columns": encoded.columns,
        },
        "codec": {
            "vae_mode": "full_image",
            "block_size": block_size,
            "tau": tau,
            "residual_codec": residual_codec,
            "packet_bytes": packet_bytes,
            "sync_interval": sync_interval,
            "residual_tile_size": residual_tile_size,
            "latent_spatial_size": latent_spatial_size,
            "latent_channel_group": latent_channel_group,
            "rs_data": rs_data,
            "rs_parity": rs_parity,
        },
        "compression": {
            "latent_bytes": latent_bytes,
            "residual_bytes": residual_bytes,
            "payload_bytes": latent_bytes + residual_bytes,
            "payload_bpp": round((latent_bytes + residual_bytes) * 8 / pixels, 4),
            "payload_percent_of_raw": round((latent_bytes + residual_bytes) * 8 / raw_bits * 100, 3),
            "dna_nt": original_dna_nt,
            "dna_nt_per_pixel": round(original_dna_nt / pixels, 4),
            "dna_percent_of_raw": round(original_dna_nt / raw_dna_nt * 100, 3),
            "dna_compression_ratio": round(raw_dna_nt / original_dna_nt, 4),
            "packets": len(packets),
            "data_packets": data_packets,
            "parity_packets": parity_packets,
        },
        "channel": {
            "substitutions": substitutions,
            "insertions": insertions,
            "deletions": deletions,
            "total_errors": substitutions + insertions + deletions,
            "damaged_packets": damaged_packets,
            "mutated_dna_nt": mutated_dna_nt,
        },
        "recovery": {
            "valid_packets": valid_packets,
            "erasures": erasures,
            "rs_recovered_packets": latent_rs_recovered,
            "corrected_latent_tiles": recovered_packets,
            "corrected_latent_codewords": corrected_latent_codewords,
            "dropped_latent_tiles": dropped_latent_tiles,
            "latent_tiles": sum(
                packet.header.stream_type == StreamType.LATENT
                and not packet.header.is_parity
                for packet in packets
            ),
            "corrected_residual_tiles": corrected_residual_tiles,
            "corrected_residual_codewords": corrected_residual_codewords,
            "dropped_residual_tiles": dropped_residual_tiles,
            "residual_tiles": sum(
                report.total_packets for report in residual_reports
            ),
            "degraded_pixel_percent": round(
                dropped_residual_tiles
                * residual_tile_size
                * residual_tile_size
                / max(pixels, 1)
                * 100,
                3,
            ),
            "reported_errors": len(recovery_errors),
            "error_examples": recovery_errors[:6],
        },
        "quality": {
            "clean": _metric_pair(reference, clean_reference),
            "decoded": None if decoded is None else _metric_pair(reference, decoded.image),
        },
        "timing": {
            "encode_seconds": round(encode_seconds, 3),
            "decode_seconds": round(decode_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "images": {
            "original": tensor_to_data_url(reference),
            "clean": tensor_to_data_url(clean_reference),
            "decoded": None if decoded is None else tensor_to_data_url(decoded.image),
            "error": None if decoded is None else _error_visual(reference, decoded.image),
        },
        "dna_preview": {
            "original": original_sequences[0][:1000] if original_sequences else "",
            "mutated": (packets[0].dna_sequence or "")[:1000] if packets else "",
            "packet_stream": (
                "latent"
                if packets and packets[0].header.stream_type == StreamType.LATENT
                else "residual"
            ),
        },
    }
    return result


class BlockDNAHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/block_dna_viewer.html"}:
            if not VIEWER_HTML.exists():
                self.send_error(404, "block_dna_viewer.html not found")
                return
            content = VIEWER_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if path == "/api/config":
            checkpoints = list_checkpoints()
            json_response(
                self,
                {
                    "images": list_images(),
                    "checkpoints": checkpoints,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "defaults": {
                        "checkpoint": next(
                            (item for item in checkpoints if "stage1" in item.lower()),
                            checkpoints[-1] if checkpoints else "",
                        ),
                        "block_size": 64,
                        "tau": 5,
                        "packet_bytes": 128,
                        "rs_data": 8,
                        "rs_parity": 8,
                        "sync_interval": 100,
                        "residual_tile_size": 16,
                        "latent_spatial_size": 4,
                        "latent_channel_group": 32,
                    },
                },
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/evaluate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            params = json.loads(self.rfile.read(length) or b"{}")
            json_response(self, evaluate(params))
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            json_response(self, {"ok": False, "error": str(exc)}, status=400)

    def log_message(self, format: str, *args) -> None:
        print(f"[block-dna] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Block DNA compression/error-correction viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    server = ExclusiveThreadingHTTPServer((args.host, args.port), BlockDNAHandler)
    print(f"Block DNA viewer: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
