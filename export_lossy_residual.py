from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.vae import quantize_image_ste
from viewer_server import image_to_tensor, model_from_checkpoint


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export full-image VAE lossy images and exact residuals."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best.pth",
        help="VAE checkpoint path.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--output-root",
        default="outputs/best_decomposition",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="Subdirectories below data-root; use an empty list to scan all files.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--residual-visual-gain",
        type=float,
        default=2.0,
        help="PNG visualization uses clip(128 + residual * gain, 0, 255).",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not save exact compressed signed int16 residual .npz files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images (useful for a smoke test).",
    )
    return parser.parse_args()


def discover_images(data_root: Path, splits: list[str]) -> list[Path]:
    roots = [data_root / split for split in splits] if splits else [data_root]
    images: list[Path] = []
    for folder in roots:
        if not folder.exists():
            continue
        images.extend(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return sorted(set(images))


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def save_pixels(pixels: torch.Tensor, path: Path) -> None:
    array = pixels.detach().cpu().clamp(0, 255).to(torch.uint8)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.shape[0] == 1:
        image = Image.fromarray(array[0].numpy(), mode="L")
    else:
        image = Image.fromarray(array.permute(1, 2, 0).numpy(), mode="RGB")
    image.save(path, format="PNG")


@torch.no_grad()
def decompose_image(
    image: torch.Tensor,
    model: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_height, original_width = image.shape[-2:]
    padded_height = (original_height + 7) // 8 * 8
    padded_width = (original_width + 7) // 8 * 8
    padded = F.pad(
        image,
        (0, padded_width - original_width, 0, padded_height - original_height),
    )
    y = model.encoder(padded)
    y_hat = model.quantize_latent(y, deterministic=True)
    x_tilde = model.decoder(y_hat)
    if x_tilde.shape[-2:] != padded.shape[-2:]:
        x_tilde = F.interpolate(
            x_tilde,
            size=padded.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    lossy_pixels = quantize_image_ste(x_tilde)[
        ..., :original_height, :original_width
    ]
    original_pixels = torch.round(image.clamp(0, 1) * 255.0)
    residual = original_pixels - lossy_pixels
    return lossy_pixels, residual


def main() -> None:
    args = parse_args()
    data_root = (ROOT / args.data_root).resolve()
    checkpoint = (ROOT / args.checkpoint).resolve()
    output_root = (ROOT / args.output_root).resolve()
    lossy_root = output_root / "lossy"
    residual_root = output_root / "residual"
    if args.residual_visual_gain <= 0:
        raise ValueError("--residual-visual-gain must be positive")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    paths = discover_images(data_root, args.splits)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No images found below {data_root}")

    device = choose_device(args.device)
    model = model_from_checkpoint(checkpoint, device)
    model.eval()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "lossy_format": "8-bit PNG",
        "residual_png": (
            "visualization only: clip(128 + signed_residual * "
            f"{args.residual_visual_gain}, 0, 255)"
        ),
        "residual_npz": (
            "exact signed int16 HWC array under key 'residual'; "
            "original = lossy + residual"
        ),
    }
    (output_root / "format.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = output_root / "manifest.csv"
    records: dict[str, dict[str, object]] = {}
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("source"):
                    records[row["source"]] = dict(row)
    processed = skipped = failed = 0
    for index, source in enumerate(paths, start=1):
        relative = source.relative_to(data_root)
        png_relative = relative.with_suffix(".png")
        lossy_path = lossy_root / png_relative
        residual_png_path = residual_root / png_relative
        residual_npz_path = residual_root / relative.with_suffix(".npz")
        required = [lossy_path, residual_png_path]
        if not args.no_raw:
            required.append(residual_npz_path)
        if not args.overwrite and all(path.exists() for path in required):
            skipped += 1
            print(f"[{index}/{len(paths)}] skip {relative}")
            continue
        try:
            image = image_to_tensor(source, model.in_channels).to(device)
            lossy, residual = decompose_image(image, model)
            save_pixels(lossy, lossy_path)
            residual_visual = (
                128.0 + residual * args.residual_visual_gain
            ).clamp(0, 255)
            save_pixels(residual_visual, residual_png_path)
            if not args.no_raw:
                residual_npz_path.parent.mkdir(parents=True, exist_ok=True)
                raw = (
                    residual[0]
                    .detach()
                    .cpu()
                    .to(torch.int16)
                    .permute(1, 2, 0)
                    .numpy()
                )
                np.savez_compressed(residual_npz_path, residual=raw)
            mse = torch.mean(
                ((image.detach().cpu() * 255.0) - lossy.cpu()) ** 2
            ).item()
            source_key = relative.as_posix()
            records[source_key] = {
                "source": source_key,
                "width": image.shape[-1],
                "height": image.shape[-2],
                "residual_min": int(residual.min().item()),
                "residual_max": int(residual.max().item()),
                "residual_abs_mean": round(
                    float(residual.abs().mean().item()), 5
                ),
                "mse_pixels": round(mse, 5),
            }
            processed += 1
            print(f"[{index}/{len(paths)}] ok   {relative}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(paths)}] FAIL {relative}: {exc}")

    if records:
        with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
            fieldnames = list(next(iter(records.values())))
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records[key] for key in sorted(records))
    print(
        f"Done: processed={processed}, skipped={skipped}, failed={failed}, "
        f"output={output_root}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
