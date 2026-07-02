from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


VALID_SPLITS = {"train", "val", "test"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a 512x512 dataset from a split manifest.")
    parser.add_argument("--source", default="Ancient_Figure_Oil_Painting")
    parser.add_argument("--manifest", default="split_manifest.csv")
    parser.add_argument("--output", default="data")
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "source", "output"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        rows = list(reader)

    seen_sources = set()
    seen_targets = set()
    for line_number, row in enumerate(rows, start=2):
        split = row["split"].strip()
        source_name = row["source"].strip()
        output_name = row["output"].strip()
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split at line {line_number}: {split!r}")
        if not source_name or not output_name:
            raise ValueError(f"Empty filename at line {line_number}")
        if source_name in seen_sources:
            raise ValueError(f"Duplicate source at line {line_number}: {source_name}")
        target_key = (split, output_name)
        if target_key in seen_targets:
            raise ValueError(f"Duplicate target at line {line_number}: {split}/{output_name}")
        seen_sources.add(source_name)
        seen_targets.add(target_key)
        row["split"] = split
        row["source"] = source_name
        row["output"] = output_name
    return rows


def save_image(source_path: Path, target_path: Path, size: int) -> None:
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(target_path, quality=95, subsampling=0)
        else:
            image.save(target_path)


def main():
    args = parse_args()
    source_root = Path(args.source)
    manifest_path = Path(args.manifest)
    output_root = Path(args.output)
    if args.size <= 0:
        raise ValueError("size must be positive")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = read_manifest(manifest_path)
    missing = [row["source"] for row in rows if not (source_root / row["source"]).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} manifest images are missing: {preview}")

    counts = Counter(row["split"] for row in rows)
    print(f"Preparing {len(rows)} images at {args.size}x{args.size}: {dict(counts)}", flush=True)
    for index, row in enumerate(rows, start=1):
        source_path = source_root / row["source"]
        target_path = output_root / row["split"] / row["output"]
        save_image(source_path, target_path, args.size)
        if index % 100 == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)}", flush=True)

    print(f"Dataset written to {output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
