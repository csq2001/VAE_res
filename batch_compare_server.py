from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from utils.bitstream import pack_tensors, payload_to_dna, unpack_tensors
from utils.metrics import bits_per_pixel
from viewer_server import image_to_tensor, model_from_checkpoint


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
CHECKPOINT_ROOT = ROOT / "outputs" / "checkpoints"
REPORT_ROOT = ROOT / "outputs" / "reports"
VIEWER_HTML = ROOT / "batch_compare.html"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
SPLITS = ("train", "val", "test")
CSV_FIELDS = (
    "index",
    "split",
    "image",
    "width",
    "height",
    "channels",
    "estimated_bpp",
    "zlib_bytes",
    "zlib_bpp",
    "zlib_size_percent",
    "zlib_dna_nt",
    "rans_bytes",
    "rans_bpp",
    "rans_residual_bpp",
    "rans_size_percent",
    "rans_dna_nt",
    "rans_binary_saving_percent",
    "rans_dna_saving_percent",
    "rans_roundtrip",
    "seconds",
    "error",
)


class BatchState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.data = self._idle_state()
        self.results: list[dict] = []

    @staticmethod
    def _idle_state() -> dict:
        return {
            "status": "idle",
            "job_id": None,
            "checkpoint": None,
            "split": None,
            "tau": None,
            "device": None,
            "total": 0,
            "completed": 0,
            "succeeded": 0,
            "errors": 0,
            "current_image": None,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "report_path": None,
            "error": None,
            "summary": {},
        }

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.data)


STATE = BatchState()


def json_response(handler, payload, status=200):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_json(handler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def resolve_under(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ValueError("Path escapes the project directory")
    return target


def list_checkpoints() -> list[str]:
    if not CHECKPOINT_ROOT.exists():
        return []
    return [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in sorted(CHECKPOINT_ROOT.rglob("*.pth"), key=lambda item: item.stat().st_mtime, reverse=True)
    ]


def image_counts() -> dict[str, int]:
    counts = {}
    for split in SPLITS:
        folder = DATA_ROOT / split
        counts[split] = (
            sum(1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            if folder.exists()
            else 0
        )
    counts["all"] = sum(counts.values())
    return counts


def list_images(split: str, limit: int = 0) -> list[tuple[str, Path]]:
    selected_splits = SPLITS if split == "all" else (split,)
    images = []
    for split_name in selected_splits:
        folder = DATA_ROOT / split_name
        if not folder.exists():
            continue
        images.extend(
            (split_name, path)
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return images[:limit] if limit > 0 else images


def empty_totals() -> dict:
    return {
        "pixels": 0,
        "original_bytes": 0,
        "estimated_bits": 0.0,
        "zlib_bytes": 0,
        "rans_bytes": 0,
        "rans_residual_bytes": 0,
        "zlib_dna_nt": 0,
        "rans_dna_nt": 0,
    }


def make_summary(totals: dict, succeeded: int, errors: int) -> dict:
    pixels = max(totals["pixels"], 1)
    original_bytes = max(totals["original_bytes"], 1)
    zlib_bytes = totals["zlib_bytes"]
    rans_bytes = totals["rans_bytes"]
    zlib_dna = totals["zlib_dna_nt"]
    rans_dna = totals["rans_dna_nt"]
    return {
        "succeeded": succeeded,
        "errors": errors,
        "estimated_bpp": totals["estimated_bits"] / pixels,
        "zlib_bpp": zlib_bytes * 8 / pixels,
        "rans_bpp": rans_bytes * 8 / pixels,
        "rans_residual_bpp": totals["rans_residual_bytes"] * 8 / pixels,
        "zlib_size_percent": zlib_bytes / original_bytes * 100.0,
        "rans_size_percent": rans_bytes / original_bytes * 100.0,
        "rans_binary_saving_percent": (zlib_bytes - rans_bytes) / max(zlib_bytes, 1) * 100.0,
        "zlib_dna_nt": zlib_dna,
        "rans_dna_nt": rans_dna,
        "rans_dna_saving_percent": (zlib_dna - rans_dna) / max(zlib_dna, 1) * 100.0,
    }


def evaluate_image(model, device, split: str, path: Path, tau: int, index: int) -> dict:
    started = time.perf_counter()
    x = image_to_tensor(path, model.in_channels).to(device)
    with torch.no_grad():
        out = model(x, tau=tau)

    height = int(x.shape[2])
    width = int(x.shape[3])
    channels = int(x.shape[1])
    pixels = height * width
    original_bytes = pixels * channels
    metadata = {
        "name": path.name,
        "split": split,
        "tau": tau,
        "height": height,
        "width": width,
        "channels": channels,
    }
    y_hat = out.y_hat.cpu()
    q = out.q.cpu()
    estimated_bpp = bits_per_pixel(out.latent_bits, x) + bits_per_pixel(out.residual_bits, x)

    zlib_payload = pack_tensors(y_hat, q, metadata, residual_codec="zlib")
    rans_payload = pack_tensors(
        y_hat,
        q,
        metadata,
        residual_codec="rans",
        residual_logits=out.residual_logits.cpu(),
        max_q=model.residual_entropy.max_q,
    )
    decoded_y, decoded_q, rans_metadata = unpack_tensors(rans_payload)
    roundtrip = torch.equal(torch.round(y_hat), decoded_y) and torch.equal(torch.round(q), decoded_q)
    if not roundtrip:
        raise RuntimeError("rANS round-trip verification failed")

    zlib_dna_nt = len(payload_to_dna(zlib_payload).dna)
    rans_dna_nt = len(payload_to_dna(rans_payload).dna)
    zlib_bytes = len(zlib_payload)
    rans_bytes = len(rans_payload)
    rans_residual_bytes = int(rans_metadata["rans_payload_bytes"])
    return {
        "index": index,
        "split": split,
        "image": path.name,
        "width": width,
        "height": height,
        "channels": channels,
        "estimated_bpp": estimated_bpp,
        "zlib_bytes": zlib_bytes,
        "zlib_bpp": zlib_bytes * 8 / pixels,
        "zlib_size_percent": zlib_bytes / original_bytes * 100.0,
        "zlib_dna_nt": zlib_dna_nt,
        "rans_bytes": rans_bytes,
        "rans_bpp": rans_bytes * 8 / pixels,
        "rans_residual_bpp": rans_residual_bytes * 8 / pixels,
        "rans_size_percent": rans_bytes / original_bytes * 100.0,
        "rans_dna_nt": rans_dna_nt,
        "rans_binary_saving_percent": (zlib_bytes - rans_bytes) / max(zlib_bytes, 1) * 100.0,
        "rans_dna_saving_percent": (zlib_dna_nt - rans_dna_nt) / max(zlib_dna_nt, 1) * 100.0,
        "rans_roundtrip": True,
        "seconds": time.perf_counter() - started,
        "error": "",
        "_pixels": pixels,
        "_original_bytes": original_bytes,
        "_rans_residual_bytes": rans_residual_bytes,
    }


def run_job(checkpoint_rel: str, split: str, tau: int, limit: int, job_id: str) -> None:
    started = time.perf_counter()
    totals = empty_totals()
    succeeded = 0
    errors = 0
    try:
        checkpoint_path = resolve_under(ROOT, checkpoint_rel)
        images = list_images(split, limit)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_rel}")
        if not images:
            raise FileNotFoundError(f"No images found for split: {split}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_from_checkpoint(checkpoint_path, device)
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_ROOT / f"codec_comparison_{job_id}.csv"

        with STATE.lock:
            STATE.data.update(
                {
                    "device": str(device),
                    "total": len(images),
                    "report_path": str(report_path.relative_to(ROOT)).replace("\\", "/"),
                }
            )

        with report_path.open("w", newline="", encoding="utf-8-sig") as report:
            writer = csv.DictWriter(report, fieldnames=CSV_FIELDS)
            writer.writeheader()
            report.flush()

            for index, (split_name, image_path) in enumerate(images, start=1):
                if STATE.cancel_event.is_set():
                    break
                with STATE.lock:
                    STATE.data["current_image"] = f"{split_name}/{image_path.name}"
                try:
                    result = evaluate_image(model, device, split_name, image_path, tau, index)
                    succeeded += 1
                    totals["pixels"] += result.pop("_pixels")
                    totals["original_bytes"] += result.pop("_original_bytes")
                    totals["estimated_bits"] += result["estimated_bpp"] * (result["width"] * result["height"])
                    totals["zlib_bytes"] += result["zlib_bytes"]
                    totals["rans_bytes"] += result["rans_bytes"]
                    totals["rans_residual_bytes"] += result.pop("_rans_residual_bytes")
                    totals["zlib_dna_nt"] += result["zlib_dna_nt"]
                    totals["rans_dna_nt"] += result["rans_dna_nt"]
                except Exception as exc:
                    errors += 1
                    result = {field: "" for field in CSV_FIELDS}
                    result.update(
                        {
                            "index": index,
                            "split": split_name,
                            "image": image_path.name,
                            "rans_roundtrip": False,
                            "error": str(exc),
                        }
                    )

                writer.writerow({field: result.get(field, "") for field in CSV_FIELDS})
                report.flush()
                with STATE.lock:
                    STATE.results.append(result)
                    completed = index
                    elapsed = time.perf_counter() - started
                    STATE.data.update(
                        {
                            "completed": completed,
                            "succeeded": succeeded,
                            "errors": errors,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": elapsed / completed * (len(images) - completed),
                            "summary": make_summary(totals, succeeded, errors),
                        }
                    )

        with STATE.lock:
            cancelled = STATE.cancel_event.is_set()
            STATE.data.update(
                {
                    "status": "cancelled" if cancelled else "completed",
                    "current_image": None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "eta_seconds": 0.0,
                    "summary": make_summary(totals, succeeded, errors),
                }
            )
    except Exception as exc:
        with STATE.lock:
            STATE.data.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - started,
                    "eta_seconds": None,
                }
            )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def start_job(payload: dict) -> dict:
    checkpoint = str(payload.get("checkpoint", ""))
    split = str(payload.get("split", "all"))
    tau = int(payload.get("tau", 5))
    limit = int(payload.get("limit", 0))
    if split not in {"all", *SPLITS}:
        raise ValueError(f"Unsupported split: {split}")
    if not 0 <= tau <= 16:
        raise ValueError("tau must be between 0 and 16")
    if limit < 0:
        raise ValueError("limit cannot be negative")
    if checkpoint not in list_checkpoints():
        raise ValueError("Select a checkpoint under outputs/checkpoints")

    with STATE.lock:
        if STATE.data["status"] == "running":
            raise RuntimeError("A comparison job is already running")
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        STATE.cancel_event = threading.Event()
        STATE.results = []
        STATE.data = {
            **STATE._idle_state(),
            "status": "running",
            "job_id": job_id,
            "checkpoint": checkpoint,
            "split": split,
            "tau": tau,
        }

    thread = threading.Thread(
        target=run_job,
        args=(checkpoint, split, tau, limit, job_id),
        daemon=True,
        name=f"codec-comparison-{job_id}",
    )
    thread.start()
    return STATE.snapshot()


class BatchCompareHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/batch_compare.html"}:
                data = VIEWER_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif parsed.path == "/api/config":
                json_response(
                    self,
                    {"checkpoints": list_checkpoints(), "image_counts": image_counts()},
                )
            elif parsed.path == "/api/status":
                json_response(self, STATE.snapshot())
            elif parsed.path == "/api/results":
                query = parse_qs(parsed.query)
                offset = max(int(query.get("offset", ["0"])[0]), 0)
                limit = min(max(int(query.get("limit", ["100"])[0]), 1), 500)
                with STATE.lock:
                    results = STATE.results[offset : offset + limit]
                    total = len(STATE.results)
                json_response(self, {"results": results, "offset": offset, "limit": limit, "total": total})
            else:
                json_response(self, {"error": "Not found"}, status=404)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/start":
                json_response(self, start_job(read_json(self)), status=202)
            elif parsed.path == "/api/cancel":
                STATE.cancel_event.set()
                json_response(self, {"status": "cancelling"})
            else:
                json_response(self, {"error": "Not found"}, status=404)
        except RuntimeError as exc:
            json_response(self, {"error": str(exc)}, status=409)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=400)

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Batch zlib/rANS comparison viewer.")
    parser.add_argument("--host", default=os.getenv("VAE_BATCH_VIEWER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VAE_BATCH_VIEWER_PORT", "8001")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), BatchCompareHandler)
    print(f"Batch comparison viewer running at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
