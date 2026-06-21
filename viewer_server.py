import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from PIL import Image

from models import VaeResidualCodec
from utils.dataset import CTImageDataset
from utils.metrics import bits_per_pixel, max_abs_error_pixels, ms_ssim, psnr


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
CHECKPOINT_ROOT = ROOT / "outputs" / "checkpoints"
VIEWER_HTML = ROOT / "viewer.html"


def json_response(handler, payload, status=200):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    dataset = CTImageDataset(path.parent, patch_size=None, training=False, channels=1)
    return dataset._to_tensor(image).unsqueeze(0)


def tensor_to_data_url(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)[0]
    array = (tensor.squeeze(0).numpy() * 255.0).round().astype("uint8")
    image = Image.fromarray(array, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def resolve_under(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError("Path escapes project root")
    return target


def list_images():
    result = []
    for split in ("train", "val", "test"):
        folder = DATA_ROOT / split
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                result.append({"split": split, "name": path.name, "path": f"{split}/{path.name}"})
    return result


def list_checkpoints():
    if not CHECKPOINT_ROOT.exists():
        return []
    return [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in sorted(CHECKPOINT_ROOT.rglob("*.pth"))
    ]


def model_from_checkpoint(path: Path, device: torch.device) -> VaeResidualCodec:
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint.get("args", {})
    state = checkpoint["model"]
    legacy_condition = "residual_condition.0.weight" not in state
    if "prior.loc" not in state:
        latent_channels = int(args.get("latent_channels", os.getenv("VAE_LATENT_CHANNELS", 64)))
        state["prior.loc"] = torch.zeros(latent_channels)
    model = VaeResidualCodec(
        latent_channels=int(args.get("latent_channels", os.getenv("VAE_LATENT_CHANNELS", 64))),
        base_channels=int(args.get("base_channels", os.getenv("VAE_BASE_CHANNELS", 64))),
        latent_quant_step=float(args.get("latent_quant_step", os.getenv("VAE_LATENT_QUANT_STEP", 1.0))),
        residual_condition_channels=0 if legacy_condition else int(args.get("residual_condition_channels", os.getenv("VAE_RESIDUAL_CONDITION_CHANNELS", 16))),
        residual_extra_blocks=0 if legacy_condition else int(args.get("residual_extra_blocks", os.getenv("VAE_RESIDUAL_EXTRA_BLOCKS", 1))),
        max_q=int(args.get("max_q", os.getenv("VAE_MAX_Q", 64))),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(params):
    checkpoint_rel = params.get("checkpoint", ["outputs/checkpoints/best.pth"])[0]
    image_rel = params.get("image", [""])[0]
    tau = int(params.get("tau", [os.getenv("VAE_TAU", "2")])[0])
    checkpoint_path = resolve_under(ROOT, checkpoint_rel)
    image_path = resolve_under(DATA_ROOT, image_rel)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_rel}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_rel}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_checkpoint(checkpoint_path, device)
    x = image_to_tensor(image_path).to(device)
    with torch.no_grad():
        out = model(x, tau=tau)
        latent_bpp = bits_per_pixel(out.latent_bits, x)
        residual_bpp = bits_per_pixel(out.residual_bits, x)
        return {
            "device": str(device),
            "image": image_rel,
            "checkpoint": checkpoint_rel,
            "tau": tau,
            "metrics": {
                "psnr": psnr(x, out.x_hat),
                "ms_ssim": ms_ssim(x, out.x_hat),
                "max_error": max_abs_error_pixels(x, out.x_hat),
                "latent_bpp": latent_bpp,
                "residual_bpp": residual_bpp,
                "total_bpp": latent_bpp + residual_bpp,
            },
            "images": {
                "input": tensor_to_data_url(x),
                "lossy": tensor_to_data_url(out.x_tilde),
                "reconstruction": tensor_to_data_url(out.x_hat),
            },
        }


class ViewerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/viewer.html"}:
                data = VIEWER_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif parsed.path == "/api/images":
                json_response(self, {"images": list_images()})
            elif parsed.path == "/api/checkpoints":
                json_response(self, {"checkpoints": list_checkpoints()})
            elif parsed.path == "/api/evaluate":
                json_response(self, evaluate(parse_qs(parsed.query)))
            else:
                json_response(self, {"error": "Not found"}, status=404)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))


def main():
    parser = argparse.ArgumentParser(description="Local viewer for VAE residual codec checkpoints.")
    parser.add_argument("--host", default=os.getenv("VAE_VIEWER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VAE_VIEWER_PORT", "8000")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Viewer running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
