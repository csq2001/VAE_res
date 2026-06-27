import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader

from models import VaeResidualCodec
from utils.bitstream import pack_tensors, payload_to_dna, unpack_tensors, write_dna_fasta
from utils.dataset import CTImageDataset
from utils.metrics import bits_per_pixel, max_abs_error_pixels, ms_ssim, psnr


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and export VAE residual near-lossless DNA streams.")
    parser.add_argument("--data-root", default=os.getenv("VAE_DATA_ROOT", "data"))
    parser.add_argument("--split", default=os.getenv("VAE_SPLIT", "test"))
    parser.add_argument("--checkpoint", default=os.getenv("VAE_CHECKPOINT", "outputs/checkpoints/best.pth"))
    parser.add_argument("--tau", type=int, default=int(os.getenv("VAE_TAU", "2")))
    parser.add_argument("--channels", type=int, default=int(os.getenv("VAE_CHANNELS", "3")))
    parser.add_argument("--latent-channels", type=int, default=int(os.getenv("VAE_LATENT_CHANNELS", "64")))
    parser.add_argument("--base-channels", type=int, default=int(os.getenv("VAE_BASE_CHANNELS", "64")))
    parser.add_argument("--max-q", type=int, default=int(os.getenv("VAE_MAX_Q", "64")))
    parser.add_argument("--export-dna", action="store_true")
    parser.add_argument(
        "--residual-codec",
        choices=["zlib", "rans", "both"],
        default=os.getenv("VAE_RESIDUAL_CODEC", "zlib"),
    )
    parser.add_argument(
        "--rans-precision",
        type=int,
        default=int(os.getenv("VAE_RANS_PRECISION", "16")),
    )
    parser.add_argument("--out-dir", default=os.getenv("VAE_DNA_OUT_DIR", "outputs/dna"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cuda_available={torch.cuda.is_available()}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    state = checkpoint["model"]
    channels = int(ckpt_args.get("channels", state.get("encoder.net.0.weight", torch.empty(0, args.channels)).shape[1]))
    dataset = CTImageDataset(Path(args.data_root) / args.split, patch_size=None, training=False, channels=channels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    legacy_condition = "residual_condition.0.weight" not in state
    checkerboard_context = any(
        key.startswith("residual_entropy.context_net.") for key in state
    )
    if "prior.loc" not in state:
        latent_channels = int(ckpt_args.get("latent_channels", args.latent_channels))
        state["prior.loc"] = torch.zeros(latent_channels)
    model = VaeResidualCodec(
        in_channels=channels,
        latent_channels=int(ckpt_args.get("latent_channels", args.latent_channels)),
        base_channels=int(ckpt_args.get("base_channels", args.base_channels)),
        latent_quant_step=float(ckpt_args.get("latent_quant_step", 1.0)),
        residual_condition_channels=0 if legacy_condition else int(ckpt_args.get("residual_condition_channels", 16)),
        residual_extra_blocks=0 if legacy_condition else int(ckpt_args.get("residual_extra_blocks", 1)),
        max_q=int(ckpt_args.get("max_q", args.max_q)),
        checkerboard_context=checkerboard_context,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    totals = {"bpp": 0.0, "latent_bpp": 0.0, "residual_bpp": 0.0, "psnr": 0.0, "msssim": 0.0, "maxerr": 0.0}
    out_dir = Path(args.out_dir)
    if args.export_dna:
        out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index, (x, names) in enumerate(loader):
            x = x.to(device)
            out = model(x, tau=args.tau)
            latent_bpp = bits_per_pixel(out.latent_bits, x)
            residual_bpp = bits_per_pixel(out.residual_bits, x)
            totals["latent_bpp"] += latent_bpp
            totals["residual_bpp"] += residual_bpp
            totals["bpp"] += latent_bpp + residual_bpp
            totals["psnr"] += psnr(x, out.x_hat)
            totals["msssim"] += ms_ssim(x, out.x_hat)
            totals["maxerr"] += max_abs_error_pixels(x, out.x_hat)

            if args.export_dna and index < 5:
                metadata = {
                    "name": names[0],
                    "tau": args.tau,
                    "height": int(x.shape[2]),
                    "width": int(x.shape[3]),
                    "channels": int(x.shape[1]),
                }
                codecs = ["zlib", "rans"] if args.residual_codec == "both" else [args.residual_codec]
                for residual_codec in codecs:
                    payload = pack_tensors(
                        out.y_hat.cpu(),
                        out.q.cpu(),
                        metadata,
                        residual_codec=residual_codec,
                        residual_logits=out.residual_logits.cpu() if residual_codec == "rans" else None,
                        max_q=model.residual_entropy.max_q,
                        rans_precision=args.rans_precision,
                        checkerboard_context=model.residual_entropy.checkerboard_context,
                    )
                    decoded_y, decoded_q, _ = unpack_tensors(payload)
                    if not torch.equal(torch.round(out.y_hat.cpu()), decoded_y):
                        raise RuntimeError(f"{residual_codec} latent round-trip verification failed")
                    if not torch.equal(torch.round(out.q.cpu()), decoded_q):
                        raise RuntimeError(f"{residual_codec} residual round-trip verification failed")

                    packed = payload_to_dna(payload)
                    stem = Path(names[0]).stem
                    fasta = out_dir / f"{stem}_{residual_codec}.fasta"
                    write_dna_fasta(packed.dna, fasta)
                    actual_bpp = len(payload) * 8 / (int(x.shape[2]) * int(x.shape[3]))
                    print(
                        f"exported {fasta} codec={residual_codec} "
                        f"payload_bpp={actual_bpp:.4f} {packed.metadata}"
                    )

    n = len(loader)
    print(
        f"split={args.split} n={n} "
        f"bpp={totals['bpp']/n:.4f} latent={totals['latent_bpp']/n:.4f} "
        f"residual={totals['residual_bpp']/n:.4f} psnr={totals['psnr']/n:.2f} "
        f"msssim={totals['msssim']/n:.4f} maxerr={totals['maxerr']/n:.1f}"
    )


if __name__ == "__main__":
    main()
