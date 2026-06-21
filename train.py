import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import VaeResidualCodec
from utils.dataset import CTImageDataset
from utils.metrics import bits_per_pixel, max_abs_error_pixels, ms_ssim_loss, psnr
from utils.train_utils import save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE + learned residual entropy model.")
    parser.add_argument("--data-root", default=os.getenv("VAE_DATA_ROOT", "data"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("VAE_EPOCHS", "50")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("VAE_BATCH_SIZE", "8")))
    parser.add_argument("--patch-size", type=int, default=int(os.getenv("VAE_PATCH_SIZE", "256")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("VAE_LR", "1e-4")))
    parser.add_argument("--tau", type=int, default=int(os.getenv("VAE_TAU", "2")))
    parser.add_argument("--lambda-distortion", type=float, default=float(os.getenv("VAE_LAMBDA_DISTORTION", "20.0")))
    parser.add_argument("--lambda-l1", type=float, default=float(os.getenv("VAE_LAMBDA_L1", "2.0")))
    parser.add_argument("--lambda-ms-ssim", type=float, default=float(os.getenv("VAE_LAMBDA_MS_SSIM", "1.0")))
    parser.add_argument("--beta-residual", type=float, default=float(os.getenv("VAE_BETA_RESIDUAL", "0.5")))
    parser.add_argument("--latent-channels", type=int, default=int(os.getenv("VAE_LATENT_CHANNELS", "64")))
    parser.add_argument("--base-channels", type=int, default=int(os.getenv("VAE_BASE_CHANNELS", "64")))
    parser.add_argument("--latent-quant-step", type=float, default=float(os.getenv("VAE_LATENT_QUANT_STEP", "1.0")))
    parser.add_argument("--residual-condition-channels", type=int, default=int(os.getenv("VAE_RESIDUAL_CONDITION_CHANNELS", "16")))
    parser.add_argument("--residual-extra-blocks", type=int, default=int(os.getenv("VAE_RESIDUAL_EXTRA_BLOCKS", "1")))
    parser.add_argument("--max-q", type=int, default=int(os.getenv("VAE_MAX_Q", "64")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("VAE_NUM_WORKERS", "0")))
    parser.add_argument("--checkpoint", default=os.getenv("VAE_CHECKPOINT", "outputs/checkpoints/best.pth"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("VAE_SEED", "42")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("VAE_LOG_INTERVAL", "10")))
    parser.add_argument("--save-metric", choices=["bpp", "loss", "lossy_psnr"], default=os.getenv("VAE_SAVE_METRIC", "lossy_psnr"))
    return parser.parse_args()


def make_loader(root, split, args, training):
    dataset = CTImageDataset(Path(root) / split, patch_size=args.patch_size, training=training, channels=1)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, optimizer, device, args, training, epoch):
    model.train(training)
    totals = {
        "loss": 0.0,
        "bpp": 0.0,
        "latent_bpp": 0.0,
        "residual_bpp": 0.0,
        "psnr": 0.0,
        "lossy_psnr": 0.0,
        "maxerr": 0.0,
    }
    count = 0
    phase = "train" if training else "val"
    total_batches = len(loader)
    for batch_idx, (x, _) in enumerate(loader, start=1):
        x = x.to(device)
        with torch.set_grad_enabled(training):
            out = model(x, tau=args.tau)
            distortion = F.mse_loss(out.x_tilde, x)
            l1_distortion = F.l1_loss(out.x_tilde, x)
            structure_distortion = ms_ssim_loss(x, out.x_tilde)
            latent_bpp = bits_per_pixel(out.latent_bits, x)
            residual_bpp = bits_per_pixel(out.residual_bits, x)
            rate_loss = (out.latent_bits + args.beta_residual * out.residual_bits).mean() / (x.shape[2] * x.shape[3])
            loss = (
                rate_loss
                + args.lambda_distortion * distortion
                + args.lambda_l1 * l1_distortion
                + args.lambda_ms_ssim * structure_distortion
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        totals["loss"] += loss.item()
        totals["latent_bpp"] += latent_bpp
        totals["residual_bpp"] += residual_bpp
        totals["bpp"] += latent_bpp + residual_bpp
        totals["psnr"] += psnr(x, out.x_hat)
        totals["lossy_psnr"] += psnr(x, out.x_tilde)
        totals["maxerr"] += max_abs_error_pixels(x, out.x_hat)
        count += 1
        should_log = (
            args.log_interval > 0
            and (batch_idx == 1 or batch_idx == total_batches or batch_idx % args.log_interval == 0)
        )
        if should_log:
            print(
                f"epoch={epoch:03d} phase={phase} batch={batch_idx:04d}/{total_batches:04d} "
                f"loss={loss.item():.4f} bpp={latent_bpp + residual_bpp:.4f} "
                f"latent={latent_bpp:.4f} residual={residual_bpp:.4f} "
                f"lossy_psnr={psnr(x, out.x_tilde):.2f} "
                f"psnr={psnr(x, out.x_hat):.2f} maxerr={max_abs_error_pixels(x, out.x_hat):.1f}",
                flush=True,
            )
    return {key: value / max(count, 1) for key, value in totals.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cuda_available={torch.cuda.is_available()}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"config epochs={args.epochs} batch_size={args.batch_size} patch_size={args.patch_size} "
        f"tau={args.tau} lr={args.lr} lambda_distortion={args.lambda_distortion} "
        f"lambda_l1={args.lambda_l1} lambda_ms_ssim={args.lambda_ms_ssim} "
        f"log_interval={args.log_interval}",
        flush=True,
    )
    train_loader = make_loader(args.data_root, "train", args, True)
    val_loader = make_loader(args.data_root, "val", args, False)
    model = VaeResidualCodec(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        latent_quant_step=args.latent_quant_step,
        residual_condition_channels=args.residual_condition_channels,
        residual_extra_blocks=args.residual_extra_blocks,
        max_q=args.max_q,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = -float("inf") if args.save_metric == "lossy_psnr" else float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, True, epoch)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, False, epoch)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_bpp={val_metrics['bpp']:.4f} "
            f"latent={val_metrics['latent_bpp']:.4f} residual={val_metrics['residual_bpp']:.4f} "
            f"lossy_psnr={val_metrics['lossy_psnr']:.2f} "
            f"psnr={val_metrics['psnr']:.2f} maxerr={val_metrics['maxerr']:.1f}",
            flush=True,
        )
        current = val_metrics[args.save_metric]
        improved = current > best if args.save_metric == "lossy_psnr" else current < best
        if improved:
            best = current
            save_checkpoint(args.checkpoint, model, optimizer, epoch, args)
            print(f"saved {args.checkpoint} metric={args.save_metric} value={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
