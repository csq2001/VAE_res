import argparse
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_TMP = os.path.join(os.getcwd(), "outputs", "tmp")
os.makedirs(PROJECT_TMP, exist_ok=True)
os.environ.setdefault("TMP", PROJECT_TMP)
os.environ.setdefault("TEMP", PROJECT_TMP)
os.environ.setdefault("TMPDIR", PROJECT_TMP)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(PROJECT_TMP, "torchinductor"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import VaeResidualCodec
from utils.dataset import CTImageDataset
from utils.metrics import bits_per_pixel, max_abs_error_pixels, ms_ssim_loss, psnr
from utils.train_utils import load_checkpoint, save_checkpoint, set_seed


STAGES = ("stage1", "stage2", "stage3")


@dataclass
class EpochMetrics:
    loss: float = 0.0
    bpp: float = 0.0
    latent_bpp: float = 0.0
    residual_bpp: float = 0.0
    lossy_psnr: float = 0.0
    psnr: float = 0.0
    maxerr: float = 0.0

    def as_dict(self):
        return self.__dict__


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE lossy codec plus residual entropy model.")
    parser.add_argument("--data-root", default=os.getenv("VAE_DATA_ROOT", "data"))
    parser.add_argument("--strategy", choices=["staged", "joint"], default=os.getenv("VAE_TRAIN_STRATEGY", "staged"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("VAE_EPOCHS", "50")))
    parser.add_argument("--stage1-epochs", type=int, default=int(os.getenv("VAE_STAGE1_EPOCHS", "0")))
    parser.add_argument("--stage2-epochs", type=int, default=int(os.getenv("VAE_STAGE2_EPOCHS", "30")))
    parser.add_argument("--stage3-epochs", type=int, default=int(os.getenv("VAE_STAGE3_EPOCHS", "0")))
    parser.add_argument("--stage3-lr-factor", type=float, default=float(os.getenv("VAE_STAGE3_LR_FACTOR", "0.02")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("VAE_BATCH_SIZE", "16")))
    parser.add_argument("--patch-size", type=int, default=int(os.getenv("VAE_PATCH_SIZE", "256")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("VAE_LR", "1e-4")))
    parser.add_argument("--tau", type=int, default=int(os.getenv("VAE_TAU", "5")))
    parser.add_argument("--lambda-distortion", type=float, default=float(os.getenv("VAE_LAMBDA_DISTORTION", "20.0")))
    parser.add_argument("--lambda-l1", type=float, default=float(os.getenv("VAE_LAMBDA_L1", "2.0")))
    parser.add_argument("--lambda-ms-ssim", type=float, default=float(os.getenv("VAE_LAMBDA_MS_SSIM", "1.0")))
    parser.add_argument("--stage1-latent-weight", type=float, default=float(os.getenv("VAE_STAGE1_LATENT_WEIGHT", "0.05")))
    parser.add_argument("--beta-residual", type=float, default=float(os.getenv("VAE_BETA_RESIDUAL", "0.5")))
    parser.add_argument("--channels", type=int, default=int(os.getenv("VAE_CHANNELS", "3")))
    parser.add_argument("--latent-channels", type=int, default=int(os.getenv("VAE_LATENT_CHANNELS", "64")))
    parser.add_argument("--latent-quant-step", type=float, default=float(os.getenv("VAE_LATENT_QUANT_STEP", "1.0")))
    parser.add_argument("--base-channels", type=int, default=int(os.getenv("VAE_BASE_CHANNELS", "64")))
    parser.add_argument("--residual-condition-channels", type=int, default=int(os.getenv("VAE_RESIDUAL_CONDITION_CHANNELS", "16")))
    parser.add_argument("--residual-extra-blocks", type=int, default=int(os.getenv("VAE_RESIDUAL_EXTRA_BLOCKS", "1")))
    parser.add_argument("--max-q", type=int, default=int(os.getenv("VAE_MAX_Q", "64")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("VAE_NUM_WORKERS", "2")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("VAE_SEED", "42")))
    parser.add_argument("--log-interval", type=int, default=int(os.getenv("VAE_LOG_INTERVAL", "20")))
    parser.add_argument("--checkpoint", default=os.getenv("VAE_CHECKPOINT", "outputs/checkpoints/best.pth"))
    parser.add_argument("--stage1-checkpoint", default=os.getenv("VAE_STAGE1_CHECKPOINT", "outputs/checkpoints/checkpoint_stage1.pth"))
    parser.add_argument("--stage2-checkpoint", default=os.getenv("VAE_STAGE2_CHECKPOINT", "outputs/checkpoints/checkpoint_stage2.pth"))
    parser.add_argument("--resume-stage1", default=os.getenv("VAE_RESUME_STAGE1", ""))
    parser.add_argument("--save-metric", choices=["bpp", "loss", "lossy_psnr"], default=os.getenv("VAE_SAVE_METRIC", "bpp"))
    return parser.parse_args()


def make_loader(root, split, args, training):
    dataset = CTImageDataset(Path(root) / split, patch_size=args.patch_size, training=training, channels=args.channels)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_model(args, device):
    return VaeResidualCodec(
        in_channels=args.channels,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        latent_quant_step=args.latent_quant_step,
        residual_condition_channels=args.residual_condition_channels,
        residual_extra_blocks=args.residual_extra_blocks,
        max_q=args.max_q,
    ).to(device)


def set_stage_trainable(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if stage == "stage1":
        modules = [model.encoder, model.decoder, model.prior]
    elif stage == "stage2":
        modules = [model.residual_entropy]
        if model.residual_condition is not None:
            modules.append(model.residual_condition)
    elif stage in {"stage3", "joint"}:
        modules = [model]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def make_optimizer(model, lr):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters were selected.")
    return torch.optim.Adam(parameters, lr=lr)


def reconstruction_loss(out, x, args):
    return (
        args.lambda_distortion * F.mse_loss(out.x_tilde, x)
        + args.lambda_l1 * F.l1_loss(out.x_tilde, x)
        + args.lambda_ms_ssim * ms_ssim_loss(x, out.x_tilde)
    )


def loss_for_stage(out, x, args, stage):
    pixels = x.shape[2] * x.shape[3]
    latent_rate = out.latent_bits.mean() / pixels
    residual_rate = out.residual_bits.mean() / pixels
    recon = reconstruction_loss(out, x, args)

    if stage == "stage1":
        return recon + args.stage1_latent_weight * latent_rate
    if stage == "stage2":
        return residual_rate
    return recon + latent_rate + args.beta_residual * residual_rate


def metric_name_for_stage(stage, args):
    if stage == "stage1":
        return "lossy_psnr"
    if stage == "stage2":
        return "residual_bpp"
    return args.save_metric


def better(metric_name, current, best):
    if metric_name == "lossy_psnr":
        return current > best
    return current < best


def initial_best(metric_name):
    return -float("inf") if metric_name == "lossy_psnr" else float("inf")


def run_epoch(model, loader, optimizer, device, args, stage, epoch, training):
    model.train(training)
    total = EpochMetrics()
    count = 0
    phase = "train" if training else "val"
    total_batches = len(loader)
    deterministic_latent = stage == "stage2" or not training

    for batch_idx, (x, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            out = model(x, tau=args.tau, deterministic=deterministic_latent)
            loss = loss_for_stage(out, x, args, stage)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

        latent_bpp = bits_per_pixel(out.latent_bits, x)
        residual_bpp = bits_per_pixel(out.residual_bits, x)
        total.loss += loss.item()
        total.latent_bpp += latent_bpp
        total.residual_bpp += residual_bpp
        total.bpp += latent_bpp + residual_bpp
        total.lossy_psnr += psnr(x, out.x_tilde)
        total.psnr += psnr(x, out.x_hat)
        total.maxerr += max_abs_error_pixels(x, out.x_hat)
        count += 1

        if args.log_interval > 0 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % args.log_interval == 0):
            print(
                f"stage={stage} epoch={epoch:03d} phase={phase} batch={batch_idx:04d}/{total_batches:04d} "
                f"loss={loss.item():.4f} bpp={latent_bpp + residual_bpp:.4f} "
                f"latent={latent_bpp:.4f} residual={residual_bpp:.4f} "
                f"lossy_psnr={psnr(x, out.x_tilde):.2f} psnr={psnr(x, out.x_hat):.2f} "
                f"maxerr={max_abs_error_pixels(x, out.x_hat):.1f}",
                flush=True,
            )

    averaged = EpochMetrics(**{key: value / max(count, 1) for key, value in total.as_dict().items()})
    return averaged.as_dict()


def print_summary(stage, epoch, train_metrics, val_metrics):
    print(
        f"stage={stage} epoch={epoch:03d} "
        f"train_loss={train_metrics['loss']:.4f} train_bpp={train_metrics['bpp']:.4f} "
        f"train_latent={train_metrics['latent_bpp']:.4f} train_residual={train_metrics['residual_bpp']:.4f} "
        f"train_lossy_psnr={train_metrics['lossy_psnr']:.2f} "
        f"val_loss={val_metrics['loss']:.4f} val_bpp={val_metrics['bpp']:.4f} "
        f"val_latent={val_metrics['latent_bpp']:.4f} val_residual={val_metrics['residual_bpp']:.4f} "
        f"val_lossy_psnr={val_metrics['lossy_psnr']:.2f} "
        f"val_psnr={val_metrics['psnr']:.2f} val_maxerr={val_metrics['maxerr']:.1f}",
        flush=True,
    )


def train_stage(model, train_loader, val_loader, device, args, stage, epochs, lr, checkpoint_path):
    set_stage_trainable(model, stage)
    optimizer = make_optimizer(model, lr)
    metric_name = metric_name_for_stage(stage, args)
    best = initial_best(metric_name)
    checkpoint_exists = Path(checkpoint_path).exists()
    saved = checkpoint_exists

    baseline = run_epoch(model, val_loader, optimizer, device, args, stage, 0, training=False)
    best = baseline[metric_name]
    print(
        f"baseline stage={stage} metric={metric_name} value={best:.4f} "
        f"bpp={baseline['bpp']:.4f} latent={baseline['latent_bpp']:.4f} residual={baseline['residual_bpp']:.4f}",
        flush=True,
    )

    if stage in {"stage2", "stage3"}:
        save_checkpoint(checkpoint_path, model, optimizer, 0, args)
        saved = True
        print(f"saved baseline {checkpoint_path}", flush=True)

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, stage, epoch, training=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, stage, epoch, training=False)
        print_summary(stage, epoch, train_metrics, val_metrics)

        current = val_metrics[metric_name]
        if better(metric_name, current, best):
            best = current
            save_checkpoint(checkpoint_path, model, optimizer, epoch, args)
            saved = True
            print(f"saved {checkpoint_path} stage={stage} metric={metric_name} value={best:.4f}", flush=True)

    if not saved:
        save_checkpoint(checkpoint_path, model, optimizer, max(epochs, 0), args)
        print(f"saved fallback {checkpoint_path}", flush=True)

    load_checkpoint(checkpoint_path, model, map_location=device)
    print(f"loaded best stage checkpoint {checkpoint_path}", flush=True)


def maybe_resume_stage1(model, args, device):
    if not args.resume_stage1:
        return
    resume_path = Path(args.resume_stage1)
    if not resume_path.exists():
        print(f"resume stage1 skipped: {resume_path} does not exist", flush=True)
        return
    load_checkpoint(resume_path, model, map_location=device)
    print(f"resumed stage1 from {resume_path}", flush=True)


def save_timestamped_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"skip timestamped checkpoint: {checkpoint_path} does not exist", flush=True)
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_{timestamp}{checkpoint_path.suffix}")
    timestamped_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, timestamped_path)
    print(f"saved timestamped checkpoint {timestamped_path}", flush=True)
    return timestamped_path


def print_config(args, device):
    print(f"device={device} cuda_available={torch.cuda.is_available()}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"config strategy={args.strategy} epochs={args.epochs} "
        f"stage1_epochs={args.stage1_epochs} stage2_epochs={args.stage2_epochs} stage3_epochs={args.stage3_epochs} "
        f"batch_size={args.batch_size} patch_size={args.patch_size} lr={args.lr} tau={args.tau} "
        f"channels={args.channels} "
        f"lambda_distortion={args.lambda_distortion} lambda_l1={args.lambda_l1} "
        f"lambda_ms_ssim={args.lambda_ms_ssim} beta_residual={args.beta_residual} "
        f"save_metric={args.save_metric}",
        flush=True,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_config(args, device)

    train_loader = make_loader(args.data_root, "train", args, training=True)
    val_loader = make_loader(args.data_root, "val", args, training=False)
    model = make_model(args, device)
    maybe_resume_stage1(model, args, device)

    if args.strategy == "joint":
        train_stage(model, train_loader, val_loader, device, args, "joint", args.epochs, args.lr, args.checkpoint)
        save_timestamped_checkpoint(args.checkpoint)
        return

    train_stage(model, train_loader, val_loader, device, args, "stage1", args.stage1_epochs, args.lr, args.stage1_checkpoint)
    train_stage(model, train_loader, val_loader, device, args, "stage2", args.stage2_epochs, args.lr, args.stage2_checkpoint)
    train_stage(
        model,
        train_loader,
        val_loader,
        device,
        args,
        "stage3",
        args.stage3_epochs,
        args.lr * args.stage3_lr_factor,
        args.checkpoint,
    )
    save_timestamped_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
