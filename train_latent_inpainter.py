from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import LatentInpainter
from models.vae import quantize_image_ste
from utils.dataset import CTImageDataset
from utils.metrics import ms_ssim_loss
from viewer_server import model_from_checkpoint


ROOT = Path(__file__).resolve().parent


@dataclass
class LossValues:
    total: float = 0.0
    latent: float = 0.0
    lossy: float = 0.0
    final: float = 0.0
    structural: float = 0.0
    uncertainty: float = 0.0
    batches: int = 0

    def update(self, values: dict[str, torch.Tensor]) -> None:
        for name, value in values.items():
            setattr(self, name, getattr(self, name) + float(value.detach().item()))
        self.batches += 1

    def average(self) -> dict[str, float]:
        count = max(self.batches, 1)
        return {
            name: round(getattr(self, name) / count, 6)
            for name in ("total", "latent", "lossy", "final", "structural", "uncertainty")
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the mask-aware latent inpainter.")
    parser.add_argument("--vae-checkpoint", default="outputs/checkpoints/best.pth")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="outputs/checkpoints/latent_inpainter_best.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--channel-group", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--context-channels", type=int, default=128)
    parser.add_argument(
        "--image-loss-interval",
        type=int,
        default=4,
        help="Compute decoder/image losses every N training batches.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=32)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Print running progress and average losses every N batches.",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA mixed precision.",
    )
    return parser.parse_args()


def make_loader(
    root: Path,
    split: str,
    *,
    channels: int,
    patch_size: int,
    batch_size: int,
    workers: int,
    training: bool,
) -> DataLoader:
    dataset = CTImageDataset(
        root / split,
        patch_size=patch_size,
        training=training,
        channels=channels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training and len(dataset) >= batch_size,
    )


def latent_tile_specs(
    channels: int,
    height: int,
    width: int,
    tile_size: int,
    channel_group: int,
) -> list[tuple[int, int, int, int, int, int]]:
    specs = []
    for channel in range(0, channels, channel_group):
        count = min(channel_group, channels - channel)
        for top in range(0, height, tile_size):
            tile_height = min(tile_size, height - top)
            for left in range(0, width, tile_size):
                tile_width = min(tile_size, width - left)
                specs.append((channel, count, top, tile_height, left, tile_width))
    return specs


def make_training_mask(
    latent: torch.Tensor,
    tile_size: int,
    channel_group: int,
    rng: random.Random,
) -> torch.Tensor:
    batch, channels, height, width = latent.shape
    specs = latent_tile_specs(channels, height, width, tile_size, channel_group)
    mask = torch.ones_like(latent)
    for batch_index in range(batch):
        mode = rng.random()
        missing: set[int] = set()
        if mode < 0.2:
            missing.update(rng.sample(range(len(specs)), k=min(rng.randint(1, 2), len(specs))))
        elif mode < 0.6:
            group_start = rng.randrange(0, len(specs), 8)
            group = list(range(group_start, min(group_start + 8, len(specs))))
            missing.update(rng.sample(group, k=min(rng.randint(5, 8), len(group))))
        elif mode < 0.9:
            anchor = rng.randrange(len(specs))
            channel, _, top, _, left, _ = specs[anchor]
            for index, spec in enumerate(specs):
                if (
                    spec[0] == channel
                    and abs(spec[2] - top) <= tile_size
                    and abs(spec[4] - left) <= tile_size
                ):
                    missing.add(index)
        else:
            count = min(rng.randint(3, 8), len(specs))
            missing.update(rng.sample(range(len(specs)), k=count))
        for index in missing:
            channel, count, top, tile_height, left, tile_width = specs[index]
            mask[
                batch_index,
                channel : channel + count,
                top : top + tile_height,
                left : left + tile_width,
            ] = 0.0
    return mask


def pad_to_eight(images: torch.Tensor) -> torch.Tensor:
    height, width = images.shape[-2:]
    return F.pad(images, (0, (-width) % 8, 0, (-height) % 8))


def compute_losses(
    images: torch.Tensor,
    vae: torch.nn.Module,
    inpainter: LatentInpainter,
    args: argparse.Namespace,
    rng: random.Random,
    compute_image_losses: bool,
    image_loss_scale: float,
) -> dict[str, torch.Tensor]:
    images = pad_to_eight(images)
    with torch.no_grad():
        clean_latent = vae.quantize_latent(
            vae.encoder(images),
            deterministic=True,
        )
        if compute_image_losses:
            clean_lossy = vae.decoder(clean_latent)
            clean_pixels = quantize_image_ste(clean_lossy)
            image_pixels = torch.round(images.clamp(0, 1) * 255.0)
            step = float(2 * args.tau + 1)
            q = torch.round((image_pixels - clean_pixels) / step).clamp(
                -vae.residual_entropy.max_q,
                vae.residual_entropy.max_q,
            )
    valid_mask = make_training_mask(
        clean_latent,
        args.tile_size,
        args.channel_group,
        rng,
    )
    prior_loc = vae.prior.loc.detach()
    prior_scale = F.softplus(vae.prior.log_scale.detach()) + 1e-5
    filled = (
        valid_mask * clean_latent
        + (1.0 - valid_mask) * prior_loc.reshape(1, -1, 1, 1)
    )
    output = inpainter(filled, valid_mask, prior_loc, prior_scale)
    missing = 1.0 - valid_mask
    latent_loss = (
        F.smooth_l1_loss(output.repaired, clean_latent, reduction="none") * missing
    ).sum() / missing.sum().clamp_min(1.0)

    zero = latent_loss.new_zeros(())
    if compute_image_losses:
        repaired_lossy = vae.decoder(output.repaired)
        lossy_loss = F.l1_loss(repaired_lossy, clean_lossy)
        repaired_pixels = quantize_image_ste(repaired_lossy)
        repaired_final = (repaired_pixels + q * step).clamp(0, 255) / 255.0
        final_loss = F.l1_loss(repaired_final, images)
        structural_loss = ms_ssim_loss(repaired_final, images)
    else:
        lossy_loss = final_loss = structural_loss = zero

    group_errors: list[torch.Tensor] = []
    group_missing: list[torch.Tensor] = []
    for start in range(0, clean_latent.shape[1], args.channel_group):
        end = min(start + args.channel_group, clean_latent.shape[1])
        group_errors.append(
            (output.predicted[:, start:end] - clean_latent[:, start:end])
            .abs()
            .mean(dim=1, keepdim=True)
        )
        group_missing.append(missing[:, start:end].amax(dim=1, keepdim=True))
    error = torch.cat(group_errors, dim=1)
    missing_groups = torch.cat(group_missing, dim=1)
    uncertainty_loss = (
        (torch.exp(-output.uncertainty) * error + output.uncertainty)
        * missing_groups
    ).sum() / missing_groups.sum().clamp_min(1.0)

    image_scale = image_loss_scale if compute_image_losses else 0.0
    total = (
        latent_loss
        + image_scale
        * (0.5 * lossy_loss + 0.2 * final_loss + 0.1 * structural_loss)
        + 0.05 * uncertainty_loss
    )
    return {
        "total": total,
        "latent": latent_loss,
        "lossy": lossy_loss,
        "final": final_loss,
        "structural": structural_loss,
        "uncertainty": uncertainty_loss,
    }


def run_epoch(
    loader: DataLoader,
    vae: torch.nn.Module,
    inpainter: LatentInpainter,
    args: argparse.Namespace,
    device: torch.device,
    rng: random.Random,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    max_batches: int | None,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    training = optimizer is not None
    inpainter.train(training)
    metrics = LossValues()
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    started_at = time.perf_counter()
    print(
        f"[Epoch {epoch:02d}/{args.epochs:02d}] {phase} started "
        f"({total_batches} batches)",
        flush=True,
    )
    for batch_index, (images, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        compute_images = (
            not training or batch_index % args.image_loss_interval == 0
        )
        amp_enabled = args.amp and device.type == "cuda"
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            losses = compute_losses(
                images,
                vae,
                inpainter,
                args,
                rng,
                compute_image_losses=compute_images,
                image_loss_scale=(
                    float(args.image_loss_interval) if training else 1.0
                ),
            )
            if training:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(inpainter.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        metrics.update(losses)
        completed = batch_index + 1
        should_log = (
            completed % args.log_interval == 0 or completed == total_batches
        )
        if should_log:
            elapsed = time.perf_counter() - started_at
            eta = elapsed / completed * max(total_batches - completed, 0)
            average = metrics.average()
            memory = ""
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                memory = f" | GPU {allocated:.2f}/{reserved:.2f} GB"
            print(
                f"[Epoch {epoch:02d}/{args.epochs:02d}] {phase} "
                f"{completed:04d}/{total_batches:04d} "
                f"| loss {average['total']:.6f} "
                f"| latent {average['latent']:.6f} "
                f"| lossy {average['lossy']:.6f} "
                f"| final {average['final']:.6f} "
                f"| ssim {average['structural']:.6f} "
                f"| uncertainty {average['uncertainty']:.6f} "
                f"| elapsed {elapsed / 60:.1f}m "
                f"| ETA {eta / 60:.1f}m"
                f"{memory}",
                flush=True,
            )
    return metrics.average()


def save_model(
    path: Path,
    inpainter: LatentInpainter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": inpainter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "config": {
                "latent_channels": inpainter.latent_channels,
                "channel_group": inpainter.channel_group,
                "hidden_channels": args.hidden_channels,
                "context_channels": args.context_channels,
                "quant_step": inpainter.quant_step,
                "tile_size": args.tile_size,
                "mask_convention": "1=valid, 0=missing",
                "vae_checkpoint": args.vae_checkpoint,
            },
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.image_loss_interval < 1:
        raise ValueError("--image-loss-interval must be positive")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    vae_path = (ROOT / args.vae_checkpoint).resolve()
    vae = model_from_checkpoint(vae_path, device)
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    inpainter = LatentInpainter(
        latent_channels=vae.prior.loc.numel(),
        channel_group=args.channel_group,
        hidden_channels=args.hidden_channels,
        context_channels=args.context_channels,
        quant_step=vae.latent_quant_step,
    ).to(device)
    optimizer = torch.optim.AdamW(
        inpainter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
    )
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        inpainter.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    data_root = (ROOT / args.data_root).resolve()
    train_loader = make_loader(
        data_root,
        "train",
        channels=vae.in_channels,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        workers=args.num_workers,
        training=True,
    )
    val_loader = make_loader(
        data_root,
        "val",
        channels=vae.in_channels,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        workers=args.num_workers,
        training=False,
    )
    output = (ROOT / args.output).resolve()
    last_output = output.with_name(f"{output.stem}_last{output.suffix}")
    history_path = output.with_suffix(".jsonl")
    train_rng = random.Random(args.seed)
    val_rng = random.Random(args.seed + 1)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader,
            vae,
            inpainter,
            args,
            device,
            train_rng,
            optimizer,
            scaler,
            args.max_train_batches,
            epoch,
            "train",
        )
        val_metrics = run_epoch(
            val_loader,
            vae,
            inpainter,
            args,
            device,
            val_rng,
            None,
            scaler,
            args.max_val_batches,
            epoch,
            "val",
        )
        improved = val_metrics["total"] < best_loss
        if improved:
            best_loss = val_metrics["total"]
            save_model(output, inpainter, optimizer, epoch, best_loss, args)
        save_model(last_output, inpainter, optimizer, epoch, best_loss, args)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "best": improved,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    print(
        f"Best latent inpainter: {output} (val_loss={best_loss:.6f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
