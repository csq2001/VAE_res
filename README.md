# VAE Residual Near-lossless Codec for DNA Storage

This project trains a joint codec for CT images:

1. A VAE produces a high-quality lossy approximation `x_tilde`.
2. A checkerboard spatial-context entropy model estimates anchor symbols from
   `x_tilde` and `y_hat`, then estimates non-anchor symbols from the decoded anchors.
3. The latent tensor and residual symbols are packed and mapped to DNA bases for storage experiments.

The near-lossless residual uses a maximum pixel error parameter `tau`:

```text
step = 2 * tau + 1
q = round((x - round(x_tilde)) / step)
x_hat = round(x_tilde) + q * step
```

This gives a strict pixel-domain error bound of approximately `tau` when residual symbols are not clipped.

## Data Layout

```text
data/
  train/
  val/
  test/
```

The current split is:

```text
train: 874
val:   109
test:  110
```

## Train

```bash
python train.py --strategy staged --stage1-epochs 0 --stage2-epochs 25 --stage3-epochs 5 --batch-size 16 --patch-size 256 --tau 5
```

The default training strategy is staged:

```text
Stage 1: train VAE reconstruction quality for 10 epochs, save outputs/checkpoints/checkpoint_stage1.pth
Stage 2: freeze VAE encoder/decoder and train the checkerboard residual entropy model for 10 epochs, save outputs/checkpoints/checkpoint_stage2.pth
Stage 3: optional joint fine-tuning with a smaller learning rate, save outputs/checkpoints/best.pth
```

The scripts below centralize environment-variable configuration and save a live timestamped log under `logs/`.

Linux/mac:

```bash
./train.sh
```

Windows PowerShell:

```powershell
.\train.ps1
```

Windows double-click:

```text
start_train.bat
```

Common environment variables:

```text
VAE_EPOCHS=50
VAE_TRAIN_STRATEGY=staged
VAE_STAGE1_EPOCHS=0
VAE_STAGE2_EPOCHS=25
VAE_STAGE3_EPOCHS=5
VAE_STAGE3_LR_FACTOR=0.05
VAE_BATCH_SIZE=16
VAE_PATCH_SIZE=256
VAE_TAU=5
VAE_LR=1e-4
VAE_BETA_RESIDUAL=0.5
VAE_LATENT_QUANT_STEP=1.0
VAE_LAMBDA_DISTORTION=20.0
VAE_LAMBDA_L1=2.0
VAE_LAMBDA_MS_SSIM=1.0
VAE_STAGE1_LATENT_WEIGHT=0.05
VAE_CHECKPOINT=outputs/checkpoints/best.pth
VAE_STAGE1_CHECKPOINT=outputs/checkpoints/checkpoint_stage1.pth
VAE_STAGE2_CHECKPOINT=outputs/checkpoints/checkpoint_stage2.pth
VAE_LOG_INTERVAL=20
VAE_RESIDUAL_CONDITION_CHANNELS=16
VAE_RESIDUAL_EXTRA_BLOCKS=1
VAE_CHECKERBOARD_CONTEXT=1
VAE_SAVE_METRIC=lossy_psnr
```

Training automatically uses CUDA when a GPU is available; otherwise it falls back to CPU.

## Evaluate

```bash
python test.py --checkpoint outputs/checkpoints/best.pth --tau 5
```

## Export Example DNA Streams

```bash
python test.py --checkpoint outputs/checkpoints/best.pth --tau 5 --export-dna --residual-codec zlib
python test.py --checkpoint outputs/checkpoints/best.pth --tau 5 --export-dna --residual-codec rans
python test.py --checkpoint outputs/checkpoints/best.pth --tau 5 --export-dna --residual-codec both
```

The default `zlib` path remains backward compatible. The optional `rans` path builds
per-image, per-channel CDFs from the learned residual entropy model, encodes `q` with
byte-rANS, and verifies that both `y_hat` and `q` survive an exact round trip before
writing DNA. Use `--residual-codec both` to export and compare both paths.

## Visual Viewer

Start the local viewer:

```bash
python viewer_server.py
```

Then open:

```text
http://127.0.0.1:8000
```

On Windows you can double-click:

```text
start_viewer.bat
```

The page scans `outputs/checkpoints` for `.pth` files and `data/train`, `data/val`, `data/test` for images. It displays the input image, VAE lossy approximation, near-lossless reconstruction, PSNR, MS-SSIM, max error, and bitrate estimates.

## Batch zlib / rANS Comparison

Start the separate batch comparison page:

```bash
python batch_compare_server.py
```

Then open:

```text
http://127.0.0.1:8001
```

On Windows you can double-click:

```text
start_batch_compare.bat
```

Select a checkpoint, dataset split, and `tau`. The background job performs real
zlib and rANS encoding for every selected image, verifies the rANS round trip,
shows live aggregate statistics, and writes per-image results to a timestamped
CSV under `outputs/reports`.
