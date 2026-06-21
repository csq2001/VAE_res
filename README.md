# VAE Residual Near-lossless Codec for DNA Storage

This project trains a joint codec for CT images:

1. A VAE produces a high-quality lossy approximation `x_tilde`.
2. A learned residual entropy model estimates `p(q | x_tilde)` for the quantized residual.
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
python train.py --epochs 50 --batch-size 8 --patch-size 256 --tau 2
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
VAE_BATCH_SIZE=16
VAE_PATCH_SIZE=256
VAE_TAU=2
VAE_LR=1e-4
VAE_BETA_RESIDUAL=0.5
VAE_LATENT_QUANT_STEP=1.0
VAE_LAMBDA_DISTORTION=20.0
VAE_LAMBDA_L1=2.0
VAE_LAMBDA_MS_SSIM=1.0
VAE_CHECKPOINT=outputs/checkpoints/best.pth
VAE_LOG_INTERVAL=20
VAE_RESIDUAL_CONDITION_CHANNELS=16
VAE_RESIDUAL_EXTRA_BLOCKS=1
VAE_SAVE_METRIC=lossy_psnr
```

Training automatically uses CUDA when a GPU is available; otherwise it falls back to CPU.

## Evaluate

```bash
python test.py --checkpoint outputs/checkpoints/best.pth --tau 2
```

## Export Example DNA Streams

```bash
python test.py --checkpoint outputs/checkpoints/best.pth --tau 2 --export-dna
```

The learned entropy model is used during training to optimize the residual bitrate. The current exported payload uses compressed tensor packing as a runnable baseline; the `pack_tensors` boundary is where arithmetic coding or rANS can be added later using the model probabilities.

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
