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

## Block dual-stream DNA codec

The optional block codec keeps the existing `VaeResidualCodec` unchanged and
adds independently recoverable latent/residual enhancement tiles:

```text
image -> one full-image VAE pass -> global latent + quantized residual
      latent -> 4x4x32 tiles -> edit code + CRC32 -> tile RS(8+4)
      residual q -> 16x16 tiles -> importance-adaptive edit/compact payload
```

Both tile types are compressed independently with zlib and do not use periodic
Markers. Their length-7 DNA inner code has minimum Levenshtein distance three,
so it corrects one substitution, insertion, or deletion in each codeword.
Multiple edits can be corrected when they fall in different codewords. CRC
rejects ambiguous or more heavily damaged tiles.

Every eight latent data tiles receive four systematic GF(256) parity tiles.
Tiles rejected by the inner edit code or CRC are recovered as erasures when no
more than four shards in their group are unavailable. Residual addresses keep
the edit code. High-energy residual payloads keep the full edit code, while
low-energy payloads use a compact two-bit-per-base mapping. Compact-payload
edits are detected by length and CRC and fall back locally instead of incurring
full edit-code redundancy.

The image is padded to a multiple of eight before the fully convolutional VAE
pass and cropped back to its original size after decoding. Tile coordinates
are global; the VAE itself no longer runs independently on 64x64 image blocks.

Run the no-error example:

```bash
python block_dna_example.py
```

Inject low-rate DNA edits:

```bash
python block_dna_example.py --substitution-rate 0.000001 --insertion-rate 0.0000005 --deletion-rate 0.0000005
```

An uncorrectable latent tile is replaced by the rounded learned latent prior;
other latent tiles remain usable. An uncorrectable residual tile is dropped
locally: its `q` values become zero, so that region falls back to the VAE lossy
reconstruction instead of aborting the whole image.

Start the visual error-correction demonstration with
`start_block_dna_viewer.bat`, or run:

```bash
python block_dna_server.py
```

Then open `http://127.0.0.1:8003`.

`reedsolo` is detected when installed and remains available for optional
within-packet symbol coding. Cross-packet erasure recovery uses the bundled
systematic GF(256) implementation, so this feature has no required third-party
dependency.

Run its tests:

```bash
python -m pytest tests/test_block_dna_codec.py -q
```

## Export lossy and residual image datasets

Decompose every image under `data/train`, `data/val`, and `data/test` with
`outputs/checkpoints/best.pth`:

```bash
python export_lossy_residual.py
```

Windows users can double-click `export_lossy_residual.bat`. Results preserve
the source split and relative path:

```text
outputs/best_decomposition/
  lossy/train/...png
  residual/train/...png
  residual/train/...npz
```

Residual PNG files are centered visualizations where 128 means zero. The NPZ
files store exact signed `int16` HWC residuals under the `residual` key, so the
original integer pixels can be reconstructed as `lossy + residual`. Existing complete outputs are
skipped unless `--overwrite` is supplied.

## Train the latent error inpainter

Train the mask-aware gated latent U-Net on top of the frozen
`outputs/checkpoints/best.pth` codec:

```bash
python train_latent_inpainter.py
```

On Windows, double-click `train_latent_inpainter.bat`. The VAE encoder,
decoder, prior, and residual model remain frozen. Training generates isolated,
spatial-burst, and RS(8+4)-overflow masks. The best model is saved as:

```text
outputs/checkpoints/latent_inpainter_best.pth
```

The optimized defaults use 20 epochs, 256x256 crops, batch size 8, CUDA AMP,
TF32, a 96/128-channel repair network, and image-domain losses every four
training batches. Latent supervision is still evaluated on every batch.
Validation uses 32 batches per epoch. A short 512x512 fine-tuning run can be
performed after the default training if full-resolution refinement is needed.

The block DNA viewer loads this checkpoint automatically when its 4x4 spatial
tile and 32-channel group configuration matches the trained model. Before that
checkpoint exists, decoding safely keeps the prior-value fallback.

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
