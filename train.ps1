$ErrorActionPreference = "Stop"

$env:KMP_DUPLICATE_LIB_OK = if ($env:KMP_DUPLICATE_LIB_OK) { $env:KMP_DUPLICATE_LIB_OK } else { "TRUE" }
$env:PYTHONUNBUFFERED = if ($env:PYTHONUNBUFFERED) { $env:PYTHONUNBUFFERED } else { "1" }

$env:VAE_DATA_ROOT = if ($env:VAE_DATA_ROOT) { $env:VAE_DATA_ROOT } else { "data" }
$env:VAE_EPOCHS = if ($env:VAE_EPOCHS) { $env:VAE_EPOCHS } else { "50" }
$env:VAE_BATCH_SIZE = if ($env:VAE_BATCH_SIZE) { $env:VAE_BATCH_SIZE } else { "16" }
$env:VAE_PATCH_SIZE = if ($env:VAE_PATCH_SIZE) { $env:VAE_PATCH_SIZE } else { "256" }
$env:VAE_LR = if ($env:VAE_LR) { $env:VAE_LR } else { "1e-4" }
$env:VAE_TAU = if ($env:VAE_TAU) { $env:VAE_TAU } else { "2" }
$env:VAE_LAMBDA_DISTORTION = if ($env:VAE_LAMBDA_DISTORTION) { $env:VAE_LAMBDA_DISTORTION } else { "20.0" }
$env:VAE_LAMBDA_L1 = if ($env:VAE_LAMBDA_L1) { $env:VAE_LAMBDA_L1 } else { "2.0" }
$env:VAE_LAMBDA_MS_SSIM = if ($env:VAE_LAMBDA_MS_SSIM) { $env:VAE_LAMBDA_MS_SSIM } else { "1.0" }
$env:VAE_BETA_RESIDUAL = if ($env:VAE_BETA_RESIDUAL) { $env:VAE_BETA_RESIDUAL } else { "0.5" }
$env:VAE_LATENT_CHANNELS = if ($env:VAE_LATENT_CHANNELS) { $env:VAE_LATENT_CHANNELS } else { "64" }
$env:VAE_LATENT_QUANT_STEP = if ($env:VAE_LATENT_QUANT_STEP) { $env:VAE_LATENT_QUANT_STEP } else { "1.0" }
$env:VAE_BASE_CHANNELS = if ($env:VAE_BASE_CHANNELS) { $env:VAE_BASE_CHANNELS } else { "64" }
$env:VAE_RESIDUAL_CONDITION_CHANNELS = if ($env:VAE_RESIDUAL_CONDITION_CHANNELS) { $env:VAE_RESIDUAL_CONDITION_CHANNELS } else { "16" }
$env:VAE_RESIDUAL_EXTRA_BLOCKS = if ($env:VAE_RESIDUAL_EXTRA_BLOCKS) { $env:VAE_RESIDUAL_EXTRA_BLOCKS } else { "1" }
$env:VAE_MAX_Q = if ($env:VAE_MAX_Q) { $env:VAE_MAX_Q } else { "64" }
$env:VAE_NUM_WORKERS = if ($env:VAE_NUM_WORKERS) { $env:VAE_NUM_WORKERS } else { "2" }
$env:VAE_SEED = if ($env:VAE_SEED) { $env:VAE_SEED } else { "42" }
$env:VAE_CHECKPOINT = if ($env:VAE_CHECKPOINT) { $env:VAE_CHECKPOINT } else { "outputs/checkpoints/best.pth" }
$env:VAE_CONDA_ENV = if ($env:VAE_CONDA_ENV) { $env:VAE_CONDA_ENV } else { "vae_res" }
$env:VAE_LOG_INTERVAL = if ($env:VAE_LOG_INTERVAL) { $env:VAE_LOG_INTERVAL } else { "20" }
$env:VAE_SAVE_METRIC = if ($env:VAE_SAVE_METRIC) { $env:VAE_SAVE_METRIC } else { "lossy_psnr" }

New-Item -ItemType Directory -Force -Path "logs", "outputs/checkpoints" | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "logs/train_$timestamp.log"

Write-Host "Writing log to $logFile"
conda run --no-capture-output -n $env:VAE_CONDA_ENV python -u train.py 2>&1 | Tee-Object -FilePath $logFile
