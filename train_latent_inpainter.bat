@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE
python -u train_latent_inpainter.py --epochs 10 --log-interval 20
pause
