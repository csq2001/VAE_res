@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE
python export_lossy_residual.py
pause
