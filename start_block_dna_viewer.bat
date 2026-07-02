@echo off
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE
start "" http://127.0.0.1:8003
python block_dna_server.py --host 127.0.0.1 --port 8003
pause
