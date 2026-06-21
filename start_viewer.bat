@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:8000
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "python viewer_server.py --host 127.0.0.1 --port 8000"
pause
