@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process pythonw -ArgumentList 'main.py' -WorkingDirectory '%~dp0'"
echo Multimodal AI Assistant launched detached in background.
echo You can close this window or Antigravity anytime!
