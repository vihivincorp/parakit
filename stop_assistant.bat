@echo off
cd /d "%~dp0"
if exist assistant.pid (
    set /p PID=<assistant.pid
    taskkill /F /PID %PID% >nul 2>&1
    del assistant.pid >nul 2>&1
)
taskkill /F /IM pythonw.exe >nul 2>&1
taskkill /F /IM pythonw3.11.exe >nul 2>&1
echo [OK] Assistant stopped.
