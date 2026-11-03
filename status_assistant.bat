@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { if (Test-Path assistant.pid) { $p = Get-Content assistant.pid; $proc = Get-Process -Id $p -ErrorAction SilentlyContinue; if ($proc) { Write-Host '============================================'; Write-Host ('[RUNNING] Assistant is ACTIVE (PID: ' + $p + ')'); Write-Host 'Mode: Background Detached (safe to close terminal/IDE)'; Write-Host '============================================' } else { Write-Host '[STOPPED] Assistant is not running.' } } else { Write-Host '[STOPPED] Assistant is not running.' } }"
