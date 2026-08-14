@echo off
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\scripts\stop_all.ps1"
if errorlevel 1 pause
