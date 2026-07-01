@echo off
cd /d "D:\coding\pothole"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_camera.ps1"
pause
