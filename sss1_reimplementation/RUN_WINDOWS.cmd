@echo off
if "%~1"=="" (
    echo Usage: RUN_WINDOWS.cmd "C:\videos\match.mp4"
    pause
    exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_windows.ps1" -InputPath %*
exit /b %ERRORLEVEL%
