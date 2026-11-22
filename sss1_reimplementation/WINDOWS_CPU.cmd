@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1" -Profile CoreCPU -InstallPython -Verify
set "SSS_EXIT=%ERRORLEVEL%"
if not "%SSS_EXIT%"=="0" echo Installation failed. See outputs\install for the log.
pause
exit /b %SSS_EXIT%
