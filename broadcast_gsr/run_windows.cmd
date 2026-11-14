@echo off
setlocal EnableExtensions
set "GSR_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%GSR_PY%" (
  echo Run broadcast_gsr\setup_windows.cmd first.
  exit /b 1
)
pushd "%~dp0.."
"%GSR_PY%" -m broadcast_gsr %*
set "GSR_EXIT=%ERRORLEVEL%"
popd
exit /b %GSR_EXIT%
