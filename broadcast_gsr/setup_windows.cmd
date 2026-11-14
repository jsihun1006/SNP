@echo off
setlocal EnableExtensions
set "GSR_DIR=%~dp0"
set "GSR_PROFILE=%~1"
if "%GSR_PROFILE%"=="" set "GSR_PROFILE=cpu"
if /I "%GSR_PROFILE%"=="cpu" goto profile_ok
if /I "%GSR_PROFILE%"=="cu128" goto profile_ok
echo Usage: setup_windows.cmd [cpu^|cu128]
exit /b 2
:profile_ok
where git >nul 2>nul
if errorlevel 1 goto missing_git
if exist "%GSR_DIR%.venv\Scripts\python.exe" goto install
py -3.11 -m venv "%GSR_DIR%.venv"
if errorlevel 1 goto missing_python
:install
set "GSR_PY=%GSR_DIR%.venv\Scripts\python.exe"
"%GSR_PY%" -m pip install --upgrade pip "setuptools==75.8.0" "wheel==0.45.1"
if errorlevel 1 goto failed
"%GSR_PY%" -m pip install "torch==2.9.1" "torchvision==0.24.1" --index-url "https://download.pytorch.org/whl/%GSR_PROFILE%"
if errorlevel 1 goto failed
"%GSR_PY%" -m pip install -r "%GSR_DIR%requirements-models.txt"
if errorlevel 1 goto failed
pushd "%GSR_DIR%.."
"%GSR_PY%" -m broadcast_gsr bootstrap
if errorlevel 1 goto failed_pop
if not exist "%GSR_DIR%weights" mkdir "%GSR_DIR%weights"
echo Setup finished. Place official weights in broadcast_gsr\weights.
echo Next: broadcast_gsr\run_windows.cmd doctor
echo Synthetic check: broadcast_gsr\run_windows.cmd smoke --out data\derived\broadcast_gsr\smoke_001
popd
exit /b 0
:missing_git
echo Install Git for Windows and retry.
exit /b 1
:missing_python
echo Install 64-bit Python 3.11 with the Python launcher, then retry.
exit /b 1
:failed_pop
popd
:failed
echo Setup failed. Existing project environments were not modified.
exit /b 1
