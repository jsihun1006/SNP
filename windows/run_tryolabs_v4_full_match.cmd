@echo off
setlocal
for %%I in ("%~dp0..") do set "MTAH_V4_PROJECT=%%~fI"
if defined MTAH_V4_PYTHON (
  set "MTAH_V4_EXEC=%MTAH_V4_PYTHON%"
) else (
  set "MTAH_V4_EXEC=%MTAH_V4_PROJECT%\..\.venv\Scripts\python.exe"
)
if not exist "%MTAH_V4_EXEC%" (
  echo ERROR: Existing Python 3.11 venv not found: "%MTAH_V4_EXEC%"
  exit /b 2
)
pushd "%MTAH_V4_PROJECT%"
if errorlevel 1 exit /b 2
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=2"
"%MTAH_V4_EXEC%" "%MTAH_V4_PROJECT%\starter\run_tryolabs_v4_streaming.py" %*
set "MTAH_V4_EXIT=%ERRORLEVEL%"
popd
exit /b %MTAH_V4_EXIT%
