@echo off
setlocal
for %%I in ("%~dp0..") do set "MTAH_V4_PROJECT=%%~fI"
if defined MTAH_V4_PYTHON (
  set "MTAH_V4_EXEC=%MTAH_V4_PYTHON%"
) else (
  set "MTAH_V4_EXEC=%MTAH_V4_PROJECT%\..\.venv\Scripts\python.exe"
)
if not exist "%MTAH_V4_EXEC%" (
  echo ERROR: Python not found: "%MTAH_V4_EXEC%"
  echo Set MTAH_V4_PYTHON to the existing Python 3.11 venv executable.
  exit /b 2
)
pushd "%MTAH_V4_PROJECT%"
if errorlevel 1 exit /b 2
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
"%MTAH_V4_EXEC%" "%MTAH_V4_PROJECT%\starter\windows_tryolabs_v4.py" %*
set "MTAH_V4_EXIT=%ERRORLEVEL%"
popd
exit /b %MTAH_V4_EXIT%
