param(
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$packageRoot = $PSScriptRoot
$temporaryRoot = Join-Path $packageRoot '.tmp'
$cacheRoot = Join-Path $packageRoot '.cache'
$pipCache = Join-Path $cacheRoot 'pip'
foreach ($directory in @($temporaryRoot, $pipCache, (Join-Path $cacheRoot 'torch'), (Join-Path $cacheRoot 'ultralytics'))) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$env:PIP_CACHE_DIR = $pipCache
$env:TORCH_HOME = Join-Path $cacheRoot 'torch'
$env:YOLO_CONFIG_DIR = Join-Path $cacheRoot 'ultralytics'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ('Command failed with exit code ' + $LASTEXITCODE + ': ' + $Executable)
    }
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $foundPython = & $launcher.Source -3.11 -I -B -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $foundPython) {
                $PythonExe = [string]($foundPython | Select-Object -Last 1)
            }
        } catch { $PythonExe = '' }
    }
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notlike '*\WindowsApps\*') {
        $PythonExe = $pythonCommand.Source
    }
}
if ([string]::IsNullOrWhiteSpace($PythonExe) -or -not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw 'Python 3.11 was not found. Install Python 3.11 on the intended drive, then rerun with -PythonExe PATH_TO_PYTHON.EXE. No Python installation on C: is performed by this script.'
}
$version = & $PythonExe -I -B -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))"
if ($LASTEXITCODE -ne 0 -or ([string]$version).Trim() -ne '3.11') {
    throw 'This package is tested with Python 3.11. Supply its full path with -PythonExe.'
}
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw 'nvidia-smi was not found. Install/update the NVIDIA driver and reopen PowerShell before running setup.'
}

$venvRoot = Join-Path $packageRoot '.venv'
$venvPython = Join-Path $venvRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Invoke-Checked -Executable $PythonExe -Arguments @('-I', '-B', '-m', 'venv', $venvRoot)
}
Invoke-Checked -Executable $venvPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip', '--cache-dir', $pipCache)
Invoke-Checked -Executable $venvPython -Arguments @('-m', 'pip', 'install', '--cache-dir', $pipCache, '--index-url', 'https://download.pytorch.org/whl/cu128', 'torch==2.9.1', 'torchvision==0.24.1')
Invoke-Checked -Executable $venvPython -Arguments @('-m', 'pip', 'install', '--cache-dir', $pipCache, '-r', (Join-Path $packageRoot 'requirements.txt'))
Invoke-Checked -Executable $venvPython -Arguments @('-I', '-B', '-c', "import torch, torchvision, cv2, ultralytics, numpy; assert torch.cuda.is_available(), 'CUDA is unavailable; check the NVIDIA driver'; assert torch.ones(1, device='cuda').sum().item() == 1; print('CUDA ready:', torch.cuda.get_device_name(0)); print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__)")
Write-Output ('Setup complete. Python: ' + $venvPython)
Write-Output 'Copy yolo26x.pt and the full broadcast video separately; do not commit them to Git.'
