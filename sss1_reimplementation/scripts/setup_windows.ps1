param(
    [ValidateSet("CoreCPU", "CUDA118")][string]$Profile = "CoreCPU",
    [switch]$InstallPython,
    [switch]$Verify
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $ProjectRoot "sss1\Scripts\python.exe"
$Constraints = Join-Path $ProjectRoot "configs\constraints-windows.txt"
if ($env:OS -ne "Windows_NT") { throw "This installer targets Windows native x64." }
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    throw "This dependency profile requires Windows x64 (Intel/AMD)."
}
$LogDirectory = Join-Path $ProjectRoot "outputs\install"
New-Item -ItemType Directory -Force $LogDirectory | Out-Null
$LogFile = Join-Path $LogDirectory ($Profile + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Start-Transcript -Path $LogFile | Out-Null

function Find-Python310 {
    $Launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($Launcher) {
        try {
            $Detected = & $Launcher.Source -3.10 -c "import sys,struct; assert struct.calcsize('P')==8; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $Detected) { return [string]($Detected | Select-Object -Last 1) }
        } catch { }
    }
    $Candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"),
        (Join-Path $env:ProgramFiles "Python310\python.exe")
    )
    $PathPython = Get-Command python -ErrorAction SilentlyContinue
    if ($PathPython -and $PathPython.Source -notmatch '\\WindowsApps\\') {
        $Candidates += $PathPython.Source
    }
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) {
            try {
                & $Candidate -c "import sys,struct; assert sys.version_info[:2]==(3,10) and struct.calcsize('P')==8" 2>$null
                if ($LASTEXITCODE -eq 0) { return $Candidate }
            } catch { }
        }
    }
    return $null
}

try {
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        $BasePython = Find-Python310
        if (-not $BasePython -and $InstallPython) {
            $Winget = Get-Command winget -ErrorAction SilentlyContinue
            if (-not $Winget) {
                throw "Install Python 3.10 x64 from https://www.python.org/downloads/release/python-31011/ then rerun. winget was not found."
            }
            & $Winget.Source install --id Python.Python.3.10 --exact --source winget --scope user --accept-source-agreements --accept-package-agreements --disable-interactivity
            if ($LASTEXITCODE -ne 0) { throw "Python installation failed. Install Python 3.10 x64 manually and rerun." }
            $BasePython = Find-Python310
        }
        if (-not $BasePython) { throw "Python 3.10 x64 was not found. Rerun with -InstallPython or install it manually." }
        & $BasePython -m venv (Join-Path $ProjectRoot "sss1")
        if ($LASTEXITCODE -ne 0) { throw "Could not create the separate sss1 environment." }
    }
    & $PythonExe -c "import sys,struct; assert sys.version_info[:2]==(3,10) and struct.calcsize('P')==8"
    if ($LASTEXITCODE -ne 0) { throw "sss1 must be a separate Python 3.10 x64 environment." }
    $env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot "outputs\.ultralytics"
    $env:MPLCONFIGDIR = Join-Path $ProjectRoot "outputs\.matplotlib"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONNOUSERSITE = "1"
    & $PythonExe -m pip install "pip==24.3.1" "setuptools==75.8.2" "wheel==0.45.1"
    if ($LASTEXITCODE -ne 0) { throw "Build tools installation failed." }
    $TorchIndex = if ($Profile -eq "CUDA118") { "https://download.pytorch.org/whl/cu118" } else { "https://download.pytorch.org/whl/cpu" }
    & $PythonExe -m pip install "torch==2.1.0" "torchvision==0.16.0" --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }
    & $PythonExe -m pip install "numpy==1.26.4" "scipy==1.13.1" -c $Constraints
    if ($LASTEXITCODE -ne 0) { throw "Numerical dependencies installation failed." }
    # chumpy's legacy setup imports pip, which is absent from isolated build environments.
    & $PythonExe -m pip install --no-build-isolation "chumpy==0.70" -c $Constraints
    if ($LASTEXITCODE -ne 0) { throw "chumpy legacy build failed." }
    & $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt") -c $Constraints
    if ($LASTEXITCODE -ne 0) { throw "Dependency resolver failed; do not upgrade torch silently." }
    if ($Profile -eq "CUDA118") {
        & $PythonExe -m pip install "mmcv==2.1.0" --only-binary=:all: -f "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html" -c $Constraints
        if ($LASTEXITCODE -ne 0) { throw "Matching MMCV binary unavailable. Do not replace it with mmcv-lite for pose." }
    } else {
        Write-Host "CoreCPU: detection, tiling, geometry and evaluation. Full pose requires the MMCV extension profile."
    }
    # Commit-pinned source archive: a ZIP checkout does not need Git installed.
    & $PythonExe -m pip install --no-deps --no-build-isolation "sskit @ https://codeload.github.com/Spiideo/sskit/zip/e64b7d89741b7bdb7e4c6d41ed6796b38516360c"
    if ($LASTEXITCODE -ne 0) { throw "Pinned SynLoc devkit installation failed." }
    & $PythonExe -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Installed package dependencies are inconsistent." }
    Push-Location (Split-Path $ProjectRoot -Parent)
    try {
        & $PythonExe -m sss1_reimplementation.scripts.probe_environment
        if ($LASTEXITCODE -ne 0) { throw "Environment probe failed." }
        if ($Verify) {
            & $PythonExe -m sss1_reimplementation.scripts.first_run --profile $Profile
            if ($LASTEXITCODE -ne 0) { throw "First run failed. See outputs\first_run and the install log." }
        }
    } finally { Pop-Location }
    Write-Host "Installation complete. Log: $LogFile"
} finally { Stop-Transcript | Out-Null }
