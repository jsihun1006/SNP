param(
    [string]$InputPath,
    [string]$OutputPath,
    [ValidateSet("full", "sahi", "baat")][string]$Mode = "baat",
    [string]$Device = "cpu",
    [string]$PoseCheckpoint,
    [string]$Camera
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $ProjectRoot "sss1\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { throw "Run WINDOWS_CPU.cmd or WINDOWS_CUDA.cmd first." }
if (-not $InputPath) { throw 'Supply an image or video: .\RUN_WINDOWS.cmd "C:\videos\match.mp4"' }
$InputPath = (Resolve-Path -LiteralPath $InputPath).Path
if (-not $OutputPath) {
    $OutputPath = Join-Path $ProjectRoot ("outputs\run_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
if ($PoseCheckpoint) { $PoseCheckpoint = (Resolve-Path -LiteralPath $PoseCheckpoint).Path }
if ($Camera) { $Camera = (Resolve-Path -LiteralPath $Camera).Path }
$Arguments = @("-m", "sss1_reimplementation.scripts.infer", "--input", $InputPath,
    "--detector", (Join-Path $ProjectRoot "checkpoints\yolo26l.pt"),
    "--mode", $Mode, "--device", $Device, "--output", $OutputPath)
if ($PoseCheckpoint) { $Arguments += @("--pose-checkpoint", $PoseCheckpoint) }
if ($Camera) { $Arguments += @("--camera", $Camera) }
Push-Location (Split-Path $ProjectRoot -Parent)
try {
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Inference failed. See the error above." }
    Write-Host "Results: $OutputPath"
} finally { Pop-Location }
