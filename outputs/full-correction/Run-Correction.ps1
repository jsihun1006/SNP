param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$outputsRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$workspaceRoot = Split-Path -Parent $outputsRoot
$python = Join-Path $workspaceRoot 'work\source\SNP-ball-tracking\ball_only\.venv\Scripts\python.exe'
$script = Join-Path $workspaceRoot 'work\run_full_ball_correction.py'
$folder = 'H:\data\ball-tracking-results\first_half\correction_recall_v1'
$stopPath = Join-Path $folder 'STOP'
if (Test-Path -LiteralPath $stopPath) {
    if (-not $Resume) { throw 'This run is stopped. Use -Resume after it has stopped at a checkpoint.' }
    Remove-Item -LiteralPath $stopPath
}
& $python -u -B $script --out $folder
if ($LASTEXITCODE -ne 0) { throw ('Correction exited with code ' + $LASTEXITCODE) }
