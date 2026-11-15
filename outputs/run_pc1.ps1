param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Join-Path $workspaceRoot 'work\source\SNP-ball-tracking'
$resultRoot = 'H:\data\ball-tracking-results\first_half'
$stopPath = Join-Path $resultRoot 'STOP'
if (Test-Path -LiteralPath $stopPath) {
    if (-not $Resume) { throw 'This job is stopped. Use -Resume to remove STOP and resume its checkpoints.' }
    Remove-Item -LiteralPath $stopPath
}
& (Join-Path $projectRoot 'ball_only\run_pc1.ps1') `
    -Video 'H:\data\broadcast\BAR-ATH_broadcast.mp4' `
    -Weights (Join-Path $projectRoot 'ball_only\weights\yolo26x.pt') `
    -Output $resultRoot `
    -FullBatch 1 -RoiBatch 4 -TileBatch 8 -ChunkFrames 32 -MaxRois 2
