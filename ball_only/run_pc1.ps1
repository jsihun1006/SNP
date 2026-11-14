param(
    [Parameter(Mandatory = $true)][string]$Video,
    [Parameter(Mandatory = $true)][string]$Weights,
    [string]$Start = '',
    [string]$End = '',
    [string]$RangesFile = '',
    [string]$Output = '',
    [ValidateRange(1, 4096)][int]$ChunkFrames = 32,
    [ValidateRange(1, 256)][int]$FullBatch = 1,
    [ValidateRange(1, 256)][int]$RoiBatch = 4,
    [ValidateRange(1, 256)][int]$TileBatch = 8,
    [ValidateRange(0, 4096)][int]$MaxRois = 2
)

$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $packageRoot
$hasStart = -not [string]::IsNullOrWhiteSpace($Start)
$hasEnd = -not [string]::IsNullOrWhiteSpace($End)
$hasRanges = -not [string]::IsNullOrWhiteSpace($RangesFile)
if ($hasStart -xor $hasEnd) { throw 'Supply both -Start and -End, or neither.' }
if ($hasStart -and $hasRanges) { throw 'Choose explicit -Start/-End or -RangesFile, not both.' }
$expectedVideoSha = ''
if (-not $hasStart) {
    if (-not $hasRanges) { $RangesFile = Join-Path $packageRoot 'jobs.bar_ath.json' }
    if (-not (Test-Path -LiteralPath $RangesFile -PathType Leaf)) {
        throw 'The range file does not exist. Supply -RangesFile or explicit -Start and -End.'
    }
    $rangeConfig = Get-Content -LiteralPath $RangesFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $selectedJobs = @($rangeConfig.jobs | Where-Object { $_.part -eq 'first_half' })
    if ($selectedJobs.Count -ne 1) { throw 'The range file must contain exactly one first_half job.' }
    $Start = [string]$selectedJobs[0].start
    $End = [string]$selectedJobs[0].end
    if ([string]::IsNullOrWhiteSpace($Start) -or [string]::IsNullOrWhiteSpace($End)) {
        throw 'Fill in the first_half start/end values before using this range file.'
    }
    $expectedVideoSha = [string]$rangeConfig.source_video_sha256
    if (-not [string]::IsNullOrWhiteSpace($expectedVideoSha) -and $expectedVideoSha -notmatch '^[a-fA-F0-9]{64}$') {
        throw 'source_video_sha256 must be a 64-character SHA256 value.'
    }
}
$python = Join-Path $packageRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Run ball_only\setup_windows.ps1 first.'
}
$videoPath = (Resolve-Path -LiteralPath $Video).Path
$weightsPath = (Resolve-Path -LiteralPath $Weights).Path
if ([string]::IsNullOrWhiteSpace($Output)) { $Output = Join-Path $packageRoot 'outputs\first_half' }
foreach ($directory in @((Join-Path $packageRoot '.tmp'), (Join-Path $packageRoot '.cache\ultralytics'), (Join-Path $packageRoot '.cache\torch'))) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$env:TEMP = Join-Path $packageRoot '.tmp'
$env:TMP = $env:TEMP
$env:YOLO_CONFIG_DIR = Join-Path $packageRoot '.cache\ultralytics'
$env:TORCH_HOME = Join-Path $packageRoot '.cache\torch'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Push-Location -LiteralPath $projectRoot
try {
    $runArguments = @('-u', '-B', '-m', 'ball_only.run', '--video', $videoPath, '--weights', $weightsPath, '--out', $Output,
        '--start', $Start, '--end', $End, '--part', 'first_half', '--chunk-frames', $ChunkFrames,
        '--full-batch', $FullBatch, '--roi-batch', $RoiBatch, '--tile-batch', $TileBatch, '--max-rois', $MaxRois,
        '--gpu-target-low', 98, '--gpu-target-high', 99.5)
    if (-not [string]::IsNullOrWhiteSpace($expectedVideoSha)) { $runArguments += @('--expected-video-sha256', $expectedVideoSha) }
    & $python @runArguments
    if ($LASTEXITCODE -ne 0) { throw ('First-half job exited with code ' + $LASTEXITCODE) }
} finally { Pop-Location }
