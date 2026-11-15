$ErrorActionPreference = 'Stop'
$progressPath = 'H:\data\ball-tracking-results\first_half\progress.json'
if (-not (Test-Path -LiteralPath $progressPath -PathType Leaf)) { Write-Output 'Waiting for the first progress record.'; exit 0 }
$progress = Get-Content -LiteralPath $progressPath -Raw -Encoding UTF8 | ConvertFrom-Json
[pscustomobject]@{
    Status = $progress.status
    CompletedFrames = $progress.completed_frames
    TotalFrames = $progress.total_frames
    Percent = [math]::Round(100 * $progress.completed_frames / $progress.total_frames, 3)
    ProcessingFps = $progress.processing_fps
    RemainingHours = $(if ($null -ne $progress.eta_seconds) { [math]::Round($progress.eta_seconds / 3600, 2) } else { $null })
    UpdatedAt = [DateTimeOffset]::FromUnixTimeSeconds([long]$progress.updated_unix).ToOffset([TimeSpan]::FromHours(9)).ToString('yyyy-MM-dd HH:mm:ss zzz')
} | Format-List
