$ErrorActionPreference = 'Stop'
$folder = 'H:\data\ball-tracking-results\first_half\correction_recall_v1'
$progressFile = Join-Path $folder 'progress.json'
if (-not (Test-Path -LiteralPath $progressFile)) { Write-Output 'Waiting for the first progress record.'; return }
$p = Get-Content -LiteralPath $progressFile -Raw -Encoding UTF8 | ConvertFrom-Json
[pscustomobject]@{
    Status = $p.status
    Phase = $p.phase
    CompletedFrames = $p.completed_frames
    TotalFrames = $p.total_frames
    Percent = [math]::Round(100 * $p.completed_frames / $p.total_frames, 2)
    ProcessingFps = $p.processing_fps
    RemainingMinutes = $(if ($null -ne $p.eta_seconds) { [math]::Round($p.eta_seconds / 60, 1) } else { $null })
    UpdatedAt = [DateTimeOffset]::FromUnixTimeSeconds([long]$p.updated_unix).ToOffset([TimeSpan]::FromHours(9)).ToString('yyyy-MM-dd HH:mm:ss zzz')
} | Format-List
if (Test-Path -LiteralPath (Join-Path $folder 'recovery_progress.json')) {
    $r = Get-Content -LiteralPath (Join-Path $folder 'recovery_progress.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    [pscustomobject]@{RecoveryJobsDone=$r.processed_jobs;RecoveryJobsPlanned=$r.planned_jobs;NewCandidateObservations=$r.new_candidates} | Format-List
}
