$ErrorActionPreference = 'Stop'
$folder = 'H:\data\ball-tracking-results\first_half\correction_recall_v1'
if (-not (Test-Path -LiteralPath $folder -PathType Container)) { throw 'The correction output folder does not exist yet.' }
New-Item -ItemType File -Path (Join-Path $folder 'STOP') -Force | Out-Null
Write-Output 'Stop requested. The current chunk will finish and save its checkpoint.'
