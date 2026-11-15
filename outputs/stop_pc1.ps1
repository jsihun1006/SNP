$ErrorActionPreference = 'Stop'
$resultRoot = 'H:\data\ball-tracking-results\first_half'
if (-not (Test-Path -LiteralPath $resultRoot -PathType Container)) { throw 'The first-half output folder has not been created yet.' }
New-Item -ItemType File -Path (Join-Path $resultRoot 'STOP') -Force | Out-Null
Write-Output 'Stop requested. The job will stop after its current chunk is committed.'
