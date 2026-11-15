# Run in your own interactive PowerShell window, not an unattended session.
# This uploads existing commits and Git LFS objects. It does not commit or merge.
[CmdletBinding()]
param(
    [string]$RepositoryPath = (Split-Path -Parent $PSScriptRoot),
    [string]$RepositoryUrl = 'https://github.com/jsihun1006/SNP.git',
    [string]$RemoteName = 'origin',
    [string]$UploadBranch = 'codex/ball-tracking-correction-20261115',
    [string]$GitHubUsername = '',
    [string]$GitExecutable = '',
    [switch]$SkipLogin,
    [switch]$TestOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($GitExecutable)) {
    $gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($gitCommand) { $GitExecutable = $gitCommand.Source }
}
if ([string]::IsNullOrWhiteSpace($GitExecutable)) {
    $candidatePaths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd\git.exe'),
        'C:\Program Files\Git\cmd\git.exe',
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe')
    )
    foreach ($candidatePath in $candidatePaths) {
        if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
            $GitExecutable = $candidatePath
            break
        }
    }
}
if (-not (Test-Path -LiteralPath $GitExecutable -PathType Leaf)) {
    throw 'Git was not found. Install Git for Windows or supply -GitExecutable PATH_TO_GIT.EXE.'
}
if (-not (Test-Path -LiteralPath $RepositoryPath -PathType Container)) {
    throw 'RepositoryPath does not exist.'
}
if ($RemoteName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') {
    throw 'RemoteName must be a simple Git remote name.'
}
if ($RepositoryUrl -notmatch '^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$') {
    throw 'RepositoryUrl must be an HTTPS GitHub repository URL, without embedded credentials.'
}

# OpenSSL avoids the Schannel credential error observed on this PC.
# These options are per command; global Git configuration is not changed.
$gitOptions = @('-c', 'http.sslBackend=openssl')
if (-not [string]::IsNullOrWhiteSpace($GitHubUsername)) {
    $gitOptions += @('-c', ('credential.username=' + $GitHubUsername))
}

function Invoke-TaskGit {
    param([Parameter(Mandatory = $true)][string[]]$GitArguments)
    & $GitExecutable @gitOptions @GitArguments
    if ($LASTEXITCODE -ne 0) {
        throw ('Git command failed (exit ' + $LASTEXITCODE + '): git ' + ($GitArguments -join ' '))
    }
}

Push-Location -LiteralPath $RepositoryPath
try {
    Invoke-TaskGit -GitArguments @('rev-parse', '--show-toplevel')
    Invoke-TaskGit -GitArguments @('rev-parse', '--verify', 'HEAD')
    Invoke-TaskGit -GitArguments @('check-ref-format', '--branch', $UploadBranch)
    Invoke-TaskGit -GitArguments @('lfs', 'version')
    Invoke-TaskGit -GitArguments @('lfs', 'fsck')

    $pendingChanges = @(& $GitExecutable status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read the Git working tree.' }
    if ($pendingChanges.Count -ne 0) {
        throw 'The working tree has uncommitted changes. Review and commit them first; this script only uploads existing commits.'
    }

    $remoteNames = @(& $GitExecutable remote)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read Git remotes.' }
    $remoteExists = $remoteNames -contains $RemoteName
    if ($remoteExists) {
        $configuredUrl = & $GitExecutable remote get-url $RemoteName
        if ($LASTEXITCODE -ne 0) { throw 'Cannot read the remote fetch URL.' }
        if (([string]$configuredUrl).TrimEnd('/') -ne $RepositoryUrl.TrimEnd('/')) {
            throw 'The existing remote URL differs from RepositoryUrl. Use a different -RemoteName; this script will not replace an existing remote.'
        }
        $pushUrls = @(& $GitExecutable remote get-url --push --all $RemoteName)
        if ($LASTEXITCODE -ne 0) { throw 'Cannot read the remote push URLs.' }
        if ($pushUrls.Count -ne 1 -or ([string]$pushUrls[0]).TrimEnd('/') -ne $RepositoryUrl.TrimEnd('/')) {
            throw 'The remote push destination differs from RepositoryUrl or has multiple targets. Review remote.pushurl before uploading.'
        }
    }

    Write-Host ('Repository: ' + (Get-Location).Path)
    Write-Host ('Upload target: ' + $RepositoryUrl + ' -> ' + $UploadBranch)
    Write-Host 'The existing ball-tracking branch will not be overwritten.'
    Write-Host 'This is a ZIP-based snapshot with separate history from upstream.'

    if ($TestOnly) {
        Write-Host 'Local checks passed. TestOnly: no login, network access, or upload was performed.'
        return
    }

    if (-not $remoteExists) {
        Invoke-TaskGit -GitArguments @('remote', 'add', $RemoteName, $RepositoryUrl)
    }

    if (-not $SkipLogin) {
        Write-Host 'Sign in with a GitHub account that has WRITE access to the SNP repository.'
        Write-Host 'Follow the device-code instructions below. Never put a token in the repository URL.'
        $loginArguments = @('credential-manager', 'github', 'login', '--device', '--force')
        if (-not [string]::IsNullOrWhiteSpace($GitHubUsername)) {
            $loginArguments += @('--username', $GitHubUsername)
        }
        Invoke-TaskGit -GitArguments $loginArguments
    }

    # Check access and the exact target branch before sending LFS objects.
    & $GitExecutable @gitOptions ls-remote --exit-code --heads $RemoteName ('refs/heads/' + $UploadBranch)
    $remoteCheckExit = $LASTEXITCODE
    if ($remoteCheckExit -ne 0 -and $remoteCheckExit -ne 2) {
        throw 'Repository access failed. Check the URL and sign in with an account granted access; login alone does not grant repository permissions.'
    }
    if ($remoteCheckExit -eq 0) {
        $trackingRef = 'refs/remotes/' + $RemoteName + '/' + $UploadBranch
        $fetchSpec = 'refs/heads/' + $UploadBranch + ':' + $trackingRef
        Invoke-TaskGit -GitArguments @('fetch', '--no-tags', $RemoteName, $fetchSpec)
        & $GitExecutable merge-base --is-ancestor $trackingRef HEAD
        if ($LASTEXITCODE -ne 0) {
            throw 'The target branch has history not contained in this snapshot. Choose a new -UploadBranch. No force push will be performed.'
        }
    }

    Invoke-TaskGit -GitArguments @('lfs', 'install', '--local')
    Invoke-TaskGit -GitArguments @('lfs', 'push', '--all', $RemoteName, 'HEAD')
    Invoke-TaskGit -GitArguments @('push', '--set-upstream', $RemoteName, ('HEAD:refs/heads/' + $UploadBranch))

    $localHead = (& $GitExecutable rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot verify local HEAD after upload.' }
    $remoteHeadLine = & $GitExecutable @gitOptions ls-remote --heads $RemoteName ('refs/heads/' + $UploadBranch)
    if ($LASTEXITCODE -ne 0 -or -not $remoteHeadLine) { throw 'Upload finished but remote verification failed.' }
    $remoteHead = (([string]$remoteHeadLine) -split '\s+')[0]
    if ($remoteHead -ne $localHead) { throw 'Remote HEAD does not match the local commit. Check the GitHub branch before retrying.' }

    $webUrl = $RepositoryUrl -replace '\.git$', ''
    Write-Host ''
    Write-Host ('Upload verified: ' + $localHead)
    Write-Host ($webUrl + '/tree/' + $UploadBranch)
    Write-Host 'Git commits and their reachable Git LFS objects were uploaded. No merge was performed.'
}
finally {
    Pop-Location
}
