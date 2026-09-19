param(
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$git = Require-Command 'git' 'Git is required to update the source checkout.'

$status = & $git -C $script:RepoRoot status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect repository status.' }
if ($status) {
    throw 'Repository has local changes. Update aborted; source and user data were not modified.'
}

$branch = (& $git -C $script:RepoRoot rev-parse --abbrev-ref HEAD | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $branch -or $branch -eq 'HEAD') {
    throw 'Update requires a named local branch; detached HEAD updates are rejected.'
}

Write-Step "Fetching branch $branch"
Invoke-External $git @('-C', $script:RepoRoot, 'fetch', 'origin', $branch)
Invoke-External $git @('-C', $script:RepoRoot, 'merge', '--ff-only', "origin/$branch")

& (Join-Path $PSScriptRoot 'install.ps1') -SkipBuild:$SkipBuild
if (-not $?) { throw 'Updated source could not be installed.' }

Write-Step 'UPDATE PASS. Existing LightToken data was preserved.'
