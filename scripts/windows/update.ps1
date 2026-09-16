param(
    [switch]$WithPython,
    [switch]$SkipPython
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$git = Require-Command 'git' 'Git is required to update from repository source.'

$status = & $git -C $script:RepoRoot status --porcelain
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect repository status.' }
if ($status) {
    throw 'Repository has local changes. Update aborted to avoid overwriting source work.'
}

$branch = (& $git -C $script:RepoRoot rev-parse --abbrev-ref HEAD | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $branch) { throw 'Unable to determine current Git branch.' }
Write-Step "Updating source branch: $branch"
Invoke-External $git @('-C', $script:RepoRoot, 'fetch', 'origin', $branch)
Invoke-External $git @('-C', $script:RepoRoot, 'merge', '--ff-only', "origin/$branch")

$install = Join-Path $PSScriptRoot 'install.ps1'
if ($WithPython) { & $install -WithPython }
elseif ($SkipPython) { & $install -SkipPython }
else { & $install }
if (-not $?) { throw 'Update installation failed.' }

Write-Step 'UPDATE PASS. Existing user workspaces and .cosmos bundles were not deleted or modified.'
