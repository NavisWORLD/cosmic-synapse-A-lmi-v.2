param(
    [switch]$RemoveData,
    [switch]$Force
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows

$pythonBindingsInstalled = $false
if (Test-Path -LiteralPath $script:InstallManifest) {
    try {
        $manifest = Get-Content -LiteralPath $script:InstallManifest -Raw | ConvertFrom-Json
        $pythonBindingsInstalled = [bool]$manifest.python_bindings_installed
    }
    catch {
        Write-Step 'Install manifest could not be parsed; continuing with program-file removal only.'
    }
}

if ($pythonBindingsInstalled) {
    $python = Get-CommandPath 'python'
    if ($python) {
        Write-Step 'Removing the Python binding package installed by A-LMI.'
        & $python -m pip uninstall -y almi-native
        if ($LASTEXITCODE -ne 0) {
            Write-Step 'Python binding uninstall returned a non-zero status; native program removal will continue.'
        }
    }
    else {
        Write-Step 'Python is unavailable, so the Python binding package could not be removed automatically.'
    }
}

if (Test-Path -LiteralPath $script:BinDir) {
    Remove-Item -LiteralPath $script:BinDir -Recurse -Force
}
if (Test-Path -LiteralPath $script:InstallManifest) {
    Remove-Item -LiteralPath $script:InstallManifest -Force
}

$dataDir = Join-Path $script:InstallRoot 'data'
if ($RemoveData -and (Test-Path -LiteralPath $dataDir)) {
    if (-not $Force) {
        $answer = Read-Host "Remove A-LMI application data at '$dataDir'? Type DELETE to confirm"
        if ($answer -ne 'DELETE') {
            throw 'Data removal cancelled. Program files may already have been removed; user data remains intact.'
        }
    }
    Remove-Item -LiteralPath $dataDir -Recurse -Force
}

if (Test-Path -LiteralPath $script:InstallRoot) {
    $remaining = Get-ChildItem -LiteralPath $script:InstallRoot -Force -ErrorAction SilentlyContinue
    if (-not $remaining) { Remove-Item -LiteralPath $script:InstallRoot -Force }
}

Write-Step 'UNINSTALL PASS'
Write-Host 'Removed A-LMI program files owned by the installer.'
Write-Host 'User workspaces, .cosmos bundles, memory ledgers, and configuration backups outside the install-owned data directory were not searched for or deleted.'
