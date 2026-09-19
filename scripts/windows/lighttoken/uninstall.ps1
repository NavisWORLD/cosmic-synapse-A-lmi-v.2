param(
    [switch]$RemoveData,
    [string]$DataConfirmation = ''
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows

# Only files and directories that the installer owns may be removed. Leave
# unknown siblings in the installation root undisturbed.
foreach ($owned in @($script:InstalledAppRoot, $script:InstalledBinRoot)) {
    if (Test-Path -LiteralPath $owned) {
        $item = Get-Item -LiteralPath $owned -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            Remove-Item -LiteralPath $owned -Force
        }
        else {
            Remove-Item -LiteralPath $owned -Recurse -Force
        }
    }
}
if (Test-Path -LiteralPath $script:InstallManifest) {
    Remove-Item -LiteralPath $script:InstallManifest -Force
}
if (Test-Path -LiteralPath $script:InstallRoot) {
    $remaining = @(Get-ChildItem -LiteralPath $script:InstallRoot -Force)
    if ($remaining.Count -eq 0) {
        Remove-Item -LiteralPath $script:InstallRoot -Force
    }
}

if ($RemoveData -and (Test-Path -LiteralPath $script:DataRoot)) {
    $required = 'DELETE LIGHTTOKEN DATA'
    if ($DataConfirmation -ne $required) {
        $DataConfirmation = Read-Host "LightToken user data is separate from program files. Type $required to delete '$script:DataRoot'"
    }
    if ($DataConfirmation -ne $required) {
        throw 'User-data deletion cancelled. Program files are removed; LightToken data remains intact.'
    }
    Remove-Item -LiteralPath $script:DataRoot -Recurse -Force
}

Write-Step 'UNINSTALL PASS'
if (Test-Path -LiteralPath $script:DataRoot) {
    Write-Step "Preserved user data: $script:DataRoot"
}
