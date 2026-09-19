param(
    [switch]$RemoveData,
    [string]$DataConfirmation = ''
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows

if (Test-Path -LiteralPath $script:InstallRoot) {
    Remove-Item -LiteralPath $script:InstallRoot -Recurse -Force
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
