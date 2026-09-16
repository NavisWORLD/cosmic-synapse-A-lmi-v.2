param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$binary = Get-InstalledBinary
if (-not (Test-Path -LiteralPath $binary)) {
    $binary = Get-ReleaseBinary
}
if (-not (Test-Path -LiteralPath $binary)) {
    throw 'A-LMI native executable was not found. Run INSTALL_WINDOWS.bat or BUILD_WINDOWS.bat first.'
}

if ($null -eq $Arguments -or $Arguments.Count -eq 0) {
    Write-Host 'A-LMI Native Core'
    Write-Host '  RUN_WINDOWS.bat doctor'
    Write-Host '  RUN_WINDOWS.bat version'
    Write-Host '  RUN_WINDOWS.bat init <workspace> --name <name> --seed <n>'
    Write-Host '  RUN_WINDOWS.bat inspect <workspace>'
    Write-Host '  RUN_WINDOWS.bat export <workspace> <bundle.cosmos>'
    Write-Host '  RUN_WINDOWS.bat verify <bundle.cosmos>'
    Write-Host '  RUN_WINDOWS.bat import <bundle.cosmos> <workspace>'
    exit 0
}

& $binary @Arguments
exit $LASTEXITCODE
