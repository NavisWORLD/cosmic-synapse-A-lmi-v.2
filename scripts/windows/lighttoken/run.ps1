param(
    [switch]$Check,
    [switch]$Smoke,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$launcher = Get-InstalledLauncher
if (-not (Test-Path -LiteralPath $launcher)) {
    throw 'LightToken workstation is not installed. Run INSTALL_LIGHTTOKEN_WINDOWS.bat first.'
}
$nativeDir = Get-InstalledNativeDir
if (-not (Test-Path -LiteralPath (Join-Path $nativeDir 'lighttoken_ffi.dll')) {
    throw 'Installed LightToken JNI DLL is missing.'
}

$env:LIGHTTOKEN_DATA_ROOT = $script:DataRoot
$cpp = Get-InstalledCppLibrary
if (Test-Path -LiteralPath $cpp) {
    $env:LIGHTTOKEN_CPP_LIB = $cpp
}
else {
    Remove-Item Env:LIGHTTOKEN_CPP_LIB -ErrorAction SilentlyContinue
}

if ($Check) {
    Write-Step "Installed launcher ready: $launcher"
    Write-Step "Data root: $script:DataRoot"
    exit 0
}

if ($Smoke) {
    $process = Start-Process -FilePath $launcher -ArgumentList '--packaged-smoke' -PassThru -Wait
    if ($process.ExitCode -ne 0) { throw "Packaged workstation smoke failed with exit code $($process.ExitCode)." }
    Write-Step 'RUN packaged smoke PASS'
    exit 0
}

Start-Process -FilePath $launcher -ArgumentList $Arguments | Out-Null
Write-Step 'LightToken workstation launched.'
