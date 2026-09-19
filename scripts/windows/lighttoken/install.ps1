param(
    [switch]$SkipBuild
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$arch = Get-HostArchitecture
Write-Step "Windows detected: $([Environment]::OSVersion.Version) / $arch"
Write-Step 'Installation is user-local and does not require administrator privileges.'

$dist = Get-BuildDistRoot
if (-not $SkipBuild -or -not (Test-Path -LiteralPath (Join-Path $dist 'app\LightTokenWorkstation.exe'))) {
    & (Join-Path $PSScriptRoot 'build.ps1')
    if (-not $?) { throw 'Build failed before installation.' }
}

Ensure-Directory $script:InstallRoot
Ensure-Directory $script:DataRoot

if (Test-Path -LiteralPath $script:InstalledAppRoot) {
    Remove-Item -LiteralPath $script:InstalledAppRoot -Recurse -Force
}
if (Test-Path -LiteralPath $script:InstalledBinRoot) {
    Remove-Item -LiteralPath $script:InstalledBinRoot -Recurse -Force
}

Copy-Item -LiteralPath (Join-Path $dist 'app') -Destination $script:InstalledAppRoot -Recurse -Force
Copy-Item -LiteralPath (Join-Path $dist 'bin') -Destination $script:InstalledBinRoot -Recurse -Force
Write-InstallManifest $arch

$launcher = Get-InstalledLauncher
if (-not (Test-Path -LiteralPath $launcher)) { throw "Installed launcher missing: $launcher" }
if (-not (Test-Path -LiteralPath (Join-Path (Get-InstalledNativeDir) 'lighttoken_ffi.dll')) {
    throw 'Installed JNI DLL is missing.'
}

Write-Step "Installed workstation: $launcher"
Write-Step "SQLite/application data root: $script:DataRoot"
Write-Step 'INSTALL PASS'
