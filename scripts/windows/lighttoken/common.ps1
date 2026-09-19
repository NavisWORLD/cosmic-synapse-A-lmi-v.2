Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$script:RustRoot = Join-Path $script:RepoRoot 'native\lighttoken-rs'
$script:CppRoot = Join-Path $script:RepoRoot 'native\lighttoken-cpp'
$script:JavaRoot = Join-Path $script:RepoRoot 'apps\lighttoken-workstation-java'
$script:DistRoot = Join-Path $script:RepoRoot 'dist\lighttoken'
$script:InstallRoot = if ($env:LIGHTTOKEN_INSTALL_ROOT) { $env:LIGHTTOKEN_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA 'Programs\A-LMI\LightToken' }
$script:DataRoot = if ($env:LIGHTTOKEN_DATA_ROOT) { $env:LIGHTTOKEN_DATA_ROOT } else { Join-Path $env:LOCALAPPDATA 'A-LMI\LightToken' }
$script:InstallManifest = Join-Path $script:InstallRoot 'install.json'
$script:InstalledAppRoot = Join-Path $script:InstallRoot 'app'
$script:InstalledBinRoot = Join-Path $script:InstallRoot 'bin'

function Write-Step([string]$Message) {
    Write-Host "[LightToken] $Message"
}

function Assert-Windows {
    if ($env:OS -ne 'Windows_NT') {
        throw 'This script is intended for Windows.'
    }
    $version = [Environment]::OSVersion.Version
    if ($version.Major -lt 10) {
        throw "Unsupported Windows version: $version. Windows 10 or newer is required."
    }
}

function Get-HostArchitecture {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    if ($arch -notin @('X64', 'Arm64')) {
        throw "Unsupported Windows architecture: $arch"
    }
    return $arch
}

function Get-CommandPath([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) { return $null }
    return $command.Source
}

function Require-Command([string]$Name, [string]$Guidance) {
    $path = Get-CommandPath $Name
    if (-not $path) { throw "$Name was not found. $Guidance" }
    return $path
}

function Invoke-External([string]$FilePath, [string[]]$Arguments) {
    Write-Step ("Running: {0} {1}" -f $FilePath, ($Arguments -join ' '))
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE: $FilePath"
    }
}

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Get-ArchSlug {
    $arch = Get-HostArchitecture
    if ($arch -eq 'X64') { return 'x64' }
    return 'arm64'
}

function Get-CMakeArchitecture {
    $arch = Get-HostArchitecture
    if ($arch -eq 'X64') { return 'x64' }
    return 'ARM64'
}

function Get-BuildDistRoot {
    return (Join-Path $script:DistRoot ("windows-" + (Get-ArchSlug)))
}

function Get-RustReleaseDir {
    return (Join-Path $script:RustRoot 'target\release')
}

function Get-JavaImageRoot {
    return (Join-Path $script:JavaRoot 'build\jpackage\LightTokenWorkstation')
}

function Get-InstalledLauncher {
    return (Join-Path $script:InstalledAppRoot 'LightTokenWorkstation.exe')
}

function Get-InstalledNativeDir {
    return (Join-Path $script:InstalledAppRoot 'app\native')
}

function Get-InstalledCppLibrary {
    return (Join-Path (Get-InstalledNativeDir) 'lighttoken_accel.dll')
}

function Write-InstallManifest([string]$Architecture) {
    Ensure-Directory $script:InstallRoot
    $manifest = [ordered]@{
        product = 'A-LMI LightToken Workstation'
        architecture = $Architecture
        install_root = $script:InstallRoot
        data_root = $script:DataRoot
        launcher = Get-InstalledLauncher
        native_dir = Get-InstalledNativeDir
        installed_at_utc = [DateTime]::UtcNow.ToString('o')
        data_policy = 'Uninstall preserves user-local LightToken data unless exact separate deletion confirmation is provided.'
    }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $script:InstallManifest -Encoding UTF8
}
