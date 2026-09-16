Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script:NativeRoot = Join-Path $script:RepoRoot 'native\almi-core-rs'
$script:InstallRoot = if ($env:ALMI_INSTALL_ROOT) { $env:ALMI_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA 'A-LMI' }
$script:BinDir = Join-Path $script:InstallRoot 'bin'
$script:InstallManifest = Join-Path $script:InstallRoot 'install.json'

function Write-Step([string]$Message) {
    Write-Host "[A-LMI] $Message"
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
    if (-not $path) {
        throw "$Name was not found. $Guidance"
    }
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

function Get-InstalledBinary {
    return (Join-Path $script:BinDir 'almi.exe')
}

function Get-ReleaseBinary {
    return (Join-Path $script:NativeRoot 'target\release\almi.exe')
}

function Get-DebugBinary {
    return (Join-Path $script:NativeRoot 'target\debug\almi.exe')
}

function Read-AlmiJson([string]$Binary, [string[]]$Arguments) {
    $output = & $Binary '--json' @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "A-LMI command failed with exit code $LASTEXITCODE: $($Arguments -join ' ')"
    }
    return ($output | Out-String | ConvertFrom-Json)
}

function Assert-DenyAuthority($Inspection) {
    foreach ($name in @('tool_authority', 'network_authority', 'filesystem_authority')) {
        $value = $Inspection.authority.$name
        if ($null -eq $value -or @($value).Count -ne 0) {
            throw "Authority default check failed: $name must be empty."
        }
    }
}

function Write-InstallManifest([bool]$PythonBindingsInstalled, [string]$Architecture) {
    Ensure-Directory $script:InstallRoot
    $manifest = [ordered]@{
        product = 'A-LMI Native Core'
        install_root = $script:InstallRoot
        executable = Get-InstalledBinary
        architecture = $Architecture
        python_bindings_installed = $PythonBindingsInstalled
        installed_at_utc = [DateTime]::UtcNow.ToString('o')
        user_data_policy = 'Program uninstall does not delete user workspaces, .cosmos bundles, memories, or backups.'
    }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $script:InstallManifest -Encoding UTF8
}
