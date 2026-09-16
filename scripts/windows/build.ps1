param(
    [switch]$WithPython
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$arch = Get-HostArchitecture
$cargo = Require-Command 'cargo' 'Install Rust from https://rustup.rs and re-run BUILD_WINDOWS.bat.'
$rustc = Require-Command 'rustc' 'Install Rust from https://rustup.rs and re-run BUILD_WINDOWS.bat.'

Write-Step "Building A-LMI Native Core for Windows $arch"
Invoke-External $rustc @('--version')
Invoke-External $cargo @('build', '--manifest-path', (Join-Path $script:NativeRoot 'Cargo.toml'), '--release', '-p', 'almi-cli', '-p', 'almi-ffi')

$binary = Get-ReleaseBinary
if (-not (Test-Path -LiteralPath $binary)) {
    throw "Release executable was not produced: $binary"
}
Write-Step "Native release binary: $binary"

if ($WithPython) {
    $python = Require-Command 'python' 'Python 3.11+ is required when -WithPython is requested.'
    Invoke-External $python @('--version')
    Invoke-External $python @('-m', 'pip', 'install', '--user', 'maturin>=1.7,<2.0')
    $wheelDir = Join-Path $script:NativeRoot 'dist\python'
    Ensure-Directory $wheelDir
    Invoke-External $python @('-m', 'maturin', 'build', '--release', '--manifest-path', (Join-Path $script:NativeRoot 'crates\almi-python\Cargo.toml'), '--out', $wheelDir)
    Write-Step "Python wheel output: $wheelDir"
}

Write-Step 'BUILD PASS'
