param(
    [switch]$WithPython,
    [switch]$SkipPython
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$arch = Get-HostArchitecture
Write-Step "Windows detected: $([Environment]::OSVersion.Version) / $arch"
Write-Step 'No administrator privileges are required; installation is user-local.'

$cargo = Require-Command 'cargo' 'Install Rust with rustup from https://rustup.rs, then re-run INSTALL_WINDOWS.bat.'
$rustc = Require-Command 'rustc' 'Install Rust with rustup from https://rustup.rs, then re-run INSTALL_WINDOWS.bat.'
Invoke-External $rustc @('--version')
Invoke-External $cargo @('--version')

$python = Get-CommandPath 'python'
$installPython = $WithPython -or ((-not $SkipPython) -and [bool]$python)
if ($WithPython -and -not $python) {
    throw 'Python 3.11+ was requested but python was not found. Install Python and re-run INSTALL_WINDOWS.bat -WithPython.'
}
if (-not $python -and -not $SkipPython) {
    Write-Step 'Python was not found. Native CLI installation will continue; Python bindings are optional.'
}

Write-Step 'Building release CLI and stable C ABI library.'
Invoke-External $cargo @('build', '--manifest-path', (Join-Path $script:NativeRoot 'Cargo.toml'), '--release', '-p', 'almi-cli', '-p', 'almi-ffi')
$releaseBinary = Get-ReleaseBinary
if (-not (Test-Path -LiteralPath $releaseBinary)) {
    throw "Release executable was not produced: $releaseBinary"
}

Ensure-Directory $script:InstallRoot
Ensure-Directory $script:BinDir
$installedBinary = Get-InstalledBinary
Copy-Item -LiteralPath $releaseBinary -Destination $installedBinary -Force

$pythonInstalled = $false
if ($installPython) {
    Write-Step 'Building and installing Python bindings into the current user Python environment.'
    Invoke-External $python @('--version')
    Invoke-External $python @('-m', 'pip', 'install', '--user', 'maturin>=1.7,<2.0')
    $scriptsDir = (& $python -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user'))" | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $scriptsDir) { throw 'Unable to resolve the Python user Scripts directory.' }
    $maturin = Join-Path $scriptsDir 'maturin.exe'
    if (-not (Test-Path -LiteralPath $maturin)) {
        $maturin = Get-CommandPath 'maturin'
    }
    if (-not $maturin) { throw 'maturin was installed but its executable could not be resolved.' }
    $wheelDir = Join-Path $script:NativeRoot 'dist\python'
    Ensure-Directory $wheelDir
    Invoke-External $maturin @('build', '--release', '--manifest-path', (Join-Path $script:NativeRoot 'crates\almi-python\Cargo.toml'), '--out', $wheelDir)
    $wheel = Get-ChildItem -LiteralPath $wheelDir -Filter '*.whl' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $wheel) { throw 'Python binding build completed without producing a wheel.' }
    Invoke-External $python @('-m', 'pip', 'install', '--user', '--force-reinstall', $wheel.FullName)
    $pythonInstalled = $true
}

Write-InstallManifest $pythonInstalled $arch
Write-Step 'Running installed binary smoke test.'
Invoke-External $installedBinary @('doctor')

$verifyScript = Join-Path $PSScriptRoot 'verify.ps1'
if (Test-Path -LiteralPath $verifyScript) {
    & $verifyScript
    if (-not $?) { throw 'Final Windows verification failed.' }
}

Write-Host ''
Write-Host 'A-LMI Native Core installation complete.'
Write-Host "Installed executable: $installedBinary"
Write-Host "Install manifest:     $script:InstallManifest"
Write-Host 'Run examples:'
Write-Host '  RUN_WINDOWS.bat doctor'
Write-Host '  RUN_WINDOWS.bat init MyStory --name "My Story" --seed 1'
Write-Host '  RUN_WINDOWS.bat verify story.cosmos'
Write-Host 'User workspaces and .cosmos bundles are not owned by the uninstaller.'
