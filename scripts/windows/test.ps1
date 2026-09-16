. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$cargo = Require-Command 'cargo' 'Install Rust with rustup before running TEST_WINDOWS.bat.'

Write-Step 'Running Rust unit, integration, and property tests.'
Invoke-External $cargo @('test', '--manifest-path', (Join-Path $script:NativeRoot 'Cargo.toml'), '--workspace')
Invoke-External $cargo @('build', '--manifest-path', (Join-Path $script:NativeRoot 'Cargo.toml'), '-p', 'almi-cli')

$python = Get-CommandPath 'python'
if ($python) {
    Write-Step 'Python detected; running isolated Python/Rust compatibility tests.'
    $tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("almi-python-test-" + [Guid]::NewGuid().ToString('N'))
    $venv = Join-Path $tempRoot 'venv'
    $wheelDir = Join-Path $tempRoot 'wheels'
    try {
        Ensure-Directory $tempRoot
        Invoke-External $python @('-m', 'venv', $venv)
        $venvPython = Join-Path $venv 'Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Temporary Python virtual environment was not created.' }
        Invoke-External $venvPython @('-m', 'pip', 'install', '--upgrade', 'pip')
        Invoke-External $venvPython @('-m', 'pip', 'install', '-e', $script:RepoRoot, 'pytest', 'maturin>=1.7,<2.0')
        Ensure-Directory $wheelDir
        Invoke-External $venvPython @('-m', 'maturin', 'build', '--release', '--manifest-path', (Join-Path $script:NativeRoot 'crates\almi-python\Cargo.toml'), '--out', $wheelDir)
        $wheel = Get-ChildItem -LiteralPath $wheelDir -Filter '*.whl' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -eq $wheel) { throw 'Native Python wheel was not produced.' }
        Invoke-External $venvPython @('-m', 'pip', 'install', $wheel.FullName)
        $env:ALMI_NATIVE_CLI = Get-DebugBinary
        Invoke-External $venvPython @('-m', 'pytest', '-q',
            (Join-Path $script:RepoRoot 'tests\test_native_interop.py'),
            (Join-Path $script:RepoRoot 'tests\test_native_cst_interop.py'),
            (Join-Path $script:NativeRoot 'crates\almi-python\python-tests\test_bindings.py'))
    }
    finally {
        Remove-Item Env:ALMI_NATIVE_CLI -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $tempRoot) { Remove-Item -LiteralPath $tempRoot -Recurse -Force }
    }
}
else {
    Write-Step 'Python not installed; Python compatibility tests were skipped. Native Rust tests still ran.'
}

Write-Step 'TEST PASS'
