$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$cargo = Require-Command 'cargo' 'Install Rust using rustup.'
$python = Require-Command 'python' 'Install Python 3.11 or newer.'
$gradle = Join-Path $script:JavaRoot 'gradlew.bat'
if (-not (Test-Path -LiteralPath $gradle)) { throw "Committed Gradle wrapper missing: $gradle" }

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("lighttoken-windows-test-" + [Guid]::NewGuid().ToString('N'))
$fixtures = Join-Path $tempRoot 'fixtures'
$repoFixtures = Join-Path $script:RepoRoot 'tests\fixtures\lighttoken'
$workspace = Join-Path $tempRoot 'workspace'
$bundle = Join-Path $tempRoot 'verified.cosmos'
$corrupt = Join-Path $tempRoot 'corrupt.cosmos'

try {
    Ensure-Directory $tempRoot
    Invoke-External $python @('-m', 'pip', 'install', '-e', $script:RepoRoot, 'pytest')
    Invoke-External $python @((Join-Path $script:RepoRoot 'scripts\generate_lighttoken_fixtures.py'), '--output', $fixtures)
    if (Test-Path -LiteralPath $repoFixtures) { Remove-Item -LiteralPath $repoFixtures -Recurse -Force }
    Copy-Item -LiteralPath $fixtures -Destination $repoFixtures -Recurse -Force
    Invoke-External $python @((Join-Path $script:RepoRoot 'scripts\generate_lighttoken_almi_fixture.py'), '--lighttoken-dir', $fixtures, '--workspace', $workspace, '--bundle', $bundle, '--corrupt-bundle', $corrupt)

    $env:LIGHTTOKEN_ALMI_WORKSPACE = $workspace
    $env:LIGHTTOKEN_ALMI_BUNDLE = $bundle
    $env:LIGHTTOKEN_ALMI_CORRUPT_BUNDLE = $corrupt

    Invoke-External $cargo @('fmt', '--manifest-path', (Join-Path $script:RustRoot 'Cargo.toml'), '--all', '--', '--check')
    Invoke-External $cargo @('clippy', '--manifest-path', (Join-Path $script:RustRoot 'Cargo.toml'), '-p', 'lighttoken-io', '-p', 'lighttoken-ffi', '--all-targets', '--', '-D', 'warnings')
    Invoke-External $cargo @('test', '--manifest-path', (Join-Path $script:RustRoot 'Cargo.toml'), '--workspace')

    & (Join-Path $PSScriptRoot 'build.ps1') -SkipTests
    if (-not $?) { throw 'Native/package build failed.' }

    $nativeDir = Get-RustReleaseDir
    Invoke-External $gradle @(
        'test',
        "-Dlighttoken.native.dir=$nativeDir",
        "-Dlighttoken.fixture.dir=$fixtures",
        "-Dlighttoken.almi.workspace=$workspace",
        "-Dlighttoken.almi.bundle=$bundle",
        "-Dlighttoken.almi.corrupt_bundle=$corrupt",
        '--no-daemon'
    )
    Write-Step 'TEST PASS'
}
finally {
    Remove-Item Env:LIGHTTOKEN_ALMI_WORKSPACE -ErrorAction SilentlyContinue
    Remove-Item Env:LIGHTTOKEN_ALMI_BUNDLE -ErrorAction SilentlyContinue
    Remove-Item Env:LIGHTTOKEN_ALMI_CORRUPT_BUNDLE -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $repoFixtures) { Remove-Item -LiteralPath $repoFixtures -Recurse -Force }
    if (Test-Path -LiteralPath $tempRoot) { Remove-Item -LiteralPath $tempRoot -Recurse -Force }
}
