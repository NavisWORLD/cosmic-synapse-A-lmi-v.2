$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$python = Require-Command 'python' 'Python 3.11+ is required for deterministic verification fixtures.'
$gradle = Join-Path $script:JavaRoot 'gradlew.bat'
$cli = Join-Path $script:InstalledBinRoot 'lighttoken.exe'
$nativeDir = Get-InstalledNativeDir
if (-not (Test-Path -LiteralPath $cli)) { throw 'Installed LightToken CLI is missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $nativeDir 'lighttoken_ffi.dll'))) { throw 'Installed JNI DLL is missing.' }

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("lighttoken-windows-verify-" + [Guid]::NewGuid().ToString('N'))
$fixtures = Join-Path $tempRoot 'fixtures'
$workspace = Join-Path $tempRoot 'workspace'
$bundle = Join-Path $tempRoot 'verified.cosmos'
$corrupt = Join-Path $tempRoot 'corrupt.cosmos'
$indexDir = Join-Path $tempRoot 'index'

try {
    Ensure-Directory $tempRoot
    Invoke-External $python @('-m', 'pip', 'install', '-e', $script:RepoRoot)
    Invoke-External $python @((Join-Path $script:RepoRoot 'scripts\generate_lighttoken_fixtures.py'), '--output', $fixtures)
    Invoke-External $python @((Join-Path $script:RepoRoot 'scripts\generate_lighttoken_almi_fixture.py'), '--lighttoken-dir', $fixtures, '--workspace', $workspace, '--bundle', $bundle, '--corrupt-bundle', $corrupt)

    $query = Join-Path $fixtures 'active_sinusoid.json'
    $indexInput = Join-Path $fixtures 'active_random.json'
    Invoke-External $cli @('--json', 'validate', $query)
    Invoke-External $cli @('--json', 'index', 'build', $indexInput, $indexDir)
    Invoke-External $cli @('--json', 'search', $indexDir, $query, '--top-k', '3', '--method', 'cosine')

    Invoke-External $gradle @(
        '-p', $script:JavaRoot,
        'test',
        '--tests', 'world.navis.lighttoken.service.WorkspaceServiceIntegrationTest',
        "-Dlighttoken.native.dir=$nativeDir",
        "-Dlighttoken.fixture.dir=$fixtures",
        "-Dlighttoken.almi.workspace=$workspace",
        "-Dlighttoken.almi.bundle=$bundle",
        "-Dlighttoken.almi.corrupt_bundle=$corrupt",
        '--no-daemon'
    )

    & (Join-Path $PSScriptRoot 'run.ps1') -Smoke
    if (-not $?) { throw 'Installed packaged launcher smoke failed.' }
    Write-Step 'VERIFY PASS: synthetic collection, verified workspace, verified .cosmos, corrupt .cosmos fail-closed, packaged JNI launcher.'
}
finally {
    if (Test-Path -LiteralPath $tempRoot) { Remove-Item -LiteralPath $tempRoot -Recurse -Force }
}
