param(
    [switch]$SkipTests
)

. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows
$arch = Get-HostArchitecture
$cmakeArch = Get-CMakeArchitecture
$cargo = Require-Command 'cargo' 'Install Rust using rustup.'
$cmake = Require-Command 'cmake' 'Install CMake 3.24 or newer.'
$java = Require-Command 'java' 'Install a Java 21 JDK.'
$python = Require-Command 'python' 'Install Python 3.11 or newer.'
$gradle = Join-Path $script:JavaRoot 'gradlew.bat'
if (-not (Test-Path -LiteralPath $gradle)) { throw "Committed Gradle wrapper missing: $gradle" }

Write-Step "Building LightToken workstation for Windows $arch"
Invoke-External $cargo @('build', '--manifest-path', (Join-Path $script:RustRoot 'Cargo.toml'), '--release', '-p', 'lighttoken-cli', '-p', 'lighttoken-ffi')

$cppBuild = Join-Path $script:CppRoot 'build-windows'
Invoke-External $cmake @('-S', $script:CppRoot, '-B', $cppBuild, '-A', $cmakeArch)
Invoke-External $cmake @('--build', $cppBuild, '--config', 'Release')
if (-not $SkipTests) {
    Invoke-External $cmake @('--build', $cppBuild, '--target', 'test', '--config', 'Release')
}

Invoke-External $java @('-version')
Invoke-External $python @('--version')
Invoke-External $gradle @('clean', 'jpackageImage', '--no-daemon')

$rustRelease = Get-RustReleaseDir
$cli = Join-Path $rustRelease 'lighttoken.exe'
$ffi = Join-Path $rustRelease 'lighttoken_ffi.dll'
if (-not (Test-Path -LiteralPath $cli)) { throw "Rust CLI missing: $cli" }
if (-not (Test-Path -LiteralPath $ffi)) { throw "Rust JNI DLL missing: $ffi" }

$cppCandidates = @(
    (Join-Path $cppBuild 'Release\lighttoken_accel.dll'),
    (Join-Path $cppBuild 'lighttoken_accel.dll')
)
$cpp = $cppCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

$image = Get-JavaImageRoot
if (-not (Test-Path -LiteralPath (Join-Path $image 'LightTokenWorkstation.exe'))) {
    throw "jpackageImage did not produce the workstation launcher: $image"
}

$dist = Get-BuildDistRoot
if (Test-Path -LiteralPath $dist) { Remove-Item -LiteralPath $dist -Recurse -Force }
Ensure-Directory $dist
Ensure-Directory (Join-Path $dist 'bin')
Copy-Item -LiteralPath $cli -Destination (Join-Path $dist 'bin\lighttoken.exe') -Force
Copy-Item -LiteralPath $image -Destination (Join-Path $dist 'app') -Recurse -Force
$nativeDir = Join-Path $dist 'app\app\native'
Ensure-Directory $nativeDir
Copy-Item -LiteralPath $ffi -Destination (Join-Path $nativeDir 'lighttoken_ffi.dll') -Force
if ($cpp) {
    Copy-Item -LiteralPath $cpp -Destination (Join-Path $nativeDir 'lighttoken_accel.dll') -Force
    Copy-Item -LiteralPath $cpp -Destination (Join-Path $dist 'bin\lighttoken_accel.dll') -Force
}
else {
    Write-Step 'Optional C++ accelerator DLL was not found; packaged workstation will use Rust fallback.'
}

Write-Step "Packaged output: $dist"
Write-Step 'BUILD PASS'
