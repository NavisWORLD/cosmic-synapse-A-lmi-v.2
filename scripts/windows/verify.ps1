. (Join-Path $PSScriptRoot 'common.ps1')
Assert-Windows

$binary = Get-InstalledBinary
if (-not (Test-Path -LiteralPath $binary)) {
    $binary = Get-ReleaseBinary
}
if (-not (Test-Path -LiteralPath $binary)) {
    throw 'A-LMI native executable is missing. Run INSTALL_WINDOWS.bat first.'
}

Write-Step "Verifying executable: $binary"
$version = Read-AlmiJson $binary @('version')
if (-not $version.version) { throw 'Version command did not return a version.' }
$doctor = Read-AlmiJson $binary @('doctor')
if ($doctor.status -ne 'ok') { throw 'Doctor command did not report ok.' }

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("almi-windows-verify-" + [Guid]::NewGuid().ToString('N'))
$source = Join-Path $tempRoot 'source'
$restored = Join-Path $tempRoot 'restored'
$bundleA = Join-Path $tempRoot 'a.cosmos'
$bundleB = Join-Path $tempRoot 'b.cosmos'

try {
    Ensure-Directory $tempRoot
    $created = Read-AlmiJson $binary @('init', $source, '--name', 'Windows Verify', '--seed', '77')
    $inspection = Read-AlmiJson $binary @('inspect', $source)
    Assert-DenyAuthority $inspection

    $exportA = Read-AlmiJson $binary @('export', $source, $bundleA)
    $exportB = Read-AlmiJson $binary @('export', $source, $bundleB)
    if (-not $exportA.valid -or -not $exportB.valid) { throw 'Bundle export did not report valid.' }

    $hashA = (Get-FileHash -LiteralPath $bundleA -Algorithm SHA256).Hash
    $hashB = (Get-FileHash -LiteralPath $bundleB -Algorithm SHA256).Hash
    if ($hashA -ne $hashB) {
        throw "Deterministic export check failed: $hashA != $hashB"
    }

    $verified = Read-AlmiJson $binary @('verify', $bundleA)
    if (-not $verified.valid) { throw 'Bundle verification did not report valid.' }

    $imported = Read-AlmiJson $binary @('import', $bundleA, $restored)
    $restoredInspection = Read-AlmiJson $binary @('inspect', $restored)
    if ($restoredInspection.system.name -ne 'Windows Verify') { throw 'Imported workspace identity mismatch.' }
    Assert-DenyAuthority $restoredInspection

    Write-Step "Deterministic bundle SHA-256: $hashA"
    Write-Step 'VERIFY PASS'
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
