from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WRAPPERS = {
    "BUILD_LIGHTTOKEN_WINDOWS.bat": "build.ps1",
    "TEST_LIGHTTOKEN_WINDOWS.bat": "test.ps1",
    "INSTALL_LIGHTTOKEN_WINDOWS.bat": "install.ps1",
    "VERIFY_LIGHTTOKEN_WINDOWS.bat": "verify.ps1",
    "RUN_LIGHTTOKEN_WINDOWS.bat": "run.ps1",
    "UPDATE_LIGHTTOKEN_WINDOWS.bat": "update.ps1",
    "UNINSTALL_LIGHTTOKEN_WINDOWS.bat": "uninstall.ps1",
}


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_all_lighttoken_windows_wrappers_are_thin_powershell_entrypoints():
    for wrapper, script in WRAPPERS.items():
        path = ROOT / wrapper
        assert path.is_file(), f"missing {wrapper}"
        text = path.read_text(encoding="utf-8").lower()
        assert "powershell.exe" in text
        assert "-executionpolicy bypass" in text
        assert f"scripts\\windows\\lighttoken\\{script}" in text


def test_windows_common_owns_only_lighttoken_program_and_data_roots():
    text = read("scripts/windows/lighttoken/common.ps1")
    assert "LIGHTTOKEN_INSTALL_ROOT" in text
    assert "A-LMI" in text
    assert "LightToken" in text
    assert "DataRoot" in text
    assert "install.json" in text
    assert "Assert-Windows" in text
    assert "Get-HostArchitecture" in text
    assert "Invoke-External" in text


def test_update_is_fail_closed_and_fast_forward_only():
    text = read("scripts/windows/lighttoken/update.ps1")
    assert "status --porcelain" in text
    assert "local changes" in text.lower()
    assert "'fetch'" in text
    assert "'merge', '--ff-only'" in text
    assert "data" in text.lower()


def test_uninstall_preserves_data_without_exact_confirmation():
    text = read("scripts/windows/lighttoken/uninstall.ps1")
    assert "RemoveData" in text
    assert "DELETE LIGHTTOKEN DATA" in text
    assert "DataRoot" in text
    assert "Remove-Item" in text


def test_build_packages_rust_cpp_java_and_runtime_image():
    text = read("scripts/windows/lighttoken/build.ps1")
    for required in ("cargo", "cmake", "java", "python"):
        assert required in text.lower()
    assert "lighttoken-cli" in text
    assert "lighttoken-ffi" in text
    assert "lighttoken_accel" in text
    assert "jpackageImage" in text


def test_gradle_has_explicit_runtime_and_app_image_tasks():
    text = read("apps/lighttoken-workstation-java/build.gradle")
    assert "jlinkRuntime" in text
    assert "jpackageImage" in text
    assert "jlink" in text
    assert "jpackage" in text


def test_generated_windows_outputs_are_ignored():
    text = read(".gitignore")
    assert "native/lighttoken-cpp/build-windows/" in text
    assert "apps/lighttoken-workstation-java/build/" in text
    assert "dist/lighttoken/" in text
