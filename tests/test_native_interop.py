import json
import os
from pathlib import Path
import subprocess

from a_lmi.continuity import export_bundle, import_bundle, initialize_workspace, inspect_workspace, verify_bundle


NATIVE = os.environ.get("ALMI_NATIVE_CLI")


def _native(*args: str) -> dict:
    assert NATIVE, "ALMI_NATIVE_CLI must point to the built native CLI"
    result = subprocess.run([NATIVE, "--json", *args], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_python_export_rust_verify_and_import(tmp_path: Path):
    source = tmp_path / "python-source"
    initialize_workspace(source, name="Python Oracle", seed=101)
    first = tmp_path / "python-a.cosmos"
    second = tmp_path / "python-b.cosmos"
    export_bundle(source, first)
    export_bundle(source, second)
    assert first.read_bytes() == second.read_bytes()

    verified = _native("verify", str(first))
    assert verified["valid"] is True
    restored = tmp_path / "rust-restored"
    _native("import", str(first), str(restored))
    assert inspect_workspace(restored)["system"]["name"] == "Python Oracle"


def test_rust_export_python_verify_and_import(tmp_path: Path):
    source = tmp_path / "rust-source"
    _native("init", str(source), "--name", "Rust Native", "--seed", "202")
    first = tmp_path / "rust-a.cosmos"
    second = tmp_path / "rust-b.cosmos"
    _native("export", str(source), str(first))
    _native("export", str(source), str(second))
    assert first.read_bytes() == second.read_bytes()

    verified = verify_bundle(first)
    assert verified["valid"] is True
    restored = tmp_path / "python-restored"
    import_bundle(first, restored)
    summary = inspect_workspace(restored)
    assert summary["system"]["name"] == "Rust Native"
    authority = summary["authority"]
    assert authority["tool_authority"] == []
    assert authority["network_authority"] == []
    assert authority["filesystem_authority"] == []
