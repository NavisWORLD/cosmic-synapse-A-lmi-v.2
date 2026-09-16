import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

from a_lmi.continuity import export_bundle, import_bundle, initialize_workspace, inspect_workspace, verify_bundle


NATIVE = os.environ.get("ALMI_NATIVE_CLI")
GOLDEN_FIXTURE = Path(__file__).parent / "fixtures" / "native_golden_vectors.json"


def _native(*args: str) -> dict:
    assert NATIVE, "ALMI_NATIVE_CLI must point to the built native CLI"
    result = subprocess.run([NATIVE, "--json", *args], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _materialize_golden_workspace(destination: Path) -> dict:
    fixture = json.loads(GOLDEN_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["fixture_version"] == 1
    for relative, value in fixture["json_files"].items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical_json_bytes(value))

    ledger = destination / "memory" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(
        b"".join(_canonical_json_bytes(record) for record in fixture["memory_records"])
    )

    for relative, expected in fixture["expected_payload_sha256"].items():
        assert _sha256((destination / relative).read_bytes()) == expected
    return fixture


def _bundle_manifest_bytes(bundle: Path) -> bytes:
    with zipfile.ZipFile(bundle, "r") as archive:
        return archive.read("bundle-manifest.json")


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
    assert inspect_workspace(restored)["name"] == "Python Oracle"


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
    assert summary["name"] == "Rust Native"
    authority = summary["authority"]
    assert authority["tool_authority"] == []
    assert authority["network_authority"] == []
    assert authority["filesystem_authority"] == []


def test_committed_golden_workspace_has_identical_cross_language_manifest(tmp_path: Path):
    source = tmp_path / "golden-source"
    fixture = _materialize_golden_workspace(source)

    python_bundle = tmp_path / "golden-python.cosmos"
    rust_bundle = tmp_path / "golden-rust.cosmos"
    export_bundle(source, python_bundle)
    _native("export", str(source), str(rust_bundle))

    assert verify_bundle(python_bundle)["valid"] is True
    assert verify_bundle(rust_bundle)["valid"] is True
    assert _native("verify", str(python_bundle))["valid"] is True
    assert _native("verify", str(rust_bundle))["valid"] is True

    python_manifest = _bundle_manifest_bytes(python_bundle)
    rust_manifest = _bundle_manifest_bytes(rust_bundle)
    assert python_manifest == rust_manifest
    assert _sha256(python_manifest) == fixture["expected_manifest_sha256"]

    manifest = json.loads(python_manifest)
    expected_hashes = fixture["expected_payload_sha256"]
    assert {entry["path"]: entry["sha256"] for entry in manifest["files"]} == expected_hashes

    with zipfile.ZipFile(python_bundle, "r") as python_zip, zipfile.ZipFile(
        rust_bundle, "r"
    ) as rust_zip:
        for relative, expected in expected_hashes.items():
            python_payload = python_zip.read(relative)
            rust_payload = rust_zip.read(relative)
            assert python_payload == rust_payload
            assert _sha256(python_payload) == expected

    rust_restored = tmp_path / "golden-rust-restored"
    python_restored = tmp_path / "golden-python-restored"
    _native("import", str(python_bundle), str(rust_restored))
    import_bundle(rust_bundle, python_restored)
    assert inspect_workspace(rust_restored)["name"] == fixture["workspace_name"]
    assert inspect_workspace(python_restored)["name"] == fixture["workspace_name"]
