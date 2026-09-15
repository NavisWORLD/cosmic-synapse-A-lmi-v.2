import json
import stat
import zipfile
from pathlib import Path

import pytest

from a_lmi.continuity import (
    ContinuityIntegrityError,
    ContinuitySecurityError,
    append_memory_record,
    export_bundle,
    import_bundle,
    initialize_workspace,
    inspect_workspace,
    verify_bundle,
)


def test_initialize_workspace_separates_state_memory_provider_and_authority(tmp_path: Path):
    workspace = tmp_path / "cosmos"
    report = initialize_workspace(workspace, name="Cosmos", seed=7)

    assert report["format_version"] == 1
    assert report["name"] == "Cosmos"
    assert (workspace / "memory" / "ledger.jsonl").read_text() == ""

    cst = json.loads((workspace / "state" / "cst.json").read_text())
    assert cst["version"] == 1
    assert cst["seed"] == 7

    provider = json.loads((workspace / "provenance" / "provider.json").read_text())
    assert provider["provider_id"] is None
    assert provider["model_id"] is None

    policy = json.loads((workspace / "policy" / "authority.json").read_text())
    assert policy["tool_authority"] == []
    assert policy["network_authority"] == []
    assert policy["filesystem_authority"] == []


def test_export_is_deterministic_verifiable_and_round_trips(tmp_path: Path):
    workspace = tmp_path / "source"
    initialize_workspace(workspace, name="Portable Cosmos", seed=11)
    append_memory_record(
        workspace,
        {
            "role": "user",
            "content": "remember the sunflower",
            "provider": None,
        },
    )

    first = tmp_path / "first.cosmos"
    second = tmp_path / "second.cosmos"
    export_bundle(workspace, first)
    export_bundle(workspace, second)

    assert first.read_bytes() == second.read_bytes()
    verified = verify_bundle(first)
    assert verified["valid"] is True
    assert verified["file_count"] >= 6

    restored = tmp_path / "restored"
    imported = import_bundle(first, restored)
    assert imported["valid"] is True
    summary = inspect_workspace(restored)
    assert summary["name"] == "Portable Cosmos"
    assert summary["memory_records"] == 1
    assert summary["authority"]["tool_authority"] == []


def test_export_rejects_secret_bearing_files(tmp_path: Path):
    workspace = tmp_path / "source"
    initialize_workspace(workspace, name="No Secrets")
    (workspace / ".env").write_text("TOKEN=do-not-export\n")

    with pytest.raises(ContinuitySecurityError, match="secret"):
        export_bundle(workspace, tmp_path / "unsafe.cosmos")


def test_import_rejects_path_traversal_before_extraction(tmp_path: Path):
    bundle = tmp_path / "traversal.cosmos"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../escape.txt", b"owned")

    with pytest.raises(ContinuitySecurityError, match="unsafe archive path"):
        import_bundle(bundle, tmp_path / "destination")

    assert not (tmp_path / "escape.txt").exists()


def test_import_rejects_symlink_members(tmp_path: Path):
    bundle = tmp_path / "symlink.cosmos"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(info, "target")

    with pytest.raises(ContinuitySecurityError, match="symlink"):
        import_bundle(bundle, tmp_path / "destination")


def test_verify_detects_payload_hash_corruption(tmp_path: Path):
    workspace = tmp_path / "source"
    initialize_workspace(workspace, name="Integrity")
    bundle = tmp_path / "good.cosmos"
    export_bundle(workspace, bundle)

    with zipfile.ZipFile(bundle, "r") as original:
        members = {info.filename: original.read(info.filename) for info in original.infolist()}

    members["memory/ledger.jsonl"] = b"tampered\n"
    corrupted = tmp_path / "corrupted.cosmos"
    with zipfile.ZipFile(corrupted, "w") as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])

    with pytest.raises(ContinuityIntegrityError, match="hash mismatch"):
        verify_bundle(corrupted)


def test_verify_rejects_undeclared_archive_members(tmp_path: Path):
    workspace = tmp_path / "source"
    initialize_workspace(workspace, name="Declared Only")
    bundle = tmp_path / "good.cosmos"
    export_bundle(workspace, bundle)

    with zipfile.ZipFile(bundle, "r") as original:
        members = {info.filename: original.read(info.filename) for info in original.infolist()}

    members["surprise.txt"] = b"not in manifest"
    changed = tmp_path / "extra.cosmos"
    with zipfile.ZipFile(changed, "w") as archive:
        for name in sorted(members):
            archive.writestr(name, members[name])

    with pytest.raises(ContinuityIntegrityError, match="undeclared"):
        verify_bundle(changed)


def test_import_refuses_non_empty_destination(tmp_path: Path):
    workspace = tmp_path / "source"
    initialize_workspace(workspace, name="Destination Safety")
    bundle = tmp_path / "bundle.cosmos"
    export_bundle(workspace, bundle)

    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("preserve me")

    with pytest.raises(ContinuitySecurityError, match="destination"):
        import_bundle(bundle, destination)

    assert (destination / "keep.txt").read_text() == "preserve me"
