from pathlib import Path

import almi_native


def test_binding_version_and_workspace_round_trip(tmp_path: Path):
    assert almi_native.abi_version() == 1
    workspace = tmp_path / "story"
    created = almi_native.init_workspace(str(workspace), "Python Native", 23)
    assert created["name"] == "Python Native"
    inspected = almi_native.inspect_workspace(str(workspace))
    assert inspected["authority"]["tool_authority"] == []
    assert inspected["authority"]["network_authority"] == []
    bundle = tmp_path / "story.cosmos"
    exported = almi_native.export_cosmos(str(workspace), str(bundle))
    assert exported["valid"] is True
    verified = almi_native.verify_cosmos(str(bundle))
    assert verified["sha256"] == exported["sha256"]
    restored = tmp_path / "restored"
    imported = almi_native.import_cosmos(str(bundle), str(restored))
    assert imported["valid"] is True
