use almi_continuity::{initialize_workspace, inspect_workspace};

#[test]
fn workspace_has_python_compatible_layout_and_deny_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("cosmos");
    initialize_workspace(&root, "Native Cosmos", 11).unwrap();
    for path in [
        "system.json",
        "memory/ledger.jsonl",
        "state/cst.json",
        "knowledge/graph.json",
        "artifacts/manifest.json",
        "provenance/provider.json",
        "routing/state.json",
        "policy/authority.json",
    ] {
        assert!(root.join(path).is_file(), "missing {path}");
    }
    let summary = inspect_workspace(&root).unwrap();
    assert_eq!(summary.system.name, "Native Cosmos");
    assert_eq!(summary.memory_records, 0);
    assert!(summary.authority.tool_authority.is_empty());
    assert!(summary.authority.network_authority.is_empty());
    assert!(summary.authority.filesystem_authority.is_empty());
    assert!(summary.authority.cloud_authority.is_empty());
    assert!(summary.authority.deployment_authority.is_empty());
    assert!(summary.authority.actuator_authority.is_empty());
}

#[test]
fn non_empty_destination_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("cosmos");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("keep.txt"), "keep").unwrap();
    assert!(initialize_workspace(&root, "No overwrite", 0).is_err());
    assert_eq!(
        std::fs::read_to_string(root.join("keep.txt")).unwrap(),
        "keep"
    );
}
