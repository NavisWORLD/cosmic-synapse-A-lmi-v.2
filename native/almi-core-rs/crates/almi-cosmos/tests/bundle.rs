use almi_continuity::{initialize_workspace, inspect_workspace};
use almi_cosmos::{export_bundle, import_bundle, validate_archive_name, verify_bundle};

#[test]
fn rust_export_is_deterministic_and_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("source");
    initialize_workspace(&root, "Portable", 5).unwrap();
    let a = dir.path().join("a.cosmos");
    let b = dir.path().join("b.cosmos");
    export_bundle(&root, &a).unwrap();
    export_bundle(&root, &b).unwrap();
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    let verified = verify_bundle(&a).unwrap();
    assert!(verified.valid);
    let restored = dir.path().join("restored");
    import_bundle(&a, &restored).unwrap();
    assert_eq!(
        inspect_workspace(&restored).unwrap().system.name,
        "Portable"
    );
}

#[test]
fn hostile_paths_fail_closed() {
    for name in ["../x", "/absolute", "C:/absolute", "a\\b", "./dot"] {
        assert!(validate_archive_name(name).is_err(), "accepted {name}");
    }
}

#[test]
fn export_rejects_secret_file() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("source");
    initialize_workspace(&root, "Secret", 0).unwrap();
    std::fs::write(root.join(".env"), "TOKEN=nope").unwrap();
    assert!(export_bundle(&root, dir.path().join("bad.cosmos")).is_err());
}
