use std::fs;
use std::path::{Path, PathBuf};

use almi_continuity::initialize_workspace;
use lighttoken_io::{
    load_collection, load_token_file, read_verified_workspace, IoError, MAX_TOKEN_JSON_BYTES,
};
use serde_json::{json, Value};
use tempfile::tempdir;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("repository root")
        .to_path_buf()
}

fn fixture() -> Vec<u8> {
    fs::read(repo_root().join("tests/fixtures/lighttoken/active_sinusoid.json"))
        .expect("generated Python oracle fixture")
}

fn write_manifest(root: &Path, artifact: Value) {
    fs::write(
        root.join("artifacts/manifest.json"),
        serde_json::to_vec(&json!({"version": 1, "artifacts": [artifact]})).unwrap(),
    )
    .unwrap();
}

#[test]
fn malformed_json_and_jsonl_fail_closed() {
    let temp = tempdir().unwrap();
    let invalid = temp.path().join("invalid.json");
    fs::write(&invalid, b"{malformed").unwrap();
    assert!(load_token_file(&invalid).is_err());

    let jsonl = temp.path().join("tokens.jsonl");
    let mut bytes = fixture();
    bytes.extend_from_slice(b"{malformed}\n");
    fs::write(&jsonl, bytes).unwrap();
    assert!(load_collection(&jsonl).is_err());
}

#[test]
fn oversized_single_token_rejected_before_parsing() {
    let temp = tempdir().unwrap();
    let oversized = temp.path().join("large.json");
    let file = fs::File::create(&oversized).unwrap();
    file.set_len(MAX_TOKEN_JSON_BYTES + 1).unwrap();
    assert!(matches!(
        load_token_file(&oversized),
        Err(IoError::Limit(_))
    ));
}

#[test]
fn manifest_traversal_and_windows_backslashes_fail_closed() {
    let temp = tempdir().unwrap();
    let root = temp.path().join("workspace");
    initialize_workspace(&root, "Synthetic Adversarial", 77).unwrap();

    for relative in [
        "../outside.json",
        "artifacts/../system.json",
        "artifacts\\lighttokens\\token.json",
        "/absolute/token.json",
        "C:/windows/token.json",
    ] {
        write_manifest(
            &root,
            json!({"kind":"lighttoken","path":relative,"token_id":"untrusted"}),
        );
        assert!(
            read_verified_workspace(&root).is_err(),
            "accepted unsafe artifact path: {relative}"
        );
    }
}

#[test]
fn unsafe_workspace_raw_reference_is_not_treated_as_verified() {
    let temp = tempdir().unwrap();
    let root = temp.path().join("workspace");
    initialize_workspace(&root, "Synthetic Adversarial", 78).unwrap();
    let mut token: Value = serde_json::from_slice(&fixture()).unwrap();
    token["raw_data_ref"] = json!("workspace://../outside.txt");
    let target = root.join("artifacts/lighttokens/token.json");
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(&target, serde_json::to_vec(&token).unwrap()).unwrap();
    write_manifest(
        &root,
        json!({"kind":"lighttoken","path":"artifacts/lighttokens/token.json","token_id":token["token_id"]}),
    );
    assert!(read_verified_workspace(&root).is_err());
}

#[cfg(unix)]
#[test]
fn symlinked_artifact_parent_fails_closed() {
    use std::os::unix::fs::symlink;
    let temp = tempdir().unwrap();
    let root = temp.path().join("workspace");
    initialize_workspace(&root, "Synthetic Adversarial", 79).unwrap();
    let outside = temp.path().join("outside");
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join("token.json"), fixture()).unwrap();
    symlink(&outside, root.join("artifacts/bridge")).unwrap();
    write_manifest(
        &root,
        json!({"kind":"lighttoken","path":"artifacts/bridge/token.json"}),
    );
    assert!(read_verified_workspace(&root).is_err());
}

#[cfg(unix)]
#[test]
fn symlinked_raw_parent_fails_closed() {
    use std::os::unix::fs::symlink;
    let temp = tempdir().unwrap();
    let root = temp.path().join("workspace");
    initialize_workspace(&root, "Synthetic Adversarial", 80).unwrap();
    let outside = temp.path().join("outside");
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join("raw.txt"), b"synthetic raw\n").unwrap();
    symlink(&outside, root.join("artifacts/rawbridge")).unwrap();
    let mut token: Value = serde_json::from_slice(&fixture()).unwrap();
    token["raw_data_ref"] = json!("workspace://artifacts/rawbridge/raw.txt");
    let path = root.join("artifacts/lighttokens/token.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, serde_json::to_vec(&token).unwrap()).unwrap();
    write_manifest(
        &root,
        json!({"kind":"lighttoken","path":"artifacts/lighttokens/token.json"}),
    );
    assert!(read_verified_workspace(&root).is_err());
}
