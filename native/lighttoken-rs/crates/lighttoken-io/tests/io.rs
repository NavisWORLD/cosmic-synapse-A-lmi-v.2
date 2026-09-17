use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use almi_continuity::initialize_workspace;
use almi_cosmos::export_bundle;
use lighttoken_io::{
    discover_almi_workspace, import_verified_cosmos, load_collection, load_token_file,
};
use tempfile::tempdir;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("repository root")
        .to_path_buf()
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures/lighttoken").join(name)
}

#[test]
fn loads_single_python_token_file() {
    let token = load_token_file(&fixture("active_random.json")).unwrap();
    assert_eq!(token.token_id, "00000000-0000-0000-0000-000000000005");
}

#[test]
fn loads_jsonl_incrementally_and_preserves_order() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("tokens.jsonl");
    let mut file = fs::File::create(&path).unwrap();
    file.write_all(&fs::read(fixture("active_zero.json")).unwrap())
        .unwrap();
    file.write_all(&fs::read(fixture("active_random.json")).unwrap())
        .unwrap();
    let tokens = load_collection(&path).unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].token_id, "00000000-0000-0000-0000-000000000001");
    assert_eq!(tokens[1].token_id, "00000000-0000-0000-0000-000000000005");
}

#[test]
fn malformed_jsonl_fails_closed_without_partial_success() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("tokens.jsonl");
    let mut bytes = fs::read(fixture("active_zero.json")).unwrap();
    bytes.extend_from_slice(b"{not-json}\n");
    fs::write(&path, bytes).unwrap();
    assert!(load_collection(&path).is_err());
}

#[test]
fn empty_valid_almi_workspace_discovers_no_lighttokens() {
    let temp = tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    initialize_workspace(&workspace, "LightToken IO Synthetic", 424242).unwrap();
    let discovered = discover_almi_workspace(&workspace).unwrap();
    assert!(discovered.is_empty());
}

#[test]
fn cosmos_read_path_delegates_to_verified_almi_importer() {
    let temp = tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    initialize_workspace(&workspace, "LightToken IO Synthetic", 424242).unwrap();
    let bundle = temp.path().join("workspace.cosmos");
    export_bundle(&workspace, &bundle).unwrap();
    let destination = temp.path().join("imported");
    let discovered = import_verified_cosmos(&bundle, &destination).unwrap();
    assert!(discovered.is_empty());
    assert!(destination.join("system.json").is_file());
}
