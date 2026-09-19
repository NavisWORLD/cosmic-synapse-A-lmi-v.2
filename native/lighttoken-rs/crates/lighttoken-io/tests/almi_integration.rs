use std::path::PathBuf;

use lighttoken_io::{read_verified_cosmos, read_verified_workspace};

fn required(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set")))
}

#[test]
fn verified_workspace_and_cosmos_discover_the_synthetic_lighttoken() {
    let workspace = read_verified_workspace(required("LIGHTTOKEN_ALMI_WORKSPACE")).unwrap();
    assert_eq!(workspace.source_kind, "workspace");
    assert!(!workspace.bundle_verified);
    assert_eq!(workspace.tokens.len(), 1);
    let token = &workspace.tokens[0];
    assert!(token.resolvable);
    assert!(token.verified);
    assert!(token.raw_resolvable);
    assert_eq!(token.token_id, "00000000-0000-0000-0000-000000000004");
    assert!(token.canonical_json.as_deref().is_some_and(|json| json.contains("joint_embedding")));

    let bundle = read_verified_cosmos(required("LIGHTTOKEN_ALMI_BUNDLE")).unwrap();
    assert_eq!(bundle.source_kind, "cosmos");
    assert!(bundle.bundle_verified);
    assert_eq!(bundle.tokens.len(), 1);
    assert!(bundle.tokens[0].verified);
}

#[test]
fn corrupted_cosmos_is_rejected_before_discovery() {
    let result = read_verified_cosmos(required("LIGHTTOKEN_ALMI_CORRUPT_BUNDLE"));
    let error = result.expect_err("corrupted bundle must fail closed");
    let message = error.to_string().to_ascii_lowercase();
    assert!(message.contains("integrity") || message.contains("hash mismatch"));
}
