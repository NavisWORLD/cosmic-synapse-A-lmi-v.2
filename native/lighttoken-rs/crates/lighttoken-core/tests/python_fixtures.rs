use std::fs;
use std::path::{Path, PathBuf};

use lighttoken_core::{
    from_json_bytes, EMBEDDING_DIMENSION, LIGHTTOKEN_SCHEMA_VERSION, SPECTRAL_DIMENSION,
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("repository root")
        .to_path_buf()
}

fn fixture(name: &str) -> Vec<u8> {
    fs::read(
        repo_root()
            .join("tests/fixtures/lighttoken")
            .join(name),
    )
    .expect("generated LightToken fixture")
}

#[test]
fn parses_active_python_fixture() {
    let token = from_json_bytes(&fixture("active_random.json")).unwrap();
    assert_eq!(LIGHTTOKEN_SCHEMA_VERSION, 1);
    assert_eq!(
        token.joint_embedding.as_ref().unwrap().len(),
        EMBEDDING_DIMENSION
    );
    assert_eq!(
        token.spectral_signature.as_ref().unwrap().len(),
        SPECTRAL_DIMENSION
    );
    token.validate().unwrap();
}

#[test]
fn parses_supported_historical_magnitude_phase_fixture() {
    let token = from_json_bytes(&fixture("historical_magnitude_phase.json")).unwrap();
    assert_eq!(token.spectral_signature.as_ref().unwrap().len(), 1536);
    assert_eq!(token.spectral_transform(), "legacy_embedding_fft");
}

#[test]
fn rejects_incomplete_active_spectral_payload() {
    let mut value: serde_json::Value =
        serde_json::from_slice(&fixture("active_random.json")).unwrap();
    value.as_object_mut().unwrap().remove("spectral_signature_imag");
    let encoded = serde_json::to_vec(&value).unwrap();
    let error = from_json_bytes(&encoded).unwrap_err().to_string();
    assert!(error.contains("spectral"));
}

#[test]
fn canonical_writer_matches_python_fixture_bytes() {
    let original = fixture("active_random.json");
    let token = from_json_bytes(&original).unwrap();
    assert_eq!(token.canonical_json_bytes().unwrap(), original);
}
