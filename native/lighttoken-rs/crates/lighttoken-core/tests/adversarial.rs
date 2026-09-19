use lighttoken_core::{from_json_bytes, EMBEDDING_DIMENSION, SPECTRAL_DIMENSION};
use proptest::prelude::*;
use serde_json::json;

fn minimal_token() -> serde_json::Value {
    json!({
        "token_id": "00000000-0000-0000-0000-000000000123",
        "timestamp": "2000-01-01T00:00:00+00:00",
        "source_uri": "fixture://adversarial",
        "modality": "synthetic",
        "raw_data_ref": "fixture://raw/adversarial",
        "metadata": {}
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn every_short_embedding_dimension_is_rejected(length in 0usize..EMBEDDING_DIMENSION) {
        let mut value = minimal_token();
        value["joint_embedding"] = json!(vec![0.0f32; length]);
        let payload = serde_json::to_vec(&value).unwrap();
        prop_assert!(from_json_bytes(&payload).is_err());
    }

    #[test]
    fn every_short_active_spectrum_dimension_is_rejected(length in 0usize..SPECTRAL_DIMENSION) {
        let mut value = minimal_token();
        value["spectral_signature_real"] = json!(vec![0.0f32; length]);
        value["spectral_signature_imag"] = json!(vec![0.0f32; length]);
        let payload = serde_json::to_vec(&value).unwrap();
        prop_assert!(from_json_bytes(&payload).is_err());
    }
}

#[test]
fn non_finite_and_float32_overflow_fail_closed() {
    for payload in [
        r#"{"token_id":"x","timestamp":"t","source_uri":"u","modality":"m","raw_data_ref":"r","joint_embedding":[NaN]}"#,
        r#"{"token_id":"x","timestamp":"t","source_uri":"u","modality":"m","raw_data_ref":"r","joint_embedding":[Infinity]}"#,
        r#"{"token_id":"x","timestamp":"t","source_uri":"u","modality":"m","raw_data_ref":"r","joint_embedding":[1e999]}"#,
    ] {
        assert!(from_json_bytes(payload.as_bytes()).is_err());
    }
    let mut token = minimal_token();
    token["joint_embedding"] = json!(vec![1e50f64; EMBEDDING_DIMENSION]);
    assert!(from_json_bytes(&serde_json::to_vec(&token).unwrap()).is_err());
}

#[test]
fn mismatched_active_spectral_components_fail_closed() {
    let mut token = minimal_token();
    token["spectral_signature_real"] = json!(vec![0.0f32; SPECTRAL_DIMENSION]);
    token["spectral_signature_imag"] = json!(vec![0.0f32; SPECTRAL_DIMENSION - 1]);
    assert!(from_json_bytes(&serde_json::to_vec(&token).unwrap()).is_err());
}
