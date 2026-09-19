use std::fs;
use std::path::{Path, PathBuf};

use lighttoken_core::{from_json_bytes, SimilarityMethod, SPECTRAL_DIMENSION};
use lighttoken_spectrum::{dominant_bin, rfft_embedding, spectral_power, token_similarity};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("repository root")
        .to_path_buf()
}

fn fixture(name: &str) -> Vec<u8> {
    fs::read(repo_root().join("tests/fixtures/lighttoken").join(name))
        .expect("generated LightToken fixture")
}

#[test]
fn rust_rfft_matches_python_active_fixture() {
    let token = from_json_bytes(&fixture("active_sinusoid.json")).unwrap();
    let embedding = token.joint_embedding.as_ref().unwrap();
    let expected = token.spectral_signature.as_ref().unwrap();
    let actual = rfft_embedding(embedding).unwrap();
    assert_eq!(actual.len(), SPECTRAL_DIMENSION);
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual.real - expected.real).abs() <= 2.0e-4,
            "real mismatch at {index}: {} vs {}",
            actual.real,
            expected.real
        );
        assert!(
            (actual.imag - expected.imag).abs() <= 2.0e-4,
            "imag mismatch at {index}: {} vs {}",
            actual.imag,
            expected.imag
        );
    }
}

#[test]
fn spectral_power_and_dominant_bin_match_python_fixture() {
    let token = from_json_bytes(&fixture("active_sinusoid.json")).unwrap();
    let expected = token.spectral_signature.as_ref().unwrap();
    let power = spectral_power(expected).unwrap();
    assert_eq!(power.len(), SPECTRAL_DIMENSION);
    let (index, magnitude) = dominant_bin(&power).unwrap();
    assert!(index < SPECTRAL_DIMENSION);
    assert!(magnitude >= 0.0);
    let expected_magnitude = expected[index].real.hypot(expected[index].imag);
    assert!((magnitude - expected_magnitude).abs() <= 2.0e-4);
}

#[test]
fn similarity_matches_python_oracle_scores() {
    let left = from_json_bytes(&fixture("active_sinusoid.json")).unwrap();
    let right = from_json_bytes(&fixture("active_random.json")).unwrap();
    let oracle: serde_json::Value = serde_json::from_slice(&fixture("similarity.json")).unwrap();

    for (method_name, method) in [
        ("power_correlation", SimilarityMethod::PowerCorrelation),
        ("cosine", SimilarityMethod::Cosine),
        ("euclidean", SimilarityMethod::Euclidean),
    ] {
        let expected = oracle["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|pair| {
                pair["a"] == "active_sinusoid"
                    && pair["b"] == "active_random"
                    && pair["method"] == method_name
            })
            .unwrap()["score"]
            .as_f64()
            .unwrap() as f32;
        let actual = token_similarity(&left, &right, method).unwrap();
        assert!(
            (actual - expected).abs() <= 2.0e-4,
            "{method_name}: {actual} vs {expected}"
        );
    }
}

#[test]
fn identical_zero_spectra_preserve_python_degenerate_semantics() {
    let zero = from_json_bytes(&fixture("active_zero.json")).unwrap();
    for method in [
        SimilarityMethod::PowerCorrelation,
        SimilarityMethod::Cosine,
        SimilarityMethod::Euclidean,
    ] {
        assert_eq!(token_similarity(&zero, &zero, method).unwrap(), 1.0);
    }
}

#[test]
fn rejects_wrong_embedding_length() {
    let error = rfft_embedding(&[0.0; 4]).unwrap_err().to_string();
    assert!(error.contains("1536"));
}
