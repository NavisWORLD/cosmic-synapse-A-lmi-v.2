use lighttoken_core::SimilarityMethod;
use lighttoken_spectrum::{backend_diagnostics, similarity, similarity_many, BackendKind};

#[test]
fn explicit_cpp_disable_forces_rust_with_identical_scores() {
    std::env::set_var("LIGHTTOKEN_DISABLE_CPP", "1");

    let diagnostics = backend_diagnostics();
    assert_eq!(diagnostics.active, BackendKind::Rust);
    assert!(!diagnostics.cpp_available);

    let query = vec![1.0f32, 2.0, 3.0, 4.0];
    let candidates = vec![
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![4.0f32, 3.0, 2.0, 1.0],
    ];

    for method in [
        SimilarityMethod::PowerCorrelation,
        SimilarityMethod::Cosine,
        SimilarityMethod::Euclidean,
    ] {
        let batch = similarity_many(&query, &candidates, method).unwrap();
        let expected: Vec<f32> = candidates
            .iter()
            .map(|candidate| similarity(&query, candidate, method).unwrap())
            .collect();
        assert_eq!(batch, expected);
    }

    std::env::remove_var("LIGHTTOKEN_DISABLE_CPP");
}
