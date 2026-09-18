use lighttoken_core::SimilarityMethod;
use lighttoken_spectrum::{backend_diagnostics, similarity, similarity_many, BackendKind};

#[test]
fn trusted_cpp_library_matches_rust_known_vectors() {
    let library = std::env::var("LIGHTTOKEN_CPP_LIB")
        .expect("LIGHTTOKEN_CPP_LIB must point to the CI-built application-owned accelerator");

    let diagnostics = backend_diagnostics();
    assert_eq!(diagnostics.active, BackendKind::Cpp, "library={library} detail={}", diagnostics.detail);
    assert!(diagnostics.cpp_available);

    let query = vec![1.0f32, 2.0, 3.0, 4.0];
    let candidates = vec![
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![4.0f32, 3.0, 2.0, 1.0],
        vec![0.0f32, 0.0, 0.0, 0.0],
    ];

    for method in [
        SimilarityMethod::PowerCorrelation,
        SimilarityMethod::Cosine,
        SimilarityMethod::Euclidean,
    ] {
        let accelerated = similarity_many(&query, &candidates, method).unwrap();
        for (candidate, actual) in candidates.iter().zip(accelerated.iter()) {
            let expected = similarity(&query, candidate, method).unwrap();
            assert!((actual - expected).abs() <= 2.0e-5, "{method:?}: {actual} != {expected}");
        }
    }
}
