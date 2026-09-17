use std::fs;
use std::path::{Path, PathBuf};

use lighttoken_core::{from_json_bytes, SimilarityMethod};
use lighttoken_index::{SearchRequest, TokenCollection};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("repository root")
        .to_path_buf()
}

fn token(name: &str) -> lighttoken_core::LightTokenRecord {
    let bytes = fs::read(repo_root().join("tests/fixtures/lighttoken").join(name)).unwrap();
    from_json_bytes(&bytes).unwrap()
}

#[test]
fn equal_scores_are_tied_by_token_id_ascending() {
    let query = token("active_random.json");
    let mut z = query.clone();
    z.token_id = "z-token".into();
    let mut a = query.clone();
    a.token_id = "a-token".into();
    let collection = TokenCollection::from_tokens(vec![z, a]).unwrap();
    let request = SearchRequest {
        method: SimilarityMethod::Cosine,
        top_k: Some(2),
        threshold: None,
        modality: None,
        source_prefix: None,
    };
    let hits = collection.search(&query, &request).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].token_id, "a-token");
    assert_eq!(hits[1].token_id, "z-token");
    assert_eq!(hits[0].score, 1.0);
    assert_eq!(hits[1].score, 1.0);
}

#[test]
fn top_k_threshold_and_filters_are_composed_deterministically() {
    let query = token("active_sinusoid.json");
    let mut same = query.clone();
    same.token_id = "same".into();
    same.modality = "audio".into();
    same.source_uri = "fixture://wanted/same".into();
    let mut random = token("active_random.json");
    random.token_id = "random".into();
    random.modality = "audio".into();
    random.source_uri = "fixture://wanted/random".into();
    let mut filtered = query.clone();
    filtered.token_id = "filtered".into();
    filtered.modality = "image".into();
    filtered.source_uri = "fixture://other/filtered".into();

    let collection = TokenCollection::from_tokens(vec![filtered, random, same]).unwrap();
    let request = SearchRequest {
        method: SimilarityMethod::PowerCorrelation,
        top_k: Some(2),
        threshold: Some(-1.0),
        modality: Some("audio".into()),
        source_prefix: Some("fixture://wanted/".into()),
    };
    let hits = collection.search(&query, &request).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].token_id, "same");
    assert_eq!(hits[0].rank, 1);
    assert_eq!(hits[1].rank, 2);
    assert!(hits.iter().all(|hit| hit.token_id != "filtered"));
}

#[test]
fn duplicate_token_ids_are_rejected() {
    let token = token("active_random.json");
    let error = TokenCollection::from_tokens(vec![token.clone(), token])
        .unwrap_err()
        .to_string();
    assert!(error.contains("duplicate"));
}
