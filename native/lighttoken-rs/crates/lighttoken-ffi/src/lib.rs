//! Stable LightToken C ABI and JNI boundary.
//!
//! C and JNI callers share the same Rust engine functions. Exported foreign
//! functions contain panics and expose deterministic ABI-v1 status/error data.

pub mod c_api;
pub mod jni_api;

use lighttoken_core::{from_json_bytes, LightTokenError, LightTokenRecord, SimilarityMethod};
use lighttoken_index::{SearchRequest, TokenCollection};
use lighttoken_spectrum::{backend_diagnostics, token_similarity, BackendKind};
use serde_json::{json, Value};
use thiserror::Error;

pub const LIGHTTOKEN_ABI_VERSION: u32 = 1;

#[derive(Debug, Default)]
#[repr(C)]
pub struct LightTokenContext {
    _private: u8,
}

#[derive(Debug, Error)]
pub(crate) enum EngineError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("invalid token: {0}")]
    InvalidToken(String),
    #[error("unsupported version: {0}")]
    UnsupportedVersion(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("JSON error: {0}")]
    Json(String),
}

fn parse_token(text: &str) -> Result<LightTokenRecord, EngineError> {
    from_json_bytes(text.as_bytes()).map_err(|error| match error {
        LightTokenError::Unsupported(message) => EngineError::UnsupportedVersion(message),
        LightTokenError::Invalid(message) => EngineError::InvalidToken(message),
        LightTokenError::Json(error) => EngineError::Json(error.to_string()),
    })
}

fn parse_method(value: &str) -> Result<SimilarityMethod, EngineError> {
    match value {
        "power_correlation" | "correlation" => Ok(SimilarityMethod::PowerCorrelation),
        "cosine" => Ok(SimilarityMethod::Cosine),
        "euclidean" => Ok(SimilarityMethod::Euclidean),
        other => Err(EngineError::InvalidArgument(format!(
            "unsupported similarity method {other:?}"
        ))),
    }
}

fn method_name(method: SimilarityMethod) -> &'static str {
    match method {
        SimilarityMethod::PowerCorrelation => "power_correlation",
        SimilarityMethod::Cosine => "cosine",
        SimilarityMethod::Euclidean => "euclidean",
    }
}

pub(crate) fn validate_json_text(text: &str) -> Result<String, EngineError> {
    let token = parse_token(text)?;
    let payload = json!({
        "valid": true,
        "token_id": token.token_id,
        "embedding_dimension": token.joint_embedding.as_ref().map(Vec::len).unwrap_or(0),
        "spectral_dimension": token.spectral_dimension().unwrap_or(0),
    });
    serde_json::to_string(&payload).map_err(|error| EngineError::Json(error.to_string()))
}

pub(crate) fn compare_json_text(
    left_json: &str,
    right_json: &str,
    method: &str,
) -> Result<String, EngineError> {
    let left = parse_token(left_json)?;
    let right = parse_token(right_json)?;
    let method = parse_method(method)?;
    let score = token_similarity(&left, &right, method)
        .map_err(|error| EngineError::Backend(error.to_string()))?;
    let payload = json!({
        "a_token_id": left.token_id,
        "b_token_id": right.token_id,
        "method": method_name(method),
        "score": score,
        "backend": "rust",
    });
    serde_json::to_string(&payload).map_err(|error| EngineError::Json(error.to_string()))
}

pub(crate) fn search_json_text(
    query_json: &str,
    collection_json: &str,
    request_json: &str,
) -> Result<String, EngineError> {
    let query = parse_token(query_json)?;
    let request: SearchRequest =
        serde_json::from_str(request_json).map_err(|error| EngineError::Json(error.to_string()))?;
    let value: Value = serde_json::from_str(collection_json)
        .map_err(|error| EngineError::Json(error.to_string()))?;
    let array = value.as_array().ok_or_else(|| {
        EngineError::InvalidArgument(
            "collection JSON must be an array of LightToken objects".into(),
        )
    })?;
    let mut tokens = Vec::with_capacity(array.len());
    for item in array {
        let encoded =
            serde_json::to_vec(item).map_err(|error| EngineError::Json(error.to_string()))?;
        tokens.push(from_json_bytes(&encoded).map_err(|error| match error {
            LightTokenError::Unsupported(message) => EngineError::UnsupportedVersion(message),
            LightTokenError::Invalid(message) => EngineError::InvalidToken(message),
            LightTokenError::Json(error) => EngineError::Json(error.to_string()),
        })?);
    }
    let collection = TokenCollection::from_tokens(tokens)
        .map_err(|error| EngineError::InvalidArgument(error.to_string()))?;
    let hits = collection
        .search(&query, &request)
        .map_err(|error| EngineError::Backend(error.to_string()))?;
    let payload = json!({
        "query_token_id": query.token_id,
        "method": method_name(request.method),
        "backend": "rust",
        "hits": hits,
    });
    serde_json::to_string(&payload).map_err(|error| EngineError::Json(error.to_string()))
}

pub(crate) fn backend_json_text() -> Result<String, EngineError> {
    let diagnostics = backend_diagnostics();
    let active = match diagnostics.active {
        BackendKind::Rust => "rust",
        BackendKind::Cpp => "cpp",
    };
    serde_json::to_string(&json!({
        "active": active,
        "rust": {"available": true},
        "cpp": {"available": diagnostics.cpp_available, "detail": diagnostics.detail},
    }))
    .map_err(|error| EngineError::Json(error.to_string()))
}
