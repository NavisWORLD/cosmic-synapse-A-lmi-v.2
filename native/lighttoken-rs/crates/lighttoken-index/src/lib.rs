//! Deterministic in-memory LightToken indexing and similarity search.

use lighttoken_core::{LightTokenRecord, SimilarityMethod};
use lighttoken_spectrum::token_similarity;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const DEFAULT_MAX_TOKENS: usize = 100_000;
pub const HARD_MAX_TOKENS: usize = 1_000_000;

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("invalid search request: {0}")]
    InvalidRequest(String),
    #[error("duplicate token id: {0}")]
    DuplicateTokenId(String),
    #[error("token collection exceeds limit of {0}")]
    CollectionLimit(usize),
    #[error("invalid token {0}: {1}")]
    InvalidToken(String, String),
    #[error("similarity error: {0}")]
    Similarity(String),
}

pub type Result<T> = std::result::Result<T, IndexError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    pub method: SimilarityMethod,
    pub top_k: Option<usize>,
    pub threshold: Option<f32>,
    pub modality: Option<String>,
    pub source_prefix: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub token_id: String,
    pub score: f32,
    pub rank: usize,
}

#[derive(Debug, Clone)]
pub struct TokenCollection {
    tokens: Vec<LightTokenRecord>,
}

impl TokenCollection {
    pub fn from_tokens(tokens: Vec<LightTokenRecord>) -> Result<Self> {
        Self::from_tokens_with_limit(tokens, DEFAULT_MAX_TOKENS)
    }

    pub fn from_tokens_with_limit(tokens: Vec<LightTokenRecord>, limit: usize) -> Result<Self> {
        if limit == 0 || limit > HARD_MAX_TOKENS {
            return Err(IndexError::InvalidRequest(format!(
                "collection limit must be between 1 and {HARD_MAX_TOKENS}"
            )));
        }
        if tokens.len() > limit {
            return Err(IndexError::CollectionLimit(limit));
        }

        let mut ids = BTreeSet::new();
        for token in &tokens {
            token
                .validate()
                .map_err(|error| IndexError::InvalidToken(token.token_id.clone(), error.to_string()))?;
            if !ids.insert(token.token_id.clone()) {
                return Err(IndexError::DuplicateTokenId(token.token_id.clone()));
            }
        }
        Ok(Self { tokens })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn tokens(&self) -> &[LightTokenRecord] {
        &self.tokens
    }

    pub fn search(&self, query: &LightTokenRecord, request: &SearchRequest) -> Result<Vec<SearchHit>> {
        query
            .validate()
            .map_err(|error| IndexError::InvalidToken(query.token_id.clone(), error.to_string()))?;
        if request.top_k == Some(0) {
            return Err(IndexError::InvalidRequest("top_k must be at least 1".into()));
        }
        if request.top_k.is_some_and(|value| value > HARD_MAX_TOKENS) {
            return Err(IndexError::InvalidRequest(format!(
                "top_k must not exceed {HARD_MAX_TOKENS}"
            )));
        }
        if request.threshold.is_some_and(|value| !value.is_finite()) {
            return Err(IndexError::InvalidRequest(
                "threshold must be finite".into(),
            ));
        }

        let mut hits = Vec::new();
        for candidate in &self.tokens {
            if request
                .modality
                .as_deref()
                .is_some_and(|modality| candidate.modality != modality)
            {
                continue;
            }
            if request
                .source_prefix
                .as_deref()
                .is_some_and(|prefix| !candidate.source_uri.starts_with(prefix))
            {
                continue;
            }
            let score = token_similarity(query, candidate, request.method)
                .map_err(|error| IndexError::Similarity(error.to_string()))?;
            if request.threshold.is_some_and(|threshold| score < threshold) {
                continue;
            }
            hits.push(SearchHit {
                token_id: candidate.token_id.clone(),
                score,
                rank: 0,
            });
        }

        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        if let Some(top_k) = request.top_k {
            hits.truncate(top_k);
        }
        for (index, hit) in hits.iter_mut().enumerate() {
            hit.rank = index + 1;
        }
        Ok(hits)
    }
}
