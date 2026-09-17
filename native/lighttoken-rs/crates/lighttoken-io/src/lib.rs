//! Safe LightToken JSON/JSONL loading and verified A-LMI read adapters.

use almi_continuity::{read_typed_json, require_workspace};
use almi_core::{AlmiError, ArtifactManifest};
use almi_cosmos::{import_bundle, verify_bundle};
use lighttoken_core::{from_json_bytes, LightTokenError, LightTokenRecord};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

pub const MAX_TOKEN_JSON_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_COLLECTION_BYTES: u64 = 512 * 1024 * 1024;
pub const DEFAULT_MAX_COLLECTION_TOKENS: usize = 100_000;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("LightToken error: {0}")]
    Token(#[from] LightTokenError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("A-LMI error: {0}")]
    Almi(#[from] AlmiError),
    #[error("security error: {0}")]
    Security(String),
    #[error("size/count limit: {0}")]
    Limit(String),
}

pub type Result<T> = std::result::Result<T, IoError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum LightTokenSource {
    ReferenceOnly {
        token_id: String,
        source_path: String,
        raw_data_ref: String,
    },
    Resolved {
        token_id: String,
        source_path: String,
        raw_path: String,
        sha256: String,
    },
}

fn require_regular_file(path: &Path, max_bytes: u64) -> Result<u64> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(IoError::Security(format!(
            "path is not a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() > max_bytes {
        return Err(IoError::Limit(format!(
            "{} exceeds byte limit of {max_bytes}",
            path.display()
        )));
    }
    Ok(metadata.len())
}

pub fn load_token_file(path: impl AsRef<Path>) -> Result<LightTokenRecord> {
    let path = path.as_ref();
    require_regular_file(path, MAX_TOKEN_JSON_BYTES)?;
    let bytes = fs::read(path)?;
    Ok(from_json_bytes(&bytes)?)
}

pub fn load_collection(path: impl AsRef<Path>) -> Result<Vec<LightTokenRecord>> {
    let path = path.as_ref();
    require_regular_file(path, MAX_COLLECTION_BYTES)?;
    if path.extension().and_then(|value| value.to_str()) == Some("jsonl") {
        return load_jsonl(path);
    }

    let bytes = fs::read(path)?;
    if let Ok(token) = from_json_bytes(&bytes) {
        return Ok(vec![token]);
    }
    let value: Value = serde_json::from_slice(&bytes)?;
    let array = value.as_array().ok_or_else(|| {
        IoError::Json(serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "collection JSON must be a LightToken object or array",
        )))
    })?;
    if array.len() > DEFAULT_MAX_COLLECTION_TOKENS {
        return Err(IoError::Limit(format!(
            "collection exceeds token limit of {DEFAULT_MAX_COLLECTION_TOKENS}"
        )));
    }
    array
        .iter()
        .map(|item| {
            let encoded = serde_json::to_vec(item)?;
            if encoded.len() as u64 > MAX_TOKEN_JSON_BYTES {
                return Err(IoError::Limit(
                    "collection member exceeds single-token byte limit".into(),
                ));
            }
            Ok(from_json_bytes(&encoded)?)
        })
        .collect()
}

fn load_jsonl(path: &Path) -> Result<Vec<LightTokenRecord>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut tokens = Vec::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        let bytes_read = reader.read_until(b'\n', &mut line)?;
        if bytes_read == 0 {
            break;
        }
        if line.len() as u64 > MAX_TOKEN_JSON_BYTES {
            return Err(IoError::Limit(
                "JSONL member exceeds single-token byte limit".into(),
            ));
        }
        if line.iter().all(|byte| byte.is_ascii_whitespace()) {
            continue;
        }
        tokens.push(from_json_bytes(&line)?);
        if tokens.len() > DEFAULT_MAX_COLLECTION_TOKENS {
            return Err(IoError::Limit(format!(
                "collection exceeds token limit of {DEFAULT_MAX_COLLECTION_TOKENS}"
            )));
        }
    }
    Ok(tokens)
}

pub fn discover_almi_workspace(path: impl AsRef<Path>) -> Result<Vec<LightTokenSource>> {
    let root = path.as_ref();
    require_workspace(root)?;
    let manifest: ArtifactManifest = read_typed_json(root.join("artifacts/manifest.json"))?;
    let mut sources = Vec::new();
    for artifact in &manifest.artifacts {
        let Some(relative) = lighttoken_artifact_path(artifact) else {
            continue;
        };
        let relative_path = safe_relative_path(relative)?;
        let token_path = root.join(&relative_path);
        if !token_path.exists() {
            sources.push(LightTokenSource::ReferenceOnly {
                token_id: artifact
                    .get("token_id")
                    .and_then(Value::as_str)
                    .unwrap_or(relative)
                    .to_owned(),
                source_path: token_path.display().to_string(),
                raw_data_ref: relative.to_owned(),
            });
            continue;
        }
        let token = load_token_file(&token_path)?;
        let raw_path = resolve_raw_reference(root, &token.raw_data_ref);
        if let Some(raw_path) = raw_path.filter(|candidate| candidate.is_file()) {
            let raw_meta = fs::symlink_metadata(&raw_path)?;
            if raw_meta.file_type().is_symlink() {
                return Err(IoError::Security(format!(
                    "raw artifact reference resolves through a symlink: {}",
                    raw_path.display()
                )));
            }
            let bytes = fs::read(&raw_path)?;
            sources.push(LightTokenSource::Resolved {
                token_id: token.token_id,
                source_path: token_path.display().to_string(),
                raw_path: raw_path.display().to_string(),
                sha256: hex_digest(&bytes),
            });
        } else {
            sources.push(LightTokenSource::ReferenceOnly {
                token_id: token.token_id,
                source_path: token_path.display().to_string(),
                raw_data_ref: token.raw_data_ref,
            });
        }
    }
    Ok(sources)
}

pub fn import_verified_cosmos(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> Result<Vec<LightTokenSource>> {
    let source = source.as_ref();
    let destination = destination.as_ref();
    verify_bundle(source)?;
    import_bundle(source, destination)?;
    discover_almi_workspace(destination)
}

fn lighttoken_artifact_path(value: &Value) -> Option<&str> {
    let object = value.as_object()?;
    let kind = object
        .get("kind")
        .or_else(|| object.get("type"))
        .or_else(|| object.get("media_type"))
        .and_then(Value::as_str)?;
    let is_lighttoken = matches!(
        kind,
        "lighttoken" | "LightToken" | "application/vnd.almi.lighttoken+json"
    );
    if !is_lighttoken {
        return None;
    }
    object
        .get("path")
        .or_else(|| object.get("relative_path"))
        .or_else(|| object.get("lighttoken_path"))
        .and_then(Value::as_str)
}

fn safe_relative_path(value: &str) -> Result<PathBuf> {
    if value.is_empty() || value.contains('\\') {
        return Err(IoError::Security(format!(
            "unsafe workspace-relative path: {value:?}"
        )));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir
                    | Component::CurDir
                    | Component::RootDir
                    | Component::Prefix(_)
            )
        })
    {
        return Err(IoError::Security(format!(
            "unsafe workspace-relative path: {value:?}"
        )));
    }
    Ok(path.to_path_buf())
}

fn resolve_raw_reference(root: &Path, raw_data_ref: &str) -> Option<PathBuf> {
    let relative = raw_data_ref
        .strip_prefix("workspace://")
        .unwrap_or(raw_data_ref);
    safe_relative_path(relative)
        .ok()
        .map(|path| root.join(path))
}

fn hex_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
