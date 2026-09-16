//! Versioned wire contracts shared by the A-LMI native crates.
//!
//! These types model the active Python product surface. The model/provider is
//! deliberately not the owner of memory, state, routing, policy, or authority.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use thiserror::Error;

pub const WORKSPACE_FORMAT_VERSION: u32 = 1;
pub const BUNDLE_FORMAT_VERSION: u32 = 1;
pub const CST_STATE_VERSION: u32 = 1;
pub const COMPONENT_SCHEMA_VERSION: u32 = 1;
pub const ABI_VERSION: u32 = 1;
pub const BUNDLE_MANIFEST: &str = "bundle-manifest.json";
pub const MAX_BUNDLE_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_BUNDLE_FILES: usize = 10_000;

pub const REQUIRED_WORKSPACE_FILES: &[&str] = &[
    "system.json",
    "memory/ledger.jsonl",
    "state/cst.json",
    "knowledge/graph.json",
    "artifacts/manifest.json",
    "provenance/provider.json",
    "routing/state.json",
    "policy/authority.json",
];

#[derive(Debug, Error)]
pub enum AlmiError {
    #[error("integrity error: {0}")]
    Integrity(String),
    #[error("security error: {0}")]
    Security(String),
    #[error("unsupported version: {0}")]
    UnsupportedVersion(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, AlmiError>;

pub fn validate_version(kind: &str, actual: u32, expected: u32) -> Result<()> {
    if actual != expected {
        return Err(AlmiError::UnsupportedVersion(format!(
            "{kind} version {actual}; expected {expected}"
        )));
    }
    Ok(())
}

/// Recursively sort JSON object keys so serialization matches Python's
/// `sort_keys=True,separators=(",", ":")` behavior.
pub fn normalize_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut ordered = BTreeMap::new();
            for (key, value) in object {
                ordered.insert(key, normalize_json(value));
            }
            let mut out = Map::new();
            for (key, value) in ordered {
                out.insert(key, value);
            }
            Value::Object(out)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(normalize_json).collect()),
        other => other,
    }
}

pub fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let value = normalize_json(serde_json::to_value(value)?);
    let mut bytes = serde_json::to_vec(&value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemIdentity {
    pub format_version: u32,
    pub name: String,
    pub created_at: String,
    pub seed: i64,
}

impl SystemIdentity {
    pub fn validate(&self) -> Result<()> {
        validate_version(
            "workspace format",
            self.format_version,
            WORKSPACE_FORMAT_VERSION,
        )?;
        if self.name.trim().is_empty() {
            return Err(AlmiError::InvalidInput(
                "system name must not be empty".into(),
            ));
        }
        if self.created_at.trim().is_empty() {
            return Err(AlmiError::Integrity("created_at must not be empty".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Native records write version=1. Existing Python ledgers that predate the
    /// field are treated as the active v1 compatibility form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl MemoryRecord {
    pub fn validate(&self) -> Result<()> {
        if let Some(version) = self.version {
            validate_version("memory record schema", version, COMPONENT_SCHEMA_VERSION)?;
        }
        if self.role.trim().is_empty() {
            return Err(AlmiError::Integrity("memory role must not be empty".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CstParameters {
    pub k: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub sync_strength: f64,
    pub audio_gain: f64,
    pub natural_frequency: f64,
}

impl Default for CstParameters {
    fn default() -> Self {
        Self {
            k: 0.5,
            gamma: 0.2,
            alpha: 0.3,
            sync_strength: 0.1,
            audio_gain: 0.25,
            natural_frequency: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CstSnapshot {
    pub x12: f64,
    pub m12: f64,
    pub omega: f64,
    pub phase: f64,
    pub energy: f64,
    pub entropy: f64,
    pub step: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CSTStateEnvelope {
    pub version: u32,
    pub seed: i64,
    pub params: CstParameters,
    pub state: CstSnapshot,
}

impl CSTStateEnvelope {
    pub fn validate_version(&self) -> Result<()> {
        validate_version("CST state", self.version, CST_STATE_VERSION)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeGraphReference {
    pub version: u32,
    #[serde(default)]
    pub nodes: Vec<Value>,
    #[serde(default)]
    pub edges: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub version: u32,
    #[serde(default)]
    pub artifacts: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderIdentity {
    pub provider_id: String,
    pub model_id: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default = "default_text_capability")]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub context_limit: Option<u64>,
}

fn default_text_capability() -> Vec<String> {
    vec!["text".into()]
}

impl ProviderIdentity {
    pub fn validate(&self) -> Result<()> {
        if self.provider_id.trim().is_empty() || self.model_id.trim().is_empty() {
            return Err(AlmiError::InvalidInput(
                "provider_id and model_id must not be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderProvenance {
    pub version: u32,
    #[serde(default)]
    pub provider_id: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub context_limit: Option<u64>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub last_response_provenance: Option<Value>,
}

impl ProviderProvenance {
    pub fn empty() -> Self {
        Self {
            version: COMPONENT_SCHEMA_VERSION,
            provider_id: None,
            model_id: None,
            revision: None,
            endpoint: None,
            capabilities: Vec::new(),
            context_limit: None,
            updated_at: None,
            last_response_provenance: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingState {
    pub version: u32,
    #[serde(default)]
    pub routes: BTreeMap<String, Value>,
}

impl Default for RoutingState {
    fn default() -> Self {
        Self {
            version: COMPONENT_SCHEMA_VERSION,
            routes: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthorityPolicy {
    pub version: u32,
    #[serde(default)]
    pub tool_authority: Vec<String>,
    #[serde(default)]
    pub network_authority: Vec<String>,
    #[serde(default)]
    pub filesystem_authority: Vec<String>,
    #[serde(default)]
    pub cloud_authority: Vec<String>,
    #[serde(default)]
    pub deployment_authority: Vec<String>,
    #[serde(default)]
    pub actuator_authority: Vec<String>,
}

impl Default for AuthorityPolicy {
    fn default() -> Self {
        Self {
            version: COMPONENT_SCHEMA_VERSION,
            tool_authority: Vec::new(),
            network_authority: Vec::new(),
            filesystem_authority: Vec::new(),
            cloud_authority: Vec::new(),
            deployment_authority: Vec::new(),
            actuator_authority: Vec::new(),
        }
    }
}

impl AuthorityPolicy {
    pub fn validate(&self) -> Result<()> {
        validate_version("authority policy", self.version, COMPONENT_SCHEMA_VERSION)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub options: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    pub text: String,
    pub provider: ProviderIdentity,
    #[serde(default)]
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderHealth {
    pub available: bool,
    #[serde(default)]
    pub reachable: Option<bool>,
    pub provider: ProviderIdentity,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub version: u32,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuityManifest {
    pub bundle_format_version: u32,
    pub workspace_format_version: u32,
    pub workspace_name: String,
    pub files: Vec<ManifestEntry>,
}

impl ContinuityManifest {
    pub fn validate_versions(&self) -> Result<()> {
        validate_version(
            "bundle format",
            self.bundle_format_version,
            BUNDLE_FORMAT_VERSION,
        )?;
        validate_version(
            "workspace format in bundle",
            self.workspace_format_version,
            WORKSPACE_FORMAT_VERSION,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CosmosBundleMetadata {
    pub valid: bool,
    pub path: String,
    #[serde(default)]
    pub name: Option<String>,
    pub file_count: usize,
    pub total_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContinuityWorkspace {
    pub system: SystemIdentity,
    pub authority: AuthorityPolicy,
    pub routing: RoutingState,
    pub provider: ProviderProvenance,
    pub state: CSTStateEnvelope,
    pub memory_records: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_nested_objects_and_adds_newline() {
        let value = serde_json::json!({"z": 1, "a": {"y": 2, "b": 3}});
        assert_eq!(
            canonical_json_bytes(&value).unwrap(),
            b"{\"a\":{\"b\":3,\"y\":2},\"z\":1}\n"
        );
    }

    #[test]
    fn authority_default_is_deny_by_default() {
        let policy = AuthorityPolicy::default();
        assert!(policy.tool_authority.is_empty());
        assert!(policy.network_authority.is_empty());
        assert!(policy.filesystem_authority.is_empty());
        assert!(policy.cloud_authority.is_empty());
        assert!(policy.deployment_authority.is_empty());
        assert!(policy.actuator_authority.is_empty());
    }
}
