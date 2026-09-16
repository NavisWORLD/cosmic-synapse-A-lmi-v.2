//! Portable user-owned continuity workspaces compatible with the active Python layout.

use almi_core::{canonical_json_bytes, validate_version, AlmiError, ArtifactManifest, AuthorityPolicy, ContinuityWorkspace, CSTStateEnvelope, KnowledgeGraphReference, ProviderProvenance, Result, RoutingState, SystemIdentity, COMPONENT_SCHEMA_VERSION, REQUIRED_WORKSPACE_FILES, WORKSPACE_FORMAT_VERSION};
use almi_memory::verify_ledger;
use almi_state::NativeCstState;
use chrono::Utc;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

pub fn initialize_workspace(path: impl AsRef<Path>, name: &str, seed: i64) -> Result<SystemIdentity> {
    let root = path.as_ref();
    if name.trim().is_empty() {
        return Err(AlmiError::InvalidInput("workspace name must not be empty".into()));
    }
    if root.exists() {
        let mut entries = fs::read_dir(root)?;
        if entries.next().transpose()?.is_some() {
            return Err(AlmiError::Security(format!("workspace destination is not empty: {}", root.display())));
        }
    }
    fs::create_dir_all(root)?;
    let system = SystemIdentity {
        format_version: WORKSPACE_FORMAT_VERSION,
        name: name.to_owned(),
        created_at: Utc::now().to_rfc3339(),
        seed,
    };
    write_json(root.join("system.json"), &system)?;

    let ledger = root.join("memory/ledger.jsonl");
    if let Some(parent) = ledger.parent() { fs::create_dir_all(parent)?; }
    fs::write(&ledger, b"")?;

    NativeCstState::new(seed).save(root.join("state/cst.json"))?;
    write_json(root.join("knowledge/graph.json"), &KnowledgeGraphReference { version: COMPONENT_SCHEMA_VERSION, nodes: vec![], edges: vec![] })?;
    write_json(root.join("artifacts/manifest.json"), &ArtifactManifest { version: COMPONENT_SCHEMA_VERSION, artifacts: vec![] })?;
    write_json(root.join("provenance/provider.json"), &ProviderProvenance::empty())?;
    write_json(root.join("routing/state.json"), &RoutingState::default())?;
    write_json(root.join("policy/authority.json"), &AuthorityPolicy::default())?;
    Ok(system)
}

pub fn inspect_workspace(path: impl AsRef<Path>) -> Result<ContinuityWorkspace> {
    let root = path.as_ref();
    require_workspace(root)?;
    let system: SystemIdentity = read_json(root.join("system.json"))?;
    system.validate()?;
    let state: CSTStateEnvelope = read_json(root.join("state/cst.json"))?;
    state.validate_version()?;
    NativeCstState::from_envelope(state.clone())?;
    let knowledge: KnowledgeGraphReference = read_json(root.join("knowledge/graph.json"))?;
    validate_version("knowledge graph", knowledge.version, COMPONENT_SCHEMA_VERSION)?;
    let artifacts: ArtifactManifest = read_json(root.join("artifacts/manifest.json"))?;
    validate_version("artifact manifest", artifacts.version, COMPONENT_SCHEMA_VERSION)?;
    let provider: ProviderProvenance = read_json(root.join("provenance/provider.json"))?;
    validate_version("provider provenance", provider.version, COMPONENT_SCHEMA_VERSION)?;
    let routing: RoutingState = read_json(root.join("routing/state.json"))?;
    validate_version("routing state", routing.version, COMPONENT_SCHEMA_VERSION)?;
    let authority: AuthorityPolicy = read_json(root.join("policy/authority.json"))?;
    authority.validate()?;
    let memory_records = verify_ledger(root.join("memory/ledger.jsonl"))?.records;
    Ok(ContinuityWorkspace { system, authority, routing, provider, state, memory_records })
}

pub fn validate_workspace(path: impl AsRef<Path>) -> Result<()> {
    inspect_workspace(path).map(|_| ())
}

pub fn required_workspace_files() -> &'static [&'static str] {
    REQUIRED_WORKSPACE_FILES
}

pub fn require_workspace(root: &Path) -> Result<()> {
    if !root.is_dir() {
        return Err(AlmiError::Integrity(format!("workspace does not exist: {}", root.display())));
    }
    for relative in REQUIRED_WORKSPACE_FILES {
        let path = root.join(relative);
        let meta = fs::symlink_metadata(&path).map_err(|_| AlmiError::Integrity(format!("workspace is missing required file: {relative}")))?;
        if meta.file_type().is_symlink() || !meta.is_file() {
            return Err(AlmiError::Security(format!("required workspace path is not a regular file: {relative}")));
        }
    }
    Ok(())
}

pub fn write_canonical_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    write_json(path.as_ref().to_path_buf(), value)
}

pub fn read_typed_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    read_json(path.as_ref().to_path_buf())
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
    fs::write(path, canonical_json_bytes(value)?)?;
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: PathBuf) -> Result<T> {
    let bytes = fs::read(&path).map_err(|e| AlmiError::Integrity(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_slice(&bytes).map_err(|e| AlmiError::Integrity(format!("invalid JSON file {}: {e}", path.display())))
}
