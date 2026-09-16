//! Persistent runtime that composes user-owned continuity with replaceable inference.

use almi_continuity::{inspect_workspace, write_canonical_json};
use almi_core::{
    AlmiError, MemoryRecord, ModelRequest, ModelResponse, ProviderProvenance, Result,
    COMPONENT_SCHEMA_VERSION,
};
use almi_memory::append_record;
use almi_provider::ModelProvider;
use chrono::Utc;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct PersistentRuntime {
    workspace: PathBuf,
    provider: Box<dyn ModelProvider>,
}

impl PersistentRuntime {
    pub fn new(workspace: impl AsRef<Path>, provider: Box<dyn ModelProvider>) -> Result<Self> {
        let workspace = workspace.as_ref().to_path_buf();
        inspect_workspace(&workspace)?;
        Ok(Self {
            workspace,
            provider,
        })
    }

    pub fn set_provider(&mut self, provider: Box<dyn ModelProvider>) {
        self.provider = provider;
    }
    pub fn provider(&self) -> &dyn ModelProvider {
        self.provider.as_ref()
    }

    pub fn interact(&mut self, prompt: &str, system: Option<&str>) -> Result<ModelResponse> {
        if prompt.trim().is_empty() {
            return Err(AlmiError::InvalidInput("prompt must not be empty".into()));
        }
        let identity = self.provider.identity().clone();
        let timestamp = Utc::now().to_rfc3339();
        append_record(
            self.workspace.join("memory/ledger.jsonl"),
            &MemoryRecord {
                version: Some(COMPONENT_SCHEMA_VERSION),
                role: "user".into(),
                content: prompt.into(),
                provider: Some(serde_json::to_value(&identity)?),
                provenance: None,
                timestamp: Some(timestamp),
                extra: BTreeMap::new(),
            },
        )?;
        let response = self.provider.generate(&ModelRequest {
            prompt: prompt.into(),
            system: system.map(str::to_owned),
            options: BTreeMap::new(),
        })?;
        if response.provider != identity {
            return Err(AlmiError::Provider(
                "provider response identity does not match the selected provider identity".into(),
            ));
        }
        append_record(
            self.workspace.join("memory/ledger.jsonl"),
            &MemoryRecord {
                version: Some(COMPONENT_SCHEMA_VERSION),
                role: "assistant".into(),
                content: response.text.clone(),
                provider: Some(serde_json::to_value(&response.provider)?),
                provenance: Some(serde_json::to_value(&response.provenance)?),
                timestamp: Some(Utc::now().to_rfc3339()),
                extra: BTreeMap::new(),
            },
        )?;
        let provenance = ProviderProvenance {
            version: COMPONENT_SCHEMA_VERSION,
            provider_id: Some(response.provider.provider_id.clone()),
            model_id: Some(response.provider.model_id.clone()),
            revision: response.provider.revision.clone(),
            endpoint: response.provider.endpoint.clone(),
            capabilities: response.provider.capabilities.clone(),
            context_limit: response.provider.context_limit,
            updated_at: Some(Utc::now().to_rfc3339()),
            last_response_provenance: Some(serde_json::to_value(&response.provenance)?),
        };
        write_canonical_json(self.workspace.join("provenance/provider.json"), &provenance)?;
        Ok(response)
    }
}
