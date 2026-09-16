use almi_continuity::initialize_workspace;
use almi_core::{ModelRequest, ModelResponse, ProviderIdentity, Result};
use almi_provider::ModelProvider;
use almi_runtime::PersistentRuntime;
use std::collections::BTreeMap;

struct DeterministicProvider { identity: ProviderIdentity, prefix: &'static str }
impl ModelProvider for DeterministicProvider {
    fn identity(&self) -> &ProviderIdentity { &self.identity }
    fn health(&self) -> Result<almi_core::ProviderHealth> { Ok(almi_core::ProviderHealth { available: true, reachable: Some(true), provider: self.identity.clone(), error: None }) }
    fn generate(&self, request: &ModelRequest) -> Result<ModelResponse> { Ok(ModelResponse { text: format!("{}:{}", self.prefix, request.prompt), provider: self.identity.clone(), provenance: BTreeMap::new() }) }
}
fn p(id: &str, model: &str, prefix: &'static str) -> DeterministicProvider { DeterministicProvider { identity: ProviderIdentity { provider_id: id.into(), model_id: model.into(), revision: Some("test".into()), endpoint: Some("test://local".into()), capabilities: vec!["text".into()], context_limit: None }, prefix } }

#[test]
fn provider_swap_keeps_continuity_and_exact_authority_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("cosmos");
    initialize_workspace(&root, "Swap", 0).unwrap();
    let authority_before = std::fs::read(root.join("policy/authority.json")).unwrap();
    let mut runtime = PersistentRuntime::new(&root, Box::new(p("a", "model-a", "A"))).unwrap();
    assert_eq!(runtime.interact("hello", None).unwrap().text, "A:hello");
    runtime.set_provider(Box::new(p("b", "model-b", "B")));
    assert_eq!(runtime.interact("continue", None).unwrap().text, "B:continue");
    assert_eq!(std::fs::read(root.join("policy/authority.json")).unwrap(), authority_before);
    let ledger = std::fs::read_to_string(root.join("memory/ledger.jsonl")).unwrap();
    assert_eq!(ledger.lines().count(), 4);
}
