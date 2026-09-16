//! Language-independent model-provider boundary. Providers perform inference only;
//! they do not own memory, state, routing, policy, tools, or authority.

use almi_core::{AlmiError, ModelRequest, ModelResponse, ProviderHealth, ProviderIdentity, Result};
use reqwest::blocking::Client;
use serde_json::Value;
use std::collections::BTreeMap;
use std::thread;
use std::time::Duration;
use url::Url;

pub const DEFAULT_OLLAMA_ENDPOINT: &str = "http://127.0.0.1:11434";

pub trait ModelProvider: Send + Sync {
    fn identity(&self) -> &ProviderIdentity;
    fn health(&self) -> Result<ProviderHealth>;
    fn generate(&self, request: &ModelRequest) -> Result<ModelResponse>;
}

pub struct OllamaProvider {
    identity: ProviderIdentity,
    client: Client,
    timeout: Duration,
    retries: u32,
}

impl OllamaProvider {
    pub fn new(model_id: &str, endpoint: Option<&str>) -> Result<Self> {
        Self::with_policy(model_id, endpoint.unwrap_or(DEFAULT_OLLAMA_ENDPOINT), None, None, Duration::from_secs(30), 0)
    }

    pub fn with_policy(model_id: &str, endpoint: &str, revision: Option<String>, context_limit: Option<u64>, timeout: Duration, retries: u32) -> Result<Self> {
        if model_id.trim().is_empty() { return Err(AlmiError::InvalidInput("model_id must not be empty".into())); }
        if timeout.is_zero() { return Err(AlmiError::InvalidInput("timeout must be positive".into())); }
        let endpoint = endpoint.trim_end_matches('/');
        let parsed = Url::parse(endpoint).map_err(|_| AlmiError::InvalidInput("Ollama endpoint must be a valid URL".into()))?;
        if !matches!(parsed.scheme(), "http" | "https") { return Err(AlmiError::InvalidInput("Ollama endpoint must use http:// or https://".into())); }
        if parsed.host_str().is_none() { return Err(AlmiError::InvalidInput("Ollama endpoint must include a hostname".into())); }
        if !parsed.username().is_empty() || parsed.password().is_some() { return Err(AlmiError::InvalidInput("Ollama endpoint must not embed credentials".into())); }
        if parsed.query().is_some() || parsed.fragment().is_some() { return Err(AlmiError::InvalidInput("Ollama endpoint must not include a query or fragment".into())); }
        let client = Client::builder().timeout(timeout).build().map_err(|e| AlmiError::Provider(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            identity: ProviderIdentity { provider_id: "ollama".into(), model_id: model_id.into(), revision, endpoint: Some(endpoint.into()), capabilities: vec!["text".into()], context_limit },
            client, timeout, retries,
        })
    }

    pub fn identity(&self) -> &ProviderIdentity { &self.identity }
    pub fn timeout(&self) -> Duration { self.timeout }

    fn request_json(&self, method: reqwest::Method, path: &str, payload: Option<&Value>) -> Result<Value> {
        let base = self.identity.endpoint.as_deref().ok_or_else(|| AlmiError::Provider("provider endpoint missing".into()))?;
        let url = format!("{base}{path}");
        let mut last_error = None;
        for attempt in 0..=self.retries {
            let mut request = self.client.request(method.clone(), &url).header("Accept", "application/json");
            if let Some(value) = payload { request = request.json(value); }
            match request.send() {
                Ok(response) => match response.error_for_status() {
                    Ok(response) => return response.json::<Value>().map_err(|_| AlmiError::Provider("provider returned invalid JSON".into())),
                    Err(error) => last_error = Some(format!("HTTP provider error: {}", error.status().map(|s| s.as_u16()).unwrap_or(0))),
                },
                Err(error) => last_error = Some(if error.is_timeout() { "provider request timed out".into() } else { "provider request failed".into() }),
            }
            if attempt < self.retries { thread::sleep(Duration::from_millis((250u64.saturating_mul(1u64 << attempt.min(2))).min(1000))); }
        }
        Err(AlmiError::Provider(last_error.unwrap_or_else(|| "provider request failed".into())))
    }
}

impl ModelProvider for OllamaProvider {
    fn identity(&self) -> &ProviderIdentity { &self.identity }

    fn health(&self) -> Result<ProviderHealth> {
        match self.request_json(reqwest::Method::GET, "/api/tags", None) {
            Ok(value) => {
                let installed = value.get("models").and_then(Value::as_array).map(|models| models.iter().filter_map(|m| m.get("name").and_then(Value::as_str)).collect::<Vec<_>>()).unwrap_or_default();
                Ok(ProviderHealth { available: installed.iter().any(|name| *name == self.identity.model_id), reachable: Some(true), provider: self.identity.clone(), error: None })
            }
            Err(error) => Ok(ProviderHealth { available: false, reachable: Some(false), provider: self.identity.clone(), error: Some(error.to_string()) }),
        }
    }

    fn generate(&self, request: &ModelRequest) -> Result<ModelResponse> {
        if request.prompt.trim().is_empty() { return Err(AlmiError::Provider("model request prompt must not be empty".into())); }
        let mut payload = serde_json::json!({"model": self.identity.model_id, "prompt": request.prompt, "stream": false});
        if let Value::Object(object) = &mut payload {
            if let Some(system) = &request.system { object.insert("system".into(), Value::String(system.clone())); }
            if !request.options.is_empty() { object.insert("options".into(), serde_json::to_value(&request.options)?); }
        }
        let result = self.request_json(reqwest::Method::POST, "/api/generate", Some(&payload))?;
        let text = result.get("response").and_then(Value::as_str).ok_or_else(|| AlmiError::Provider("Ollama response does not contain text in 'response'".into()))?;
        let mut provenance = BTreeMap::new();
        provenance.insert("transport".into(), Value::String("ollama-http".into()));
        for key in ["done","done_reason","created_at","total_duration","load_duration","prompt_eval_count","eval_count"] {
            if let Some(value) = result.get(key) { provenance.insert(key.into(), value.clone()); }
        }
        Ok(ModelResponse { text: text.into(), provider: self.identity.clone(), provenance })
    }
}
