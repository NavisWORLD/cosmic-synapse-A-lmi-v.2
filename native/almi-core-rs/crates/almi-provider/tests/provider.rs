use almi_provider::{OllamaProvider, DEFAULT_OLLAMA_ENDPOINT};

#[test]
fn ollama_defaults_loopback_and_rejects_credentials() {
    let provider = OllamaProvider::new("qwen2:latest", None).unwrap();
    assert_eq!(provider.identity().endpoint.as_deref(), Some(DEFAULT_OLLAMA_ENDPOINT));
    assert!(OllamaProvider::new("qwen2:latest", Some("http://user:secret@127.0.0.1:11434")).is_err());
}
