import json
from dataclasses import replace
from pathlib import Path

import pytest

from a_lmi.continuity import initialize_workspace, inspect_workspace
from a_lmi.providers import (
    ModelRequest,
    ModelResponse,
    OllamaProvider,
    ProviderError,
    ProviderIdentity,
)
from a_lmi.runtime import PersistentRuntime


class DeterministicProvider:
    def __init__(self, provider_id: str, model_id: str, prefix: str):
        self.identity = ProviderIdentity(
            provider_id=provider_id,
            model_id=model_id,
            revision="test-revision",
            endpoint="test://local",
            capabilities=("text",),
        )
        self.prefix = prefix

    def health(self):
        return {"available": True, "provider": self.identity.to_dict()}

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            text=f"{self.prefix}:{request.prompt}",
            provider=self.identity,
            provenance={"transport": "deterministic-test-provider"},
        )


class FailingProvider(DeterministicProvider):
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise ProviderError("provider unavailable")


def _ledger(path: Path):
    lines = (path / "memory" / "ledger.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_provider_identity_is_explicit_and_serializable():
    identity = ProviderIdentity(
        provider_id="ollama",
        model_id="example:latest",
        revision="sha256:abc",
        endpoint="http://127.0.0.1:11434",
        capabilities=("text",),
        context_limit=8192,
    )

    payload = identity.to_dict()
    assert payload == {
        "provider_id": "ollama",
        "model_id": "example:latest",
        "revision": "sha256:abc",
        "endpoint": "http://127.0.0.1:11434",
        "capabilities": ["text"],
        "context_limit": 8192,
    }


def test_ollama_defaults_to_loopback_and_does_not_claim_health_without_contact():
    provider = OllamaProvider(model_id="qwen2:latest")
    assert provider.identity.provider_id == "ollama"
    assert provider.identity.endpoint == "http://127.0.0.1:11434"
    assert provider.identity.capabilities == ("text",)


def test_runtime_persists_memory_across_provider_swap_without_transferring_authority(tmp_path: Path):
    workspace = tmp_path / "cosmos"
    initialize_workspace(workspace, name="Swap Test")
    authority_path = workspace / "policy" / "authority.json"
    before_authority = authority_path.read_bytes()

    first = DeterministicProvider("test-a", "model-a", "A")
    runtime = PersistentRuntime(workspace, first)
    response_a = runtime.interact("hello")
    assert response_a.text == "A:hello"

    second = DeterministicProvider("test-b", "model-b", "B")
    runtime.set_provider(second)
    response_b = runtime.interact("continue")
    assert response_b.text == "B:continue"

    records = _ledger(workspace)
    assert [record["role"] for record in records] == ["user", "assistant", "user", "assistant"]
    assert records[0]["content"] == "hello"
    assert records[1]["content"] == "A:hello"
    assert records[2]["content"] == "continue"
    assert records[3]["content"] == "B:continue"

    summary = inspect_workspace(workspace)
    assert summary["provider"]["provider_id"] == "test-b"
    assert summary["provider"]["model_id"] == "model-b"
    assert authority_path.read_bytes() == before_authority
    assert summary["authority"]["tool_authority"] == []


def test_runtime_does_not_fabricate_assistant_output_when_provider_fails(tmp_path: Path):
    workspace = tmp_path / "cosmos"
    initialize_workspace(workspace, name="Failure Test")
    runtime = PersistentRuntime(workspace, FailingProvider("fail", "none", "X"))

    with pytest.raises(ProviderError, match="provider unavailable"):
        runtime.interact("can you hear me")

    records = _ledger(workspace)
    assert len(records) == 1
    assert records[0]["role"] == "user"
    assert records[0]["content"] == "can you hear me"
    assert not any(record["role"] == "assistant" for record in records)


def test_runtime_rejects_provider_response_identity_mismatch(tmp_path: Path):
    workspace = tmp_path / "cosmos"
    initialize_workspace(workspace, name="Identity Guard")

    class MismatchProvider(DeterministicProvider):
        def generate(self, request: ModelRequest) -> ModelResponse:
            response = super().generate(request)
            other = replace(response.provider, model_id="different-model")
            return replace(response, provider=other)

    runtime = PersistentRuntime(workspace, MismatchProvider("test", "expected-model", "X"))
    with pytest.raises(ProviderError, match="identity"):
        runtime.interact("hello")

    records = _ledger(workspace)
    assert [record["role"] for record in records] == ["user"]
