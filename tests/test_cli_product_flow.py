import json
from pathlib import Path

from a_lmi import cli
from a_lmi.providers import ModelResponse, ProviderIdentity


class FakeOllamaProvider:
    def __init__(self, model_id: str, **kwargs):
        self.identity = ProviderIdentity(
            provider_id="ollama",
            model_id=model_id,
            revision="test",
            endpoint=kwargs.get("endpoint", "http://127.0.0.1:11434"),
            capabilities=("text",),
        )

    def health(self):
        return {"available": True, "provider": self.identity.to_dict()}

    def generate(self, request):
        return ModelResponse(
            text=f"echo:{request.prompt}",
            provider=self.identity,
            provenance={"transport": "test"},
        )


def _json_stdout(capsys):
    return json.loads(capsys.readouterr().out)


def test_cli_portable_continuity_round_trip(tmp_path: Path, capsys):
    source = tmp_path / "source"
    bundle = tmp_path / "portable.cosmos"
    restored = tmp_path / "restored"

    assert cli.main(["init", str(source), "--name", "My Cosmos", "--seed", "42", "--json"]) == 0
    initialized = _json_stdout(capsys)
    assert initialized["name"] == "My Cosmos"
    assert initialized["format_version"] == 1

    assert cli.main(["inspect", str(source), "--json"]) == 0
    inspected = _json_stdout(capsys)
    assert inspected["name"] == "My Cosmos"
    assert inspected["memory_records"] == 0

    assert cli.main(["export", str(source), str(bundle), "--json"]) == 0
    exported = _json_stdout(capsys)
    assert exported["valid"] is True
    assert bundle.is_file()

    assert cli.main(["verify", str(bundle), "--json"]) == 0
    verified = _json_stdout(capsys)
    assert verified["valid"] is True
    assert verified["name"] == "My Cosmos"

    assert cli.main(["import", str(bundle), str(restored), "--json"]) == 0
    imported = _json_stdout(capsys)
    assert imported["valid"] is True

    assert cli.main(["inspect", str(restored), "--json"]) == 0
    restored_status = _json_stdout(capsys)
    assert restored_status["name"] == "My Cosmos"
    assert restored_status["authority"]["tool_authority"] == []


def test_cli_providers_is_non_networking_by_default(capsys):
    assert cli.main(["providers", "--json"]) == 0
    payload = _json_stdout(capsys)
    assert payload["network_checked"] is False
    ollama = next(item for item in payload["providers"] if item["provider_id"] == "ollama")
    assert ollama["default_endpoint"] == "http://127.0.0.1:11434"
    assert ollama["status"] == "UNCONFIGURED"


def test_cli_run_uses_selected_provider_and_persists_interaction(tmp_path: Path, capsys, monkeypatch):
    workspace = tmp_path / "cosmos"
    assert cli.main(["init", str(workspace), "--name", "Runtime", "--json"]) == 0
    _json_stdout(capsys)

    monkeypatch.setattr(cli, "OllamaProvider", FakeOllamaProvider)
    assert (
        cli.main(
            [
                "run",
                str(workspace),
                "--provider",
                "ollama",
                "--model",
                "local-test",
                "--prompt",
                "hello",
                "--json",
            ]
        )
        == 0
    )
    response = _json_stdout(capsys)
    assert response["text"] == "echo:hello"
    assert response["provider"]["model_id"] == "local-test"

    assert cli.main(["inspect", str(workspace), "--json"]) == 0
    state = _json_stdout(capsys)
    assert state["memory_records"] == 2
    assert state["provider"]["model_id"] == "local-test"
    assert state["authority"]["tool_authority"] == []


def test_cli_run_parser_requires_explicit_model_and_prompt():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "/tmp/cosmos",
            "--provider",
            "ollama",
            "--model",
            "qwen2:latest",
            "--prompt",
            "hello",
        ]
    )
    assert args.provider == "ollama"
    assert args.model == "qwen2:latest"
    assert args.prompt == "hello"
