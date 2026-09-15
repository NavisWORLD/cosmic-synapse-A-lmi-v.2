"""Persistent runtime that composes user-owned continuity with a model provider."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .continuity import append_memory_record, inspect_workspace
from .providers import ModelProvider, ModelRequest, ModelResponse, ProviderError


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")


class PersistentRuntime:
    """Run model inference without making the provider the owner of the system.

    The continuity workspace owns memory and authority. Changing providers only
    changes which inference component receives subsequent requests.
    """

    def __init__(self, workspace: str | Path, provider: ModelProvider) -> None:
        self.workspace = Path(workspace)
        inspect_workspace(self.workspace)
        self.provider = provider

    def set_provider(self, provider: ModelProvider) -> None:
        """Replace only the inference provider; persistent policy remains untouched."""
        self.provider = provider

    def _write_provider_provenance(self, response: ModelResponse) -> None:
        identity = response.provider.to_dict()
        record = {
            "version": 1,
            **identity,
            "updated_at": _utc_now(),
            "last_response_provenance": dict(response.provenance),
        }
        _write_json(self.workspace / "provenance" / "provider.json", record)

    def interact(self, prompt: str, *, system: str | None = None) -> ModelResponse:
        """Persist a user request, invoke the provider, then persist verified output."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")

        identity = self.provider.identity
        append_memory_record(
            self.workspace,
            {
                "role": "user",
                "content": prompt,
                "provider": identity.to_dict(),
                "timestamp": _utc_now(),
            },
        )

        response = self.provider.generate(ModelRequest(prompt=prompt, system=system))
        if response.provider != identity:
            raise ProviderError("provider response identity does not match the selected provider identity")
        if not isinstance(response.text, str):
            raise ProviderError("provider response text is invalid")

        append_memory_record(
            self.workspace,
            {
                "role": "assistant",
                "content": response.text,
                "provider": response.provider.to_dict(),
                "provenance": dict(response.provenance),
                "timestamp": _utc_now(),
            },
        )
        self._write_provider_provenance(response)
        return response
