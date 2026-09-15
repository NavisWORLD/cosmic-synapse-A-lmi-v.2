"""Replaceable model-provider contracts for the dependency-light runtime.

A provider supplies inference only. It does not own persistent memory, CST state,
routing, policy, tools, or authority. Provider identity is explicit and is written
into response provenance by the surrounding runtime.
"""

from __future__ import annotations

import http.client
import json
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit


DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"


class ProviderError(RuntimeError):
    """Raised when a provider cannot return a trustworthy model response."""


@dataclass(frozen=True)
class ProviderIdentity:
    provider_id: str
    model_id: str
    revision: str | None = None
    endpoint: str | None = None
    capabilities: tuple[str, ...] = ("text",)
    context_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "endpoint": self.endpoint,
            "capabilities": list(self.capabilities),
            "context_limit": self.context_limit,
        }


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    system: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider: ProviderIdentity
    provenance: Mapping[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    identity: ProviderIdentity

    def health(self) -> Mapping[str, Any]: ...

    def generate(self, request: ModelRequest) -> ModelResponse: ...


class OllamaProvider:
    """Small stdlib client for a user-selected Ollama model.

    The default endpoint is loopback. Constructing this provider performs no
    network access. ``health`` and ``generate`` report only what an actual HTTP
    request establishes. Endpoint credentials are rejected so they cannot leak
    through provider provenance or error reporting.
    """

    def __init__(
        self,
        model_id: str,
        *,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        revision: str | None = None,
        context_limit: int | None = None,
        timeout: float = 30.0,
        retries: int = 0,
    ) -> None:
        endpoint = endpoint.rstrip("/")
        parsed_endpoint = urlsplit(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"}:
            raise ValueError("Ollama endpoint must use http:// or https://")
        if not parsed_endpoint.hostname:
            raise ValueError("Ollama endpoint must include a hostname")
        if parsed_endpoint.username is not None or parsed_endpoint.password is not None:
            raise ValueError("Ollama endpoint must not embed credentials")
        if parsed_endpoint.query or parsed_endpoint.fragment:
            raise ValueError("Ollama endpoint must not include a query or fragment")
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.identity = ProviderIdentity(
            provider_id="ollama",
            model_id=model_id,
            revision=revision,
            endpoint=endpoint,
            capabilities=("text",),
            context_limit=context_limit,
        )
        self.timeout = float(timeout)
        self.retries = int(retries)

    def _request_json(
        self,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        method: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self.identity.endpoint}{path}"
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ProviderError("Ollama request resolved to an invalid endpoint")

        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request_method = method or ("POST" if data is not None else "GET")
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
            try:
                connection_type = (
                    http.client.HTTPSConnection
                    if parsed.scheme == "https"
                    else http.client.HTTPConnection
                )
                connection = connection_type(
                    parsed.hostname,
                    port=parsed.port,
                    timeout=self.timeout,
                )
                connection.request(request_method, target, body=data, headers=headers)
                response = connection.getresponse()
                body = response.read()
                if response.status < 200 or response.status >= 300:
                    raise OSError(
                        f"HTTP {response.status} {response.reason or 'provider error'}"
                    )
                decoded = json.loads(body.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ProviderError("provider returned a non-object JSON response")
                return decoded
            except ProviderError:
                raise
            except (
                http.client.HTTPException,
                socket.timeout,
                TimeoutError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                OSError,
            ) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
            finally:
                if connection is not None:
                    connection.close()
        raise ProviderError(f"Ollama request failed for {url}: {last_error}") from last_error

    def health(self) -> dict[str, Any]:
        try:
            payload = self._request_json("/api/tags", method="GET")
        except ProviderError as exc:
            return {
                "available": False,
                "provider": self.identity.to_dict(),
                "error": str(exc),
            }

        models = payload.get("models")
        installed = []
        if isinstance(models, list):
            installed = [
                item.get("name")
                for item in models
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
        return {
            "available": self.identity.model_id in installed,
            "reachable": True,
            "provider": self.identity.to_dict(),
            "installed_models": installed,
        }

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise ProviderError("model request prompt must not be empty")
        payload: dict[str, Any] = {
            "model": self.identity.model_id,
            "prompt": request.prompt,
            "stream": False,
        }
        if request.system:
            payload["system"] = request.system
        if request.options:
            payload["options"] = dict(request.options)

        result = self._request_json("/api/generate", payload=payload, method="POST")
        text = result.get("response")
        if not isinstance(text, str):
            raise ProviderError("Ollama response does not contain text in 'response'")

        provenance = {
            "transport": "ollama-http",
            "done": result.get("done"),
            "done_reason": result.get("done_reason"),
            "created_at": result.get("created_at"),
            "total_duration": result.get("total_duration"),
            "load_duration": result.get("load_duration"),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"),
        }
        return ModelResponse(text=text, provider=self.identity, provenance=provenance)
