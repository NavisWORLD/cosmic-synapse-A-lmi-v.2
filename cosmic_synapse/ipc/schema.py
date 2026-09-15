"""Versioned JSON message schema shared by the Python/Unity IPC bridge."""

from __future__ import annotations

import json
from typing import Any, Mapping

PROTOCOL_VERSION = 1
MESSAGE_TYPES = {"command", "command_received", "status", "pattern_data"}


def encode_message(message_type: str, payload: Mapping[str, Any] | None = None) -> str:
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"Unknown IPC message type: {message_type}")
    message = {
        "version": PROTOCOL_VERSION,
        "type": message_type,
        "payload": dict(payload or {}),
    }
    return json.dumps(message, separators=(",", ":"), sort_keys=True)


def decode_message(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        message = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid IPC JSON message") from exc
    if not isinstance(message, dict):
        raise ValueError("IPC message must be a JSON object")
    if message.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported IPC protocol version: {message.get('version')}")
    if message.get("type") not in MESSAGE_TYPES:
        raise ValueError(f"Unknown IPC message type: {message.get('type')}")
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("IPC payload must be an object")
    return message
