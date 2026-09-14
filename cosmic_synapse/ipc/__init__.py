"""Versioned Cosmic Synapse IPC public API.

The message schema is dependency-light. The WebSocket bridge is loaded only
when explicitly requested so schema validation does not require ``websockets``.
"""

from .schema import PROTOCOL_VERSION, decode_message, encode_message

__all__ = ["PROTOCOL_VERSION", "encode_message", "decode_message", "IPCBridge"]


def __getattr__(name):
    if name == "IPCBridge":
        from .bridge import IPCBridge

        return IPCBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
