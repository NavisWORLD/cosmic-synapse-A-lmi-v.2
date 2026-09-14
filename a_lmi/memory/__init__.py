"""Memory-layer integrations.

The active package keeps infrastructure clients optional. Importing
``a_lmi.memory`` must not require Milvus, MinIO, or Neo4j; those SDKs are
loaded only when their concrete client is requested/constructed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["VectorDBClient", "ObjectStorageClient", "TKGClient"]

_LAZY_IMPORTS = {
    "VectorDBClient": ("a_lmi.memory.vector_db_client", "VectorDBClient"),
    "ObjectStorageClient": ("a_lmi.memory.object_storage_client", "ObjectStorageClient"),
    "TKGClient": ("a_lmi.memory.tkg_client", "TKGClient"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
