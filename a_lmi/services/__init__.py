"""Perception and processing services.

Service integrations are intentionally lazy so importing the package does not
require optional microphone, crawler, Kafka, or ML stacks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["AudioProcessor", "WebCrawlerService", "ProcessingCore"]

_LAZY_IMPORTS = {
    "AudioProcessor": ("a_lmi.services.audio_processor", "AudioProcessor"),
    "WebCrawlerService": ("a_lmi.services.web_crawler", "WebCrawlerService"),
    "ProcessingCore": ("a_lmi.services.processing_core", "ProcessingCore"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
