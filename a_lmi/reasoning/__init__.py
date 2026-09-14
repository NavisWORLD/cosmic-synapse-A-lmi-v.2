"""Reasoning integrations loaded on demand.

The dependency-light package can expose hypothesis/model interfaces without
requiring optional scikit-learn, Neo4j, or external model SDKs at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SpectralPatternRecognizer",
    "AnomalyDetector",
    "HypothesisGenerator",
    "MathEngine",
    "LogicEngine",
]

_LAZY_IMPORTS = {
    "SpectralPatternRecognizer": ("a_lmi.reasoning.pattern_recognition", "SpectralPatternRecognizer"),
    "AnomalyDetector": ("a_lmi.reasoning.pattern_recognition", "AnomalyDetector"),
    "HypothesisGenerator": ("a_lmi.reasoning.hypothesis_generator", "HypothesisGenerator"),
    "MathEngine": ("a_lmi.reasoning.math_engine", "MathEngine"),
    "LogicEngine": ("a_lmi.reasoning.logic_engine", "LogicEngine"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
