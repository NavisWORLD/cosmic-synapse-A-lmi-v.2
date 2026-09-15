"""Visualization integrations.

Rendering stacks are optional. Importing ``interfaces.visualization`` or a
single renderer must not require Dash/Plotly unless that renderer is actually
used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["KnowledgeGraph3D", "GraphVisualizationApp"]

_LAZY_IMPORTS = {
    "KnowledgeGraph3D": ("interfaces.visualization.graph_3d", "KnowledgeGraph3D"),
    "GraphVisualizationApp": ("interfaces.visualization.webapp", "GraphVisualizationApp"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
