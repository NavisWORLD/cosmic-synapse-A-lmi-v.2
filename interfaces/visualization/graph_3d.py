"""Renderer-neutral Neo4j graph loading with optional Plotly visualization."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np


class KnowledgeGraph3D:
    """Load a real knowledge-graph snapshot and optionally render it in 3D."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fig = None
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.colors = {}

    def load_from_neo4j(self, neo4j_client) -> None:
        """Execute graph queries and normalize IDs to renderer indices."""

        if hasattr(neo4j_client, "fetch_graph"):
            snapshot = neo4j_client.fetch_graph(limit=1000)
        else:
            with neo4j_client.driver.session() as session:
                node_rows = session.run(
                    """
                    MATCH (n)
                    RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties
                    LIMIT 1000
                    """
                )
                edge_rows = session.run(
                    """
                    MATCH (a)-[r]->(b)
                    RETURN elementId(a) AS source, elementId(b) AS target,
                           type(r) AS type, properties(r) AS properties
                    LIMIT 5000
                    """
                )
                snapshot = {
                    "nodes": [
                        {
                            "id": row["id"],
                            "type": (list(row.get("labels") or []) or ["Unknown"])[0],
                            "labels": list(row.get("labels") or []),
                            **dict(row.get("properties") or {}),
                        }
                        for row in node_rows
                    ],
                    "edges": [
                        {
                            "source": row["source"],
                            "target": row["target"],
                            "type": row["type"],
                            "properties": dict(row.get("properties") or {}),
                        }
                        for row in edge_rows
                    ],
                }

        self.nodes = list(snapshot.get("nodes") or [])
        index_by_id = {node["id"]: index for index, node in enumerate(self.nodes)}
        normalized_edges = []
        for edge in snapshot.get("edges") or []:
            source_id = edge["source"]
            target_id = edge["target"]
            if source_id not in index_by_id or target_id not in index_by_id:
                self.logger.warning(
                    "Skipping edge with unloaded endpoint: %s -> %s", source_id, target_id
                )
                continue
            normalized_edges.append(
                {
                    **edge,
                    "source": index_by_id[source_id],
                    "target": index_by_id[target_id],
                }
            )
        self.edges = normalized_edges
        self.logger.info("Loaded %d nodes and %d edges", len(self.nodes), len(self.edges))

    def create_3d_graph(self):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise RuntimeError(
                "3D graph rendering requires the optional visualization dependencies"
            ) from exc

        positions = self._compute_layout()
        node_trace = go.Scatter3d(
            x=[pos[0] for pos in positions],
            y=[pos[1] for pos in positions],
            z=[pos[2] for pos in positions],
            mode="markers+text",
            marker=dict(
                size=10,
                color=[self._get_node_color(node) for node in self.nodes],
                showscale=False,
            ),
            text=[self._get_node_label(node) for node in self.nodes],
            textposition="middle center",
            name="Entities",
        )

        edge_traces = []
        for edge in self.edges:
            source = positions[edge["source"]]
            target = positions[edge["target"]]
            edge_traces.append(
                go.Scatter3d(
                    x=[source[0], target[0], None],
                    y=[source[1], target[1], None],
                    z=[source[2], target[2], None],
                    mode="lines",
                    hovertext=edge.get("type", "relationship"),
                    hoverinfo="text",
                    showlegend=False,
                )
            )

        self.fig = go.Figure(data=[node_trace] + edge_traces)
        self.fig.update_layout(
            title="A-LMI Knowledge Graph - 3D View",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            height=800,
        )
        return self.fig

    def _compute_layout(self) -> List[List[float]]:
        num_nodes = len(self.nodes)
        if num_nodes == 0:
            return []
        positions = []
        for index in range(num_nodes):
            angle1 = 2 * np.pi * index / num_nodes
            angle2 = np.pi * (index % 10) / 10
            positions.append(
                [
                    float(np.cos(angle1) * np.sin(angle2)),
                    float(np.sin(angle1) * np.sin(angle2)),
                    float(np.cos(angle2)),
                ]
            )
        return positions

    def _get_node_color(self, node: Dict[str, Any]) -> str:
        color_map = {
            "PERSON": "#FF6B6B",
            "Person": "#FF6B6B",
            "ORG": "#4ECDC4",
            "GPE": "#FFE66D",
            "LightToken": "#A8E6CF",
            "Unknown": "#CCCCCC",
        }
        return color_map.get(node.get("type", "Unknown"), color_map["Unknown"])

    def _get_node_label(self, node: Dict[str, Any]) -> str:
        return str(node.get("name", node.get("id", "Unknown")))[:40]

    def save_html(self, filepath: str) -> None:
        self.create_3d_graph().write_html(filepath)
