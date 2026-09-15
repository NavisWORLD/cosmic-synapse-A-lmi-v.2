"""Hypothesis generation with explicit evidence, uncertainty, and provenance.

Patterns observed in the knowledge graph are not promoted to facts. Generated
hypotheses are labelled speculation and carry the exact observation/evidence
inputs that produced them. External investigation is provider-driven; this
module never fabricates search URLs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import ConfigSource, load_config


class HypothesisGenerator:
    """Generate testable hypotheses from graph observations."""

    def __init__(
        self,
        config_source: ConfigSource = "infrastructure/config.yaml",
        *,
        driver: Any = None,
        search_provider: Any = None,
    ):
        self.config = load_config(config_source)
        self.logger = logging.getLogger(__name__)
        self.search_provider = search_provider

        if driver is None:
            neo4j_config = self.config.get("infrastructure", {}).get("neo4j")
            if neo4j_config and neo4j_config.get("uri"):
                try:
                    from neo4j import GraphDatabase
                except ImportError:
                    driver = None
                else:
                    driver = GraphDatabase.driver(
                        neo4j_config["uri"],
                        auth=(neo4j_config.get("username"), neo4j_config.get("password")),
                    )
        self.driver = driver

    def _require_driver(self):
        if self.driver is None:
            raise RuntimeError(
                "Knowledge-graph analysis requires an injected Neo4j driver or the graph extra"
            )
        return self.driver

    def analyze_knowledge_gaps(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (a)-[r]->(b)
        WHERE NOT EXISTS {
            MATCH (a)-[:SIMILAR_TO]->(c)-[r]->(b)
        }
        RETURN a, r, b, count(*) as frequency
        ORDER BY frequency DESC
        LIMIT 10
        """
        driver = self._require_driver()
        with driver.session() as session:
            return [
                {
                    "type": "knowledge_gap",
                    "entity_a": dict(record["a"]),
                    "relationship": dict(record["r"]),
                    "entity_b": dict(record["b"]),
                    "frequency": record["frequency"],
                }
                for record in session.run(query)
            ]

    def detect_temporal_anomalies(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (a)-[r]->(b)
        WHERE r.timestamp IS NOT NULL
        RETURN a.name as entity, type(r) as relationship, b.name as target,
               collect(r.timestamp) as timestamps
        LIMIT 50
        """
        driver = self._require_driver()
        anomalies: List[Dict[str, Any]] = []
        with driver.session() as session:
            for record in session.run(query):
                timestamps = list(record["timestamps"] or [])
                if len(timestamps) > 5:
                    anomalies.append(
                        {
                            "type": "temporal_anomaly",
                            "entity": record["entity"],
                            "relationship": record["relationship"],
                            "target": record["target"],
                            "timestamps": timestamps,
                        }
                    )
        return anomalies

    def find_correlations(self) -> List[Dict[str, Any]]:
        query = """
        MATCH (a)-[r1]->(c)<-[r2]-(b)
        WHERE a <> b
        RETURN a.name as entity_a, b.name as entity_b, count(*) as cooccurrence
        ORDER BY cooccurrence DESC
        LIMIT 20
        """
        driver = self._require_driver()
        with driver.session() as session:
            return [
                {
                    "type": "correlation",
                    "entity_a": record["entity_a"],
                    "entity_b": record["entity_b"],
                    "cooccurrence": record["cooccurrence"],
                }
                for record in session.run(query)
            ]

    @staticmethod
    def _describe_pattern(pattern: Dict[str, Any]) -> tuple[str, float, list[str]]:
        pattern_type = pattern.get("type", "generic")
        if pattern_type == "knowledge_gap":
            first = pattern.get("entity_a", {})
            second = pattern.get("entity_b", {})
            first_name = first.get("name", "unknown") if isinstance(first, dict) else str(first)
            second_name = second.get("name", "unknown") if isinstance(second, dict) else str(second)
            text = f"A relationship between {first_name} and {second_name} may warrant investigation."
            queries = [f"{first_name} {second_name} relationship evidence"]
            return text, 0.55, queries
        if pattern_type == "correlation":
            first_name = str(pattern.get("entity_a", "unknown"))
            second_name = str(pattern.get("entity_b", "unknown"))
            text = (
                f"The observed co-occurrence of {first_name} and {second_name} may reflect "
                "a relationship that should be tested against independent evidence."
            )
            queries = [f"{first_name} {second_name} independent evidence"]
            return text, 0.5, queries
        if pattern_type == "temporal_anomaly":
            entity = str(pattern.get("entity", "unknown"))
            target = str(pattern.get("target", "unknown"))
            text = f"The timing pattern involving {entity} and {target} may contain a reproducible anomaly."
            queries = [f"{entity} {target} temporal pattern evidence"]
            return text, 0.4, queries
        return "The observed pattern may warrant a controlled follow-up test.", 0.25, ["pattern independent evidence"]

    def _search(self, queries: List[str]) -> List[Dict[str, Any]]:
        if self.search_provider is None:
            return []
        results: List[Dict[str, Any]] = []
        for query in queries:
            provider_results = self.search_provider.search(query)
            for item in provider_results or []:
                record = dict(item)
                record.setdefault("query", query)
                results.append(record)
        return results

    def generate_hypothesis(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(pattern, dict) or not pattern:
            raise ValueError("pattern must be a non-empty mapping")

        text, confidence, queries = self._describe_pattern(pattern)
        provider_name = None
        if self.search_provider is not None:
            provider_name = getattr(
                self.search_provider,
                "name",
                self.search_provider.__class__.__name__,
            )
        investigation_results = self._search(queries)
        created_at = datetime.now(timezone.utc).isoformat()
        hypothesis = {
            "id": f"hyp_{created_at}",
            "classification": "hypothesis",
            "type": pattern.get("type", "generic"),
            "hypothesis_text": text,
            "observation": dict(pattern),
            "speculation": True,
            "confidence": confidence,
            "uncertainty": 1.0 - confidence,
            "evidence_inputs": [pattern],
            "investigation_queries": queries,
            "investigation_results": investigation_results,
            "provenance": {
                "generator": "a_lmi.reasoning.hypothesis_generator",
                "search_provider": provider_name,
                "generated_at": created_at,
            },
            "created_at": created_at,
        }
        self.logger.info("Generated hypothesis from %s observation", hypothesis["type"])
        return hypothesis

    def generate_all_hypotheses(self) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        patterns.extend(self.analyze_knowledge_gaps())
        patterns.extend(self.detect_temporal_anomalies())
        patterns.extend(self.find_correlations())
        return [self.generate_hypothesis(pattern) for pattern in patterns]

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
