"""Temporal knowledge-graph integration with safe dynamic identifiers.

Neo4j is optional at import time and the driver can be injected for deterministic
tests. Dynamic labels/relationship types are validated before interpolation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..core.light_token import LightToken

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_cypher_identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Unsafe Cypher identifier: {value!r}")
    return value


# Compatibility alias used by newer visualization tests/docs.
validate_label = validate_cypher_identifier


class TKGClient:
    """Neo4j temporal graph client with injectable driver."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        driver: Any = None,
        verify_connection: bool = True,
    ):
        self.config = config["infrastructure"]["neo4j"]
        self.logger = logging.getLogger(__name__)
        if driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError(
                    "Neo4j support is optional; install the graph extra or inject a driver"
                ) from exc
            driver = GraphDatabase.driver(
                self.uri,
                auth=(self.config["username"], self.config["password"]),
            )
        self.driver = driver
        if verify_connection:
            with self._session() as session:
                session.run("RETURN 1").consume()
        self.logger.info("Neo4j client ready for %s", self.uri)

    @property
    def uri(self) -> str:
        return self.config["uri"]

    def _session(self):
        database = self.config.get("database")
        return self.driver.session(database=database) if database else self.driver.session()

    def store_entities_from_token(self, token: LightToken) -> int:
        if not token.content_text:
            return 0

        from ..services.ner_service import NERService

        entities = NERService().extract_entities(token.content_text)
        with self._session() as session:
            session.run(
                """
                MERGE (t:LightToken {token_id: $token_id})
                SET t.modality = $modality,
                    t.timestamp = $timestamp,
                    t.content_text = $content_text,
                    t.source_uri = $source_uri
                RETURN t
                """,
                token_id=token.token_id,
                modality=token.modality,
                timestamp=token.timestamp,
                content_text=token.content_text[:1000],
                source_uri=token.source_uri,
            ).consume()

            for entity in entities:
                label = validate_cypher_identifier(str(entity["label"]))
                session.run(
                    f"""
                    MERGE (e:{label} {{name: $name}})
                    SET e.description = $description
                    RETURN e
                    """,
                    name=entity["text"],
                    description=entity.get("description", label),
                ).consume()
                session.run(
                    f"""
                    MATCH (t:LightToken {{token_id: $token_id}})
                    MATCH (e:{label} {{name: $entity_name}})
                    MERGE (t)-[r:CONTAINS_ENTITY]->(e)
                    SET r.timestamp = $timestamp
                    RETURN r
                    """,
                    token_id=token.token_id,
                    entity_name=entity["text"],
                    timestamp=token.timestamp,
                ).consume()
        return len(entities)

    def create_entity(
        self,
        entity_type: str,
        entity_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        label = validate_cypher_identifier(entity_type)
        with self._session() as session:
            result = session.run(
                f"MERGE (e:{label} {{name: $name}}) SET e += $properties RETURN e",
                name=entity_name,
                properties=properties or {},
            )
            return result.single()

    def create_relationship(
        self,
        from_entity: str,
        to_entity: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ):
        rel_type = validate_cypher_identifier(relationship_type)
        with self._session() as session:
            result = session.run(
                f"""
                MATCH (a {{name: $from_name}}), (b {{name: $to_name}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $properties
                RETURN r
                """,
                from_name=from_entity,
                to_name=to_entity,
                properties=properties or {},
            )
            return result.single()

    def query_entities(self, entity_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        label = validate_cypher_identifier(entity_type)
        with self._session() as session:
            result = session.run(f"MATCH (e:{label}) RETURN e LIMIT $limit", limit=limit)
            return [dict(record["e"]) for record in result]

    def temporal_query(
        self,
        entity_name: str,
        relationship_type: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rel_type = validate_cypher_identifier(relationship_type)
        with self._session() as session:
            result = session.run(
                f"""
                MATCH (a {{name: $entity_name}})-[r:{rel_type}]->(b)
                WHERE ($start_time IS NULL OR r.timestamp >= $start_time)
                  AND ($end_time IS NULL OR r.timestamp <= $end_time)
                RETURN r, b
                ORDER BY r.timestamp DESC
                """,
                entity_name=entity_name,
                start_time=start_time,
                end_time=end_time,
            )
            return [
                {"relationship": dict(record["r"]), "target": dict(record["b"])}
                for record in result
            ]

    def fetch_graph(self, limit: int = 1000) -> Dict[str, List[Dict[str, Any]]]:
        """Return a renderer-neutral graph snapshot using stable Neo4j element IDs."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        edge_limit = max(limit, min(limit * 5, 5000))
        with self._session() as session:
            node_rows = session.run(
                """
                MATCH (n)
                RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties
                LIMIT $limit
                """,
                limit=limit,
            )
            nodes = []
            for row in node_rows:
                labels = list(row.get("labels") or [])
                properties = dict(row.get("properties") or {})
                nodes.append(
                    {
                        "id": row["id"],
                        "type": labels[0] if labels else "Unknown",
                        "labels": labels,
                        **properties,
                    }
                )

            edge_rows = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN elementId(a) AS source, elementId(b) AS target,
                       type(r) AS type, properties(r) AS properties
                LIMIT $limit
                """,
                limit=edge_limit,
            )
            edges = [
                {
                    "source": row["source"],
                    "target": row["target"],
                    "type": row["type"],
                    "properties": dict(row.get("properties") or {}),
                }
                for row in edge_rows
            ]
        return {"nodes": nodes, "edges": edges}

    def close(self) -> None:
        self.driver.close()
