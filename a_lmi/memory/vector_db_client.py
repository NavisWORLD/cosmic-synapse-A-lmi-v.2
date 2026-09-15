"""Milvus vector-memory integration for active LightToken records.

Milvus is optional at import time. Pure schema helpers are available in the
minimal package so dimensions can be verified without installing or starting a
vector database.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.light_token import EMBEDDING_DIMENSION, SPECTRAL_DIMENSION, LightToken


def vector_schema_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    data_structures = config.get("a_lmi", {}).get("data_structures", {})
    configured_embedding = int(
        data_structures.get("embedding_dimension", EMBEDDING_DIMENSION)
    )
    configured_spectral = int(
        data_structures.get("spectral_signature_size", SPECTRAL_DIMENSION)
    )
    if configured_embedding != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Configured embedding dimension {configured_embedding} does not match "
            f"LightToken dimension {EMBEDDING_DIMENSION}"
        )
    if configured_spectral != SPECTRAL_DIMENSION:
        raise ValueError(
            f"Configured spectral dimension {configured_spectral} does not match "
            f"LightToken one-sided spectrum {SPECTRAL_DIMENSION}"
        )
    milvus = config.get("infrastructure", {}).get("milvus", {})
    return {
        "collection_name": milvus.get("collection_name", "light_tokens_v2"),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "spectral_dimension": SPECTRAL_DIMENSION,
        "stores_embedding_space": True,
    }


class VectorDBClient:
    """Milvus client for semantic and experimental spectral retrieval."""

    def __init__(self, config: Dict[str, Any]):
        self.root_config = config
        self.config = config["infrastructure"]["milvus"]
        self.contract = vector_schema_contract(config)
        self.logger = logging.getLogger(__name__)

        try:
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                utility,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Milvus support is optional; install the storage/vector extra"
            ) from exc

        self._Collection = Collection
        self._CollectionSchema = CollectionSchema
        self._DataType = DataType
        self._FieldSchema = FieldSchema
        self._utility = utility
        connections.connect(
            alias="default", host=self.config["host"], port=self.config["port"]
        )
        self.collection_name = self.contract["collection_name"]
        self.collection = None
        self._setup_collection()

    def _setup_collection(self) -> None:
        if self._utility.has_collection(self.collection_name):
            self.collection = self._Collection(self.collection_name)
            self.logger.info("Loaded existing collection: %s", self.collection_name)
        else:
            self._create_collection()

    def _create_collection(self) -> None:
        FieldSchema = self._FieldSchema
        DataType = self._DataType
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="token_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="modality", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="embedding_space", dtype=DataType.VARCHAR, max_length=160),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="joint_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.contract["embedding_dimension"],
            ),
            FieldSchema(
                name="spectral_power",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.contract["spectral_dimension"],
            ),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = self._CollectionSchema(
            fields=fields,
            description=(
                "A-LMI LightToken vectors. embedding_space must be checked before "
                "cross-modal semantic comparison."
            ),
        )
        self.collection = self._Collection(name=self.collection_name, schema=schema)
        index_params = {
            "metric_type": self.config.get("metric_type", "L2"),
            "index_type": self.config.get("index_type", "IVF_FLAT"),
            "params": self.config.get("index_params", {"nlist": 1024}),
        }
        self.collection.create_index("joint_embedding", index_params=index_params)
        self.collection.create_index("spectral_power", index_params=index_params)
        self.logger.info("Created collection: %s", self.collection_name)

    def insert_token(self, token: LightToken) -> None:
        if token.joint_embedding is None:
            raise ValueError("Cannot store vector record without a joint_embedding")
        power = token.get_spectral_power()
        if power.shape != (self.contract["spectral_dimension"],):
            raise ValueError(
                f"Unexpected spectral shape {power.shape}; expected "
                f"({self.contract['spectral_dimension']},)"
            )
        embedding_space = str(token.metadata.get("embedding_space", "unknown"))
        data = [
            {
                "id": token.token_id,
                "token_id": token.token_id,
                "modality": token.modality,
                "embedding_space": embedding_space,
                "timestamp": token.timestamp,
                "content_text": token.content_text[:65535] if token.content_text else "",
                "joint_embedding": token.joint_embedding.tolist(),
                "spectral_power": power.tolist(),
                "metadata": token.metadata,
            }
        ]
        self.collection.insert(data)
        self.collection.flush()

    def _format_results(self, results) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for hits in results:
            for hit in hits:
                formatted.append(
                    {
                        "token_id": hit.entity.get("token_id"),
                        "modality": hit.entity.get("modality"),
                        "embedding_space": hit.entity.get("embedding_space"),
                        "timestamp": hit.entity.get("timestamp"),
                        "content_text": hit.entity.get("content_text"),
                        "metadata": hit.entity.get("metadata"),
                        "distance": float(hit.distance),
                        "score": 1.0 / (1.0 + hit.distance),
                    }
                )
        return formatted

    def search_semantic(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filter_expr: Optional[str] = None,
        embedding_space: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.shape != (self.contract["embedding_dimension"],):
            raise ValueError("query_embedding has the wrong dimension")
        expressions = []
        if filter_expr:
            expressions.append(f"({filter_expr})")
        if embedding_space:
            safe = embedding_space.replace("\\", "\\\\").replace('"', '\\"')
            expressions.append(f'embedding_space == "{safe}"')
        expr = " and ".join(expressions) or None
        results = self.collection.search(
            data=[vector.tolist()],
            anns_field="joint_embedding",
            param={"metric_type": self.config.get("metric_type", "L2"), "params": {"nprobe": 10}},
            limit=limit,
            expr=expr,
            output_fields=[
                "token_id",
                "modality",
                "embedding_space",
                "timestamp",
                "content_text",
                "metadata",
            ],
        )
        return self._format_results(results)

    def search_spectral(
        self,
        query_spectral_power: np.ndarray,
        limit: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        vector = np.asarray(query_spectral_power, dtype=np.float32)
        if vector.shape != (self.contract["spectral_dimension"],):
            raise ValueError("query_spectral_power has the wrong dimension")
        results = self.collection.search(
            data=[vector.tolist()],
            anns_field="spectral_power",
            param={"metric_type": self.config.get("metric_type", "L2"), "params": {"nprobe": 10}},
            limit=limit,
            expr=filter_expr,
            output_fields=[
                "token_id",
                "modality",
                "embedding_space",
                "timestamp",
                "content_text",
                "metadata",
            ],
        )
        return self._format_results(results)

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_spectral_power: np.ndarray,
        semantic_weight: float = 0.7,
        spectral_weight: float = 0.3,
        limit: int = 10,
        embedding_space: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        semantic = self.search_semantic(
            query_embedding, limit=limit * 2, embedding_space=embedding_space
        )
        spectral = self.search_spectral(query_spectral_power, limit=limit * 2)
        combined: Dict[str, Dict[str, Any]] = {}
        for result in semantic:
            item = combined.setdefault(
                result["token_id"],
                {"result": result, "semantic_score": 0.0, "spectral_score": 0.0},
            )
            item["semantic_score"] = result["score"]
        for result in spectral:
            item = combined.setdefault(
                result["token_id"],
                {"result": result, "semantic_score": 0.0, "spectral_score": 0.0},
            )
            item["spectral_score"] = result["score"]
        final = []
        for item in combined.values():
            final.append(
                {
                    **item["result"],
                    "combined_score": semantic_weight * item["semantic_score"]
                    + spectral_weight * item["spectral_score"],
                }
            )
        final.sort(key=lambda item: item["combined_score"], reverse=True)
        return final[:limit]
