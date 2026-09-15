"""Live VM integration probes for external service gates.

These probes intentionally fail closed. They are not mocks: every command talks to
an actual service process started by the CI VM. Persistence checks are split into
write/read commands so the workflow can restart the service between them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from a_lmi.core.light_token import LightToken
from a_lmi.memory.object_storage_client import ObjectStorageClient
from a_lmi.memory.tkg_client import TKGClient
from a_lmi.memory.vector_db_client import VectorDBClient
from a_lmi.providers import ModelRequest, OllamaProvider

PAYLOAD = b"cosmic-synapse-vm-persistence-v1"
MINIO_OBJECT = "vm-gate/persistence-v1.bin"


def _json(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True))


def _minio_config() -> dict:
    return {
        "infrastructure": {
            "minio": {
                "endpoint": os.getenv("A_LMI_MINIO_ENDPOINT", "127.0.0.1:9000"),
                "access_key": os.environ["A_LMI_MINIO_ACCESS_KEY"],
                "secret_key": os.environ["A_LMI_MINIO_SECRET_KEY"],
                "bucket": os.getenv("A_LMI_MINIO_BUCKET", "a-lmi-vm-gate"),
                "secure": False,
            }
        }
    }


def minio_write() -> None:
    client = ObjectStorageClient(_minio_config())
    record = client.store_bytes(PAYLOAD, MINIO_OBJECT, "application/octet-stream")
    assert record.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert client.retrieve_uri(record.uri) == PAYLOAD
    _json({"gate": "minio-write", "uri": record.uri, "sha256": record.sha256})


def minio_read() -> None:
    client = ObjectStorageClient(_minio_config())
    uri = f"minio://{client.bucket_name}/{MINIO_OBJECT}"
    recovered = client.retrieve_uri(uri)
    assert recovered == PAYLOAD
    _json({"gate": "minio-read-after-restart", "uri": uri, "sha256": hashlib.sha256(recovered).hexdigest()})


def _neo4j_config() -> dict:
    return {
        "infrastructure": {
            "neo4j": {
                "uri": os.getenv("A_LMI_NEO4J_URI", "bolt://127.0.0.1:7687"),
                "username": os.getenv("A_LMI_NEO4J_USERNAME", "neo4j"),
                "password": os.environ["A_LMI_NEO4J_PASSWORD"],
                "database": "neo4j",
            }
        }
    }


def neo4j_write(marker: str) -> None:
    left = f"vm-left-{marker}"
    right = f"vm-right-{marker}"
    client = TKGClient(_neo4j_config())
    try:
        assert client.create_entity("CITest", left, {"marker": marker}) is not None
        assert client.create_entity("CITest", right, {"marker": marker}) is not None
        assert client.create_relationship(left, right, "VM_LINK", {"marker": marker}) is not None
        graph = client.fetch_graph(limit=100)
        names = {node.get("name") for node in graph["nodes"]}
        assert left in names and right in names
        assert any(edge["type"] == "VM_LINK" for edge in graph["edges"])
    finally:
        client.close()
    _json({"gate": "neo4j-write", "marker": marker})


def neo4j_read(marker: str) -> None:
    left = f"vm-left-{marker}"
    right = f"vm-right-{marker}"
    client = TKGClient(_neo4j_config())
    try:
        entities = client.query_entities("CITest", limit=100)
        names = {entity.get("name") for entity in entities}
        assert left in names and right in names
        graph = client.fetch_graph(limit=100)
        assert any(edge["type"] == "VM_LINK" for edge in graph["edges"])
    finally:
        client.close()
    _json({"gate": "neo4j-read-after-restart", "marker": marker})


def _milvus_config() -> dict:
    return {
        "infrastructure": {
            "milvus": {
                "host": os.getenv("A_LMI_MILVUS_HOST", "127.0.0.1"),
                "port": int(os.getenv("A_LMI_MILVUS_PORT", "19530")),
                "collection_name": os.getenv("A_LMI_MILVUS_COLLECTION", "light_tokens_vm_gate"),
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "index_params": {"nlist": 32},
            }
        },
        "a_lmi": {
            "data_structures": {
                "embedding_dimension": 1536,
                "spectral_signature_size": 769,
            }
        },
    }


def _vm_token(marker: str) -> LightToken:
    token = LightToken(
        source_uri="vm://integration",
        modality="text",
        raw_data_ref="vm://none",
        content_text=f"VM integration marker {marker}",
        metadata={"embedding_space": "vm:test-space", "marker": marker},
    )
    token.token_id = f"vm-token-{marker}"
    vector = np.linspace(-1.0, 1.0, 1536, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    token.set_embedding(vector)
    return token


def milvus_write(marker: str) -> None:
    token = _vm_token(marker)
    client = VectorDBClient(_milvus_config())
    client.insert_token(token)
    results = client.search_semantic(
        token.joint_embedding,
        limit=5,
        embedding_space="vm:test-space",
    )
    assert any(item["token_id"] == token.token_id for item in results)
    _json({"gate": "milvus-write-search", "marker": marker, "token_id": token.token_id})


def milvus_read(marker: str) -> None:
    token = _vm_token(marker)
    client = VectorDBClient(_milvus_config())
    results = client.search_semantic(
        token.joint_embedding,
        limit=5,
        embedding_space="vm:test-space",
    )
    assert any(item["token_id"] == token.token_id for item in results)
    _json({"gate": "milvus-read-after-restart", "marker": marker, "token_id": token.token_id})


def kafka_write(topic: str) -> None:
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=os.getenv("A_LMI_KAFKA_BOOTSTRAP", "127.0.0.1:9092"),
        value_serializer=lambda value: value.encode("utf-8"),
        retries=5,
    )
    value = f"before-restart:{topic}"
    metadata = producer.send(topic, value=value).get(timeout=20)
    producer.flush(timeout=20)
    producer.close(timeout=20)
    _json({"gate": "kafka-write", "topic": topic, "partition": metadata.partition, "offset": metadata.offset})


def kafka_read(topic: str) -> None:
    from kafka import KafkaConsumer, KafkaProducer

    bootstrap = os.getenv("A_LMI_KAFKA_BOOTSTRAP", "127.0.0.1:9092")
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=15000,
        group_id=f"vm-gate-reader-{topic}",
        value_deserializer=lambda value: value.decode("utf-8"),
    )
    expected = f"before-restart:{topic}"
    observed = [record.value for record in consumer]
    consumer.close()
    assert expected in observed, f"persisted Kafka record not found: {observed!r}"

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda value: value.encode("utf-8"),
    )
    producer.send(topic, value=f"after-restart:{topic}").get(timeout=20)
    producer.flush(timeout=20)
    producer.close(timeout=20)
    _json({"gate": "kafka-read-after-restart", "topic": topic, "observed": observed})


def ollama_live(model: str) -> None:
    provider = OllamaProvider(model_id=model, timeout=120.0, retries=1)
    health = provider.health()
    assert health.get("reachable") is True
    assert health.get("available") is True
    response = provider.generate(ModelRequest(prompt="Reply with exactly: VM_GATE_OK"))
    assert response.text.strip(), "Ollama returned empty text"
    assert response.provider.provider_id == "ollama"
    assert response.provider.model_id == model
    _json(
        {
            "gate": "ollama-live",
            "model": model,
            "response_chars": len(response.text),
            "provider": response.provider.to_dict(),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "kafka-write",
            "kafka-read",
            "minio-write",
            "minio-read",
            "neo4j-write",
            "neo4j-read",
            "milvus-write",
            "milvus-read",
            "ollama-live",
        ],
    )
    parser.add_argument("--marker", default=os.getenv("GITHUB_RUN_ID", str(int(time.time()))))
    parser.add_argument("--topic", default=None)
    parser.add_argument("--model", default="qwen2.5:0.5b")
    args = parser.parse_args()

    if args.command == "kafka-write":
        kafka_write(args.topic or f"cosmic-vm-{args.marker}")
    elif args.command == "kafka-read":
        kafka_read(args.topic or f"cosmic-vm-{args.marker}")
    elif args.command == "minio-write":
        minio_write()
    elif args.command == "minio-read":
        minio_read()
    elif args.command == "neo4j-write":
        neo4j_write(args.marker)
    elif args.command == "neo4j-read":
        neo4j_read(args.marker)
    elif args.command == "milvus-write":
        milvus_write(args.marker)
    elif args.command == "milvus-read":
        milvus_read(args.marker)
    elif args.command == "ollama-live":
        ollama_live(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
