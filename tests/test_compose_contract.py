from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "infrastructure" / "docker-compose.yml"


def _compose():
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_active_compose_uses_environment_credentials_not_committed_passwords():
    text = COMPOSE.read_text(encoding="utf-8")
    assert "admin123456" not in text
    assert "vibrational123" not in text
    assert "change-me-local-only" not in text
    assert "change-me-milvus-local-only" not in text
    assert "${A_LMI_MINIO_ACCESS_KEY:?" in text
    assert "${A_LMI_MINIO_SECRET_KEY:?" in text
    assert "${MILVUS_MINIO_ACCESS_KEY:?" in text
    assert "${MILVUS_MINIO_SECRET_KEY:?" in text
    assert "${A_LMI_NEO4J_USERNAME" in text
    assert "${A_LMI_NEO4J_PASSWORD:?" in text


def test_active_compose_host_ports_bind_loopback_only():
    compose = _compose()
    for service in compose["services"].values():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:")


def test_kafka_hostname_and_internal_advertising_are_valid():
    kafka = _compose()["services"]["kafka"]
    assert kafka["hostname"] == "kafka"
    advertised = kafka["environment"]["KAFKA_ADVERTISED_LISTENERS"]
    assert "PLAINTEXT_INTERNAL://kafka:9093" in advertised


def test_minio_services_use_the_supported_quay_registry():
    compose = _compose()
    assert compose["services"]["minio"]["image"].startswith("quay.io/minio/minio")
    assert compose["services"]["minio-storage"]["image"].startswith("quay.io/minio/minio")


def test_milvus_runs_the_multivector_capable_standalone_server():
    milvus = _compose()["services"]["milvus"]
    assert milvus["image"] == "milvusdb/milvus:v2.4.23"
    assert milvus["command"] == ["milvus", "run", "standalone"]
