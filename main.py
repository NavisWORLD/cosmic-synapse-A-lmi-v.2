#!/usr/bin/env python3
"""Canonical orchestration entry point for the active A-LMI runtime.

Heavy/optional services are imported lazily so configuration inspection does
not require every ML, audio, database, or hardware dependency.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).parent))

from a_lmi.config import ConfigSource, load_config


class ALMIOrchestrator:
    """Coordinate infrastructure-backed A-LMI services and graceful shutdown."""

    def __init__(self, config_source: ConfigSource = "infrastructure/config.yaml"):
        self.config: Mapping[str, Any] = load_config(config_source)

        log_file = Path(self.config.get("logging", {}).get("file", "logs/a_lmi.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_level = getattr(
            logging, str(self.config.get("logging", {}).get("level", "INFO")).upper(), logging.INFO
        )
        log_format = self.config.get("logging", {}).get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
        )
        self.logger = logging.getLogger(__name__)

        self.agent = None
        self.processing_core = None
        self.audio_processor = None
        self.service_threads: list[threading.Thread] = []
        self.running = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.logger.info("A-LMI orchestrator initialized")

    def _signal_handler(self, signum, frame):
        self.logger.info("Received signal %s; shutting down", signum)
        self.stop()
        raise SystemExit(0)

    def check_infrastructure(self) -> bool:
        checks = {
            "Kafka": self._check_kafka(),
            "MinIO": self._check_minio(),
            "Milvus": self._check_milvus(),
            "Neo4j": self._check_neo4j(),
        }
        for service, status in checks.items():
            self.logger.info("  %s: %s", service, "UP" if status else "DOWN")
        if not all(checks.values()):
            self.logger.error(
                "Full-stack infrastructure is not ready. Start the documented local stack "
                "or use deterministic/unit modes that do not require external services."
            )
            return False
        return True

    def _check_kafka(self) -> bool:
        try:
            from kafka import KafkaProducer

            producer = KafkaProducer(
                bootstrap_servers=self.config["infrastructure"]["kafka"]["bootstrap_servers"],
                request_timeout_ms=5000,
            )
            producer.close()
            return True
        except Exception as exc:
            self.logger.debug("Kafka health check failed: %s", exc)
            return False

    def _check_minio(self) -> bool:
        try:
            from minio import Minio

            cfg = self.config["infrastructure"]["minio"]
            if not cfg.get("access_key") or not cfg.get("secret_key"):
                return False
            client = Minio(
                cfg["endpoint"],
                access_key=cfg["access_key"],
                secret_key=cfg["secret_key"],
                secure=bool(cfg.get("secure", False)),
            )
            client.list_buckets()
            return True
        except Exception as exc:
            self.logger.debug("MinIO health check failed: %s", exc)
            return False

    def _check_milvus(self) -> bool:
        try:
            from pymilvus import connections

            cfg = self.config["infrastructure"]["milvus"]
            connections.connect(host=cfg["host"], port=cfg["port"])
            return True
        except Exception as exc:
            self.logger.debug("Milvus health check failed: %s", exc)
            return False

    def _check_neo4j(self) -> bool:
        try:
            from neo4j import GraphDatabase

            cfg = self.config["infrastructure"]["neo4j"]
            if not cfg.get("password"):
                return False
            driver = GraphDatabase.driver(
                cfg["uri"], auth=(cfg["username"], cfg["password"])
            )
            with driver.session(database=cfg.get("database")) as session:
                session.run("RETURN 1").consume()
            driver.close()
            return True
        except Exception as exc:
            self.logger.debug("Neo4j health check failed: %s", exc)
            return False

    def initialize_services(self) -> None:
        # Imports live here intentionally: minimal/config-only use does not need
        # Kafka, PyTorch/Transformers, PyAudio, Milvus, or Neo4j installed.
        from a_lmi.core.agent import ALMIAgent
        from a_lmi.services.processing_core import ProcessingCore

        self.agent = ALMIAgent(self.config)
        self.processing_core = ProcessingCore(self.config)

        try:
            from a_lmi.services.audio_processor import AudioProcessor

            self.audio_processor = AudioProcessor(self.config)
        except (ImportError, RuntimeError, OSError) as exc:
            self.logger.warning("Optional audio path unavailable: %s", exc)
            self.audio_processor = None

    def start(self) -> None:
        if self.running:
            self.logger.warning("Services already running")
            return
        if not self.check_infrastructure():
            return
        try:
            self.initialize_services()
        except Exception as exc:
            self.logger.error("Failed to initialize full-stack services: %s", exc, exc_info=True)
            return

        self.running = True
        if self.agent:
            thread = threading.Thread(target=self.agent.run, daemon=True)
            thread.start()
            self.service_threads.append(thread)
        if self.processing_core:
            thread = threading.Thread(target=self.processing_core.run, daemon=True)
            thread.start()
            self.service_threads.append(thread)
        if self.audio_processor:
            thread = threading.Thread(target=self._run_audio, daemon=True)
            thread.start()
            self.service_threads.append(thread)

        self.logger.info("A-LMI full-stack runtime is running")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def _run_audio(self) -> None:
        assert self.audio_processor is not None
        self.audio_processor.start_recording()
        try:
            while self.running:
                time.sleep(1)
        finally:
            self.audio_processor.stop_recording()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.audio_processor:
            self.audio_processor.stop_recording()
        for thread in self.service_threads:
            thread.join(timeout=5.0)
        self.logger.info("A-LMI runtime stopped")

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "active_threads": len(self.service_threads),
            "agent": self.agent is not None,
            "processing_core": self.processing_core is not None,
            "audio_processor": self.audio_processor is not None,
        }


def main() -> None:
    ALMIOrchestrator().start()


if __name__ == "__main__":
    main()
