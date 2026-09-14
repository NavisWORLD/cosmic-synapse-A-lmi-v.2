"""A-LMI autonomous perception/cognition/action orchestration loop."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from kafka import KafkaConsumer, KafkaProducer

from ..config import ConfigSource, load_config
from .light_token import LightToken


class ALMIAgent:
    """Infrastructure-backed A-LMI agent with explicit external memory layers."""

    def __init__(self, config_source: ConfigSource = "infrastructure/config.yaml"):
        self.config = load_config(config_source)
        logging.basicConfig(
            level=getattr(logging, str(self.config["logging"]["level"]).upper()),
            format=self.config["logging"]["format"],
        )
        self.logger = logging.getLogger(__name__)
        bootstrap_servers = self.config["infrastructure"]["kafka"]["bootstrap_servers"]
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )
        self.perception_consumer = KafkaConsumer(
            "web_crawler",
            "audio_processing",
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda message: json.loads(message.decode("utf-8")),
            enable_auto_commit=True,
        )
        self.reasoning_consumer = KafkaConsumer(
            "reasoning",
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda message: json.loads(message.decode("utf-8")),
            enable_auto_commit=True,
        )

    def run_perception_loop(self) -> None:
        for message in self.perception_consumer:
            try:
                if message.topic == "web_crawler":
                    self._process_web_data(message.value)
                elif message.topic == "audio_processing":
                    self._process_audio_data(message.value)
            except Exception as exc:
                self.logger.error("Perception error: %s", exc, exc_info=True)

    def _process_web_data(self, data: dict) -> None:
        token = LightToken(
            source_uri=data["url"],
            modality="text",
            raw_data_ref=data.get("html_ref") or "",
            content_text=data.get("extracted_text"),
            metadata={
                "crawled_at": data["timestamp"],
                "domain": data.get("domain"),
                "raw_sha256": data.get("html_sha256"),
            },
        )
        self._send_to_processing(token)

    def _process_audio_data(self, data: dict) -> None:
        token = LightToken(
            source_uri=data.get("stream_id", "microphone"),
            modality="audio" if data.get("type") == "audio" else "speech",
            raw_data_ref=data.get("audio_ref") or "",
            content_text=data.get("transcription"),
            metadata={
                "esc_class": data.get("esc_class"),
                "timestamp": data["timestamp"],
                "sample_rate": data.get("sample_rate"),
                "raw_sha256": data.get("audio_sha256"),
            },
        )
        self._send_to_processing(token)

    def _send_to_processing(self, token: LightToken) -> None:
        self.producer.send("light_tokens", token.to_dict())

    def run_cognition_loop(self) -> None:
        processed_consumer = KafkaConsumer(
            "light_tokens_processed",
            bootstrap_servers=self.config["infrastructure"]["kafka"]["bootstrap_servers"],
            value_deserializer=lambda message: json.loads(message.decode("utf-8")),
            enable_auto_commit=True,
        )
        for message in processed_consumer:
            try:
                token = LightToken.from_dict(message.value)
                self._store_in_memory(token)
                self._check_reasoning_triggers(token)
            except Exception as exc:
                self.logger.error("Cognition error: %s", exc, exc_info=True)

    def _store_in_memory(self, token: LightToken) -> None:
        from ..memory.object_storage_client import ObjectStorageClient
        from ..memory.tkg_client import TKGClient
        from ..memory.vector_db_client import VectorDBClient

        if token.joint_embedding is not None:
            VectorDBClient(self.config).insert_token(token)

        # The ingestion services are responsible for persisting raw bytes and
        # placing their URI/hash on the token. This call verifies/reference-
        # manages an already persisted artifact; it no longer fabricates data.
        if token.raw_data_ref:
            ObjectStorageClient(self.config).store_raw_data(token)

        if token.content_text:
            client = TKGClient(self.config)
            try:
                client.store_entities_from_token(token)
            finally:
                client.close()

    def _check_reasoning_triggers(self, token: LightToken) -> None:
        self.logger.debug("Reasoning trigger check for %s", token.token_id)

    def run_action_loop(self) -> None:
        for message in self.reasoning_consumer:
            try:
                result = message.value
                if result.get("type") == "hypothesis":
                    self._handle_hypothesis(result)
                elif result.get("type") == "response":
                    self._handle_response(result)
                elif result.get("type") == "learning_goal":
                    self._handle_learning_goal(result)
            except Exception as exc:
                self.logger.error("Action error: %s", exc, exc_info=True)

    def _handle_hypothesis(self, result: dict) -> None:
        for url in result.get("investigation_urls", []):
            self.producer.send(
                "web_crawler_queue",
                {
                    "url": url,
                    "priority": "high",
                    "reason": "hypothesis_investigation",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    def _handle_response(self, result: dict) -> None:
        self.logger.info("Response ready: %s", str(result.get("response_text", ""))[:100])

    def _handle_learning_goal(self, result: dict) -> None:
        self.logger.info("Learning goal: %s", result.get("goal_text"))

    def run(self) -> None:
        import threading

        threads = [
            threading.Thread(target=self.run_perception_loop, daemon=True),
            threading.Thread(target=self.run_cognition_loop, daemon=True),
            threading.Thread(target=self.run_action_loop, daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("A-LMI agent loop stopped")


def main() -> None:
    ALMIAgent().run()


if __name__ == "__main__":
    main()
