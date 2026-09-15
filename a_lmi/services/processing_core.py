"""Processing service that turns raw LightTokens into encoded LightTokens.

This compatibility layer now accepts the canonical configuration mapping or a
path. Production image/audio paths no longer manufacture random vectors; if
raw bytes cannot be resolved the token remains explicitly unembedded.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import numpy as np
from kafka import KafkaConsumer, KafkaProducer

from ..config import ConfigSource, load_config
from ..core.light_token import LightToken
from .multimodal_encoder import MultimodalEncoder


class ProcessingCore:
    def __init__(self, config_source: ConfigSource = "infrastructure/config.yaml"):
        self.config = load_config(config_source)
        self.logger = logging.getLogger(__name__)
        self.embedding_model = MultimodalEncoder()
        bootstrap_servers = self.config["infrastructure"]["kafka"]["bootstrap_servers"]
        self.consumer = KafkaConsumer(
            "light_tokens",
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda message: json.loads(message.decode("utf-8")),
            enable_auto_commit=True,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )

    def process_token(self, token_dict: Dict[str, Any]) -> Dict[str, Any]:
        modality = token_dict["modality"]
        content_text = token_dict.get("content_text", "")
        raw_data_ref = token_dict.get("raw_data_ref") or ""
        embedding = self._generate_embedding(modality, content_text, raw_data_ref)
        phash = self._generate_perceptual_hash(modality, content_text, raw_data_ref)
        token = LightToken(
            source_uri=token_dict["source_uri"],
            modality=modality,
            raw_data_ref=raw_data_ref,
            content_text=content_text,
            metadata=token_dict.get("metadata", {}),
        )
        token.token_id = token_dict["token_id"]
        token.timestamp = token_dict["timestamp"]
        if embedding is not None:
            token.set_embedding(embedding)
            token.metadata.setdefault(
                "embedding_space", self.embedding_model.embedding_space_for(modality)
            )
        if phash:
            token.set_perceptual_hash(phash)
        return token.to_dict()

    def _resolve_raw_bytes(self, raw_ref: str) -> bytes | None:
        if not raw_ref:
            return None
        parsed = urlparse(raw_ref)
        if parsed.scheme in ("", "file"):
            path = Path(parsed.path if parsed.scheme == "file" else raw_ref)
            if path.is_file():
                return path.read_bytes()
        if parsed.scheme == "minio":
            from ..memory.object_storage_client import ObjectStorageClient

            return ObjectStorageClient(self.config).retrieve_uri(raw_ref)
        self.logger.warning("Unsupported raw artifact reference: %s", raw_ref)
        return None

    def _generate_embedding(
        self, modality: str, content: str, raw_ref: str
    ) -> np.ndarray | None:
        try:
            if modality == "text":
                return self.embedding_model.encode_text(content) if content else None
            if modality == "image":
                raw = self._resolve_raw_bytes(raw_ref)
                return self.embedding_model.encode_image(raw) if raw else None
            if modality in {"audio", "speech"}:
                raw = self._resolve_raw_bytes(raw_ref)
                return self.embedding_model.encode_audio(raw) if raw else None
            return None
        except Exception as exc:
            self.logger.error("Embedding generation failed for %s: %s", modality, exc)
            return None

    def _generate_perceptual_hash(
        self, modality: str, content: str, raw_ref: str
    ) -> str | None:
        raw = None
        if modality in {"image", "audio", "speech"}:
            raw = self._resolve_raw_bytes(raw_ref)
        if raw is not None:
            return hashlib.sha256(raw).hexdigest()
        if modality == "text" and content:
            return hashlib.sha256(content.encode("utf-8")).hexdigest()
        return None

    def run(self) -> None:
        for message in self.consumer:
            try:
                self.producer.send(
                    "light_tokens_processed", self.process_token(message.value)
                )
            except Exception as exc:
                self.logger.error("Token processing failed: %s", exc, exc_info=True)


def main() -> None:
    ProcessingCore().run()


if __name__ == "__main__":
    main()
