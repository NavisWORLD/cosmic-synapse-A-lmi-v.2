"""LightToken: inspectable multimodal information container.

The active implementation keeps three independently inspectable layers:
1. semantic/model embedding;
2. perceptual/content fingerprint;
3. an embedding spectral transform.

The spectral layer is a one-dimensional real-input FFT over embedding
coordinates. It is intentionally *not* described as a Graph Fourier
Transform because no graph Laplacian/eigenbasis is involved.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

EMBEDDING_DIMENSION = 1536
SPECTRAL_DIMENSION = EMBEDDING_DIMENSION // 2 + 1
SPECTRAL_TRANSFORM = "embedding_rfft"


class LightToken:
    """Atomic, serializable information record used by the active A-LMI layer."""

    def __init__(
        self,
        source_uri: str,
        modality: str,
        raw_data_ref: str,
        content_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.token_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.source_uri = source_uri
        self.modality = modality
        self.raw_data_ref = raw_data_ref
        self.content_text = content_text
        self.metadata = dict(metadata or {})

        self.joint_embedding: Optional[np.ndarray] = None
        self.perceptual_hash: Optional[str] = None
        self.spectral_signature: Optional[np.ndarray] = None

        self._embedding_set = False
        self._hash_computed = False
        self._spectral_computed = False

    def set_embedding(self, embedding: np.ndarray) -> None:
        """Set a 1536-dimensional embedding and derive its one-sided spectrum."""

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Expected embedding shape ({EMBEDDING_DIMENSION},), got {vector.shape}"
            )

        self.joint_embedding = vector
        self._embedding_set = True
        self.spectral_signature = self._compute_spectral_signature(vector)
        self._spectral_computed = True
        self.metadata["spectral_transform"] = SPECTRAL_TRANSFORM
        self.metadata["spectral_dimension"] = SPECTRAL_DIMENSION

    def set_perceptual_hash(self, phash: str) -> None:
        self.perceptual_hash = phash
        self._hash_computed = True

    def _compute_spectral_signature(self, embedding: np.ndarray) -> np.ndarray:
        """Return the one-sided DFT of a real-valued embedding vector."""

        return np.fft.rfft(np.asarray(embedding, dtype=np.float32)).astype(np.complex64)

    def get_spectral_power(self) -> np.ndarray:
        """Return magnitude of the stored embedding spectrum as real float32 values."""

        if not self._spectral_computed or self.spectral_signature is None:
            raise ValueError("Spectral signature not computed. Call set_embedding first.")
        return np.abs(self.spectral_signature).astype(np.float32)

    def get_dominant_frequency(self) -> tuple[int, float]:
        """Return dominant *spectral-bin index* and magnitude.

        The index is not a physical frequency unless an external mapping is
        explicitly defined for the embedding coordinate axis.
        """

        power = self.get_spectral_power()
        dominant_idx = int(np.argmax(power))
        return dominant_idx, float(power[dominant_idx])

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "token_id": self.token_id,
            "timestamp": self.timestamp,
            "source_uri": self.source_uri,
            "modality": self.modality,
            "raw_data_ref": self.raw_data_ref,
            "content_text": self.content_text,
            "metadata": self.metadata,
            "perceptual_hash": self.perceptual_hash,
        }

        if self.joint_embedding is not None:
            result["joint_embedding"] = self.joint_embedding.tolist()

        if self.spectral_signature is not None:
            # Store complex64 components directly. Magnitude/phase was used by
            # the historical serializer, but reconstructing a complex number
            # through atan/exp introduces avoidable numeric drift. Direct real
            # and imaginary components round-trip the active representation.
            result["spectral_signature_real"] = self.spectral_signature.real.astype(
                np.float32
            ).tolist()
            result["spectral_signature_imag"] = self.spectral_signature.imag.astype(
                np.float32
            ).tolist()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LightToken":
        token = cls(
            source_uri=data["source_uri"],
            modality=data["modality"],
            raw_data_ref=data["raw_data_ref"],
            content_text=data.get("content_text"),
            metadata=data.get("metadata"),
        )

        token.token_id = data["token_id"]
        token.timestamp = data["timestamp"]
        token.perceptual_hash = data.get("perceptual_hash")
        token._hash_computed = token.perceptual_hash is not None

        if "joint_embedding" in data:
            vector = np.asarray(data["joint_embedding"], dtype=np.float32)
            if vector.shape != (EMBEDDING_DIMENSION,):
                raise ValueError(
                    f"Serialized embedding must have shape ({EMBEDDING_DIMENSION},), got {vector.shape}"
                )
            token.joint_embedding = vector
            token._embedding_set = True

        if "spectral_signature_real" in data or "spectral_signature_imag" in data:
            if "spectral_signature_real" not in data or "spectral_signature_imag" not in data:
                raise ValueError("Serialized spectral real/imag components are incomplete")
            real = np.asarray(data["spectral_signature_real"], dtype=np.float32)
            imag = np.asarray(data["spectral_signature_imag"], dtype=np.float32)
            if real.shape != imag.shape:
                raise ValueError("Serialized spectral real/imag shapes do not match")
            token.spectral_signature = (real + 1j * imag).astype(np.complex64)
            token._spectral_computed = True
            token._label_spectral_shape(real.size)
        elif "spectral_signature_magnitude" in data:
            # Backward-compatible reader for historical magnitude/phase JSON.
            magnitude = np.asarray(data["spectral_signature_magnitude"], dtype=np.float32)
            phase = np.asarray(data["spectral_signature_phase"], dtype=np.float32)
            if magnitude.shape != phase.shape:
                raise ValueError("Serialized spectral magnitude/phase shapes do not match")
            token.spectral_signature = (
                magnitude * np.exp(1j * phase)
            ).astype(np.complex64)
            token._spectral_computed = True
            token._label_spectral_shape(magnitude.size)

        return token

    def _label_spectral_shape(self, size: int) -> None:
        """Describe modern vs historical spectral payloads without relabeling history."""

        if size == SPECTRAL_DIMENSION:
            self.metadata.setdefault("spectral_transform", SPECTRAL_TRANSFORM)
            self.metadata.setdefault("spectral_dimension", SPECTRAL_DIMENSION)
        else:
            self.metadata.setdefault("spectral_transform", "legacy_embedding_fft")
            self.metadata.setdefault("spectral_dimension", int(size))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "LightToken":
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        status = []
        if self._embedding_set:
            status.append("EMB")
        if self._hash_computed:
            status.append("HASH")
        if self._spectral_computed:
            status.append("SPEC")
        status_str = "/".join(status) if status else "RAW"
        return f"LightToken({self.modality}, {status_str}, ID={self.token_id[:8]}...)"


def spectral_similarity(
    token_a: LightToken, token_b: LightToken, method: str = "power_correlation"
) -> float:
    """Compare embedding spectra; this is an experimental retrieval signal."""

    if not (token_a._spectral_computed and token_b._spectral_computed):
        raise ValueError("Both tokens must have computed spectral signatures")

    power_a = token_a.get_spectral_power()
    power_b = token_b.get_spectral_power()
    if power_a.shape != power_b.shape:
        raise ValueError(
            f"Spectral shapes must match, got {power_a.shape} and {power_b.shape}"
        )

    if method == "power_correlation":
        if np.std(power_a) == 0 or np.std(power_b) == 0:
            return 1.0 if np.array_equal(power_a, power_b) else 0.0
        return float(np.corrcoef(power_a, power_b)[0, 1])
    if method == "cosine":
        denominator = np.linalg.norm(power_a) * np.linalg.norm(power_b)
        if denominator == 0:
            return 1.0 if np.array_equal(power_a, power_b) else 0.0
        return float(np.dot(power_a, power_b) / denominator)
    if method == "euclidean":
        distance = np.linalg.norm(power_a - power_b)
        max_distance = np.linalg.norm(power_a) + np.linalg.norm(power_b)
        if max_distance == 0:
            return 1.0
        return float(1.0 - (distance / max_distance))
    raise ValueError(f"Unknown similarity method: {method}")


def resonance_match(
    query_token: LightToken,
    candidate_tokens: list[LightToken],
    threshold: float = 0.7,
) -> list[tuple[LightToken, float]]:
    """Return candidates above an experimental spectral-similarity threshold."""

    matches = []
    for candidate in candidate_tokens:
        similarity = spectral_similarity(query_token, candidate)
        if similarity >= threshold:
            matches.append((candidate, similarity))
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches
