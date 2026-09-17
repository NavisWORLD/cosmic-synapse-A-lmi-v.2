#!/usr/bin/env python3
"""Generate synthetic LightToken cross-language golden fixtures.

The generator intentionally uses the preserved Python implementation as the
oracle. It never reads user data and all IDs/timestamps/arrays are fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from a_lmi.core.light_token import LightToken, spectral_similarity

FIXED_TIMESTAMP = "2000-01-01T00:00:00+00:00"
FIXTURE_IDS = {
    "active_zero": "00000000-0000-0000-0000-000000000001",
    "active_constant": "00000000-0000-0000-0000-000000000002",
    "active_impulse": "00000000-0000-0000-0000-000000000003",
    "active_sinusoid": "00000000-0000-0000-0000-000000000004",
    "active_random": "00000000-0000-0000-0000-000000000005",
}


def _canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _vectors() -> dict[str, np.ndarray]:
    impulse = np.zeros(1536, dtype=np.float32)
    impulse[17] = 1.0
    indices = np.arange(1536, dtype=np.float32)
    return {
        "active_zero": np.zeros(1536, dtype=np.float32),
        "active_constant": np.full(1536, 0.25, dtype=np.float32),
        "active_impulse": impulse,
        "active_sinusoid": np.sin(2.0 * np.pi * indices / 64.0).astype(np.float32),
        "active_random": np.random.default_rng(424242).standard_normal(1536).astype(np.float32),
    }


def _make_active(name: str, vector: np.ndarray) -> LightToken:
    token = LightToken(
        source_uri=f"fixture://{name}",
        modality="synthetic",
        raw_data_ref=f"fixture://raw/{name}",
        content_text=name,
        metadata={"fixture": name},
    )
    token.token_id = FIXTURE_IDS[name]
    token.timestamp = FIXED_TIMESTAMP
    token.set_perceptual_hash(f"phash-{name}")
    token.set_embedding(vector)
    return token


def _historical_fixture(vector: np.ndarray) -> dict:
    # Historical compatibility sample: magnitude/phase JSON and a full FFT
    # length. The active writer never emits this shape.
    spectrum = np.fft.fft(vector.astype(np.float32)).astype(np.complex64)
    return {
        "token_id": "00000000-0000-0000-0000-000000000099",
        "timestamp": FIXED_TIMESTAMP,
        "source_uri": "fixture://historical_magnitude_phase",
        "modality": "synthetic",
        "raw_data_ref": "fixture://raw/historical_magnitude_phase",
        "content_text": "historical_magnitude_phase",
        "metadata": {"fixture": "historical_magnitude_phase"},
        "perceptual_hash": "phash-historical",
        "joint_embedding": vector.tolist(),
        "spectral_signature_magnitude": np.abs(spectrum).astype(np.float32).tolist(),
        "spectral_signature_phase": np.angle(spectrum).astype(np.float32).tolist(),
    }


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    vectors = _vectors()
    active = {name: _make_active(name, vector) for name, vector in vectors.items()}
    files: dict[str, dict] = {}

    for name, token in active.items():
        path = output / f"{name}.json"
        payload = token.to_dict()
        encoded = _canonical_bytes(payload)
        path.write_bytes(encoded)
        files[path.name] = {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}

    historical_path = output / "historical_magnitude_phase.json"
    historical_bytes = _canonical_bytes(_historical_fixture(vectors["active_random"]))
    historical_path.write_bytes(historical_bytes)
    files[historical_path.name] = {
        "sha256": hashlib.sha256(historical_bytes).hexdigest(),
        "bytes": len(historical_bytes),
    }

    pairs = []
    names = list(active)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            for method in ("power_correlation", "cosine", "euclidean"):
                score = spectral_similarity(active[left_name], active[right_name], method)
                pairs.append(
                    {
                        "a": left_name,
                        "b": right_name,
                        "method": method,
                        "score": score,
                    }
                )
    similarity_bytes = _canonical_bytes({"pairs": pairs})
    (output / "similarity.json").write_bytes(similarity_bytes)
    files["similarity.json"] = {
        "sha256": hashlib.sha256(similarity_bytes).hexdigest(),
        "bytes": len(similarity_bytes),
    }

    manifest = {
        "fixture_version": 1,
        "synthetic_only": True,
        "embedding_dimension": 1536,
        "spectral_dimension": 769,
        "spectral_transform": "embedding_rfft",
        "timestamp": FIXED_TIMESTAMP,
        "rng_seed": 424242,
        "files": files,
    }
    manifest_bytes = _canonical_bytes(manifest)
    (output / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/lighttoken"))
    args = parser.parse_args()
    manifest = generate(args.output)
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
