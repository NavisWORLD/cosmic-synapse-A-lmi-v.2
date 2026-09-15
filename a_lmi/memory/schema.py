"""Pure memory-schema contracts shared by storage integrations."""

from __future__ import annotations

from ..core.light_token import EMBEDDING_DIMENSION, SPECTRAL_DIMENSION


def vector_field_dimensions() -> dict[str, int]:
    return {
        "joint_embedding": EMBEDDING_DIMENSION,
        "spectral_power": SPECTRAL_DIMENSION,
    }
