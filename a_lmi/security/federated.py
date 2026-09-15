"""Federated-update utilities with precise privacy/security terminology.

This module implements plain weighted averaging and optional *experimental*
Gaussian noise injection. It does not implement a formal differential-privacy
mechanism or cryptographic secure aggregation. Historical method names remain
as compatibility aliases, but their docstrings and warnings state the actual
properties.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


class FederatedLearning:
    """Coordinator for weighted client-update aggregation experiments."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.clients: list[dict[str, Any]] = []
        self.global_model = None

    def add_client(self, client_id: str, model: Any) -> None:
        self.clients.append({"id": client_id, "model": model, "updates": 0})

    @staticmethod
    def privacy_properties() -> Dict[str, bool]:
        return {
            "formal_differential_privacy": False,
            "cryptographic_secure_aggregation": False,
            "experimental_noise_injection": True,
        }

    @staticmethod
    def weighted_average(
        client_updates: Sequence[Mapping[str, Any]],
        weights: Sequence[float] | None = None,
    ) -> Dict[str, Any]:
        """Compute a transparent weighted average of matching model updates."""

        if not client_updates:
            raise ValueError("No client updates provided")
        keys = tuple(client_updates[0].keys())
        if not keys:
            return {}
        for update in client_updates[1:]:
            if tuple(update.keys()) != keys:
                raise ValueError("All client updates must contain the same ordered keys")

        if weights is None:
            normalized = np.full(len(client_updates), 1.0 / len(client_updates))
        else:
            if len(weights) != len(client_updates):
                raise ValueError("Weights and updates must have the same length")
            raw = np.asarray(weights, dtype=np.float64)
            if not np.all(np.isfinite(raw)) or np.any(raw < 0):
                raise ValueError("Weights must be finite and non-negative")
            total = float(raw.sum())
            if total <= 0:
                raise ValueError("Weights must sum to a positive value")
            normalized = raw / total

        aggregated: Dict[str, Any] = {}
        for key in keys:
            value = None
            for update, weight in zip(client_updates, normalized):
                term = update[key] * float(weight)
                value = term if value is None else value + term
            aggregated[key] = value
        return aggregated

    @staticmethod
    def add_experimental_gaussian_noise(
        model_update: Mapping[str, Any],
        *,
        stddev: float = 1.0,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        """Add reproducible Gaussian noise for experiments.

        This is *not* a formal differential-privacy mechanism: there is no
        clipping/sensitivity proof, delta/accounting, or validated privacy
        accountant in this implementation.
        """

        if not np.isfinite(stddev) or stddev < 0:
            raise ValueError("stddev must be finite and non-negative")
        rng = np.random.default_rng(seed)
        noisy: Dict[str, Any] = {}
        for key, value in model_update.items():
            # Native NumPy arrays remain NumPy. Torch-like tensors are supported
            # when available without importing torch into the minimal package.
            if isinstance(value, np.ndarray):
                noise = rng.normal(0.0, stddev, size=value.shape).astype(value.dtype)
                noisy[key] = value + noise
                continue
            shape = tuple(getattr(value, "shape", ()))
            if not shape:
                raise TypeError(f"Unsupported update value for {key!r}: {type(value)!r}")
            noise_np = rng.normal(0.0, stddev, size=shape)
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "Non-NumPy tensor noise injection requires the optional ML dependencies"
                ) from exc
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unsupported update value for {key!r}: {type(value)!r}")
            noise = torch.as_tensor(noise_np, dtype=value.dtype, device=value.device)
            noisy[key] = value + noise
        return noisy

    @classmethod
    def aggregate_with_experimental_noise(
        cls,
        client_updates: Sequence[Mapping[str, Any]],
        *,
        weights: Sequence[float] | None = None,
        noise_stddev: float = 0.0,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        averaged = cls.weighted_average(client_updates, weights=weights)
        if noise_stddev == 0:
            return averaged
        return cls.add_experimental_gaussian_noise(
            averaged, stddev=noise_stddev, seed=seed
        )

    # Historical compatibility aliases. These intentionally warn because the
    # old names overstated what the implementation provided.
    def federated_averaging(
        self,
        client_updates: List[Dict[str, Any]],
        weights: List[float] | None = None,
    ) -> Dict[str, Any]:
        return self.weighted_average(client_updates, weights=weights)

    def apply_differential_privacy(
        self,
        model_update: Dict[str, Any],
        sensitivity: float = 1.0,
        epsilon: float = 1.0,
    ) -> Dict[str, Any]:
        warnings.warn(
            "apply_differential_privacy is a historical misnomer; this only adds "
            "experimental Gaussian noise and does not provide a formal DP guarantee",
            DeprecationWarning,
            stacklevel=2,
        )
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return self.add_experimental_gaussian_noise(
            model_update, stddev=float(sensitivity) / float(epsilon)
        )

    def secure_aggregation(
        self,
        client_updates: List[Dict[str, Any]],
        enable_dp: bool = True,
    ) -> Dict[str, Any]:
        warnings.warn(
            "secure_aggregation is a historical misnomer; client updates are not "
            "cryptographically hidden from the aggregator",
            DeprecationWarning,
            stacklevel=2,
        )
        averaged = self.weighted_average(client_updates)
        if enable_dp:
            return self.add_experimental_gaussian_noise(averaged, stddev=1.0)
        return averaged
