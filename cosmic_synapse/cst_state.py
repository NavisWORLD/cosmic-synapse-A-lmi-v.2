"""Canonical active CST computational-state adapter.

This module provides a small, deterministic engineering interface for the
adaptive-state ideas that appear throughout the historical 8D/12D Cosmic
Synapse lineage. ``x12`` and ``m12`` are internal dimensionless software
state variables. The name "12D" is retained as historical terminology; this
module does not assert an additional physical dimension or new physics.

Historical engines remain preserved in their original repository locations.
This adapter exists so tests, examples, and integrations have one stable,
serializable state contract without rewriting those artifacts.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

_STATE_VERSION = 1


@dataclass(frozen=True)
class CSTParameters:
    """Parameters for the bounded adaptive computational state."""

    k: float = 0.5
    gamma: float = 0.2
    alpha: float = 0.3
    sync_strength: float = 0.1
    audio_gain: float = 0.25
    natural_frequency: float = 1.0

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.gamma < 0 or self.alpha < 0:
            raise ValueError("gamma and alpha must be non-negative")


@dataclass(frozen=True)
class CSTEvent:
    """One deterministic input frame for replay."""

    dt: float
    omega: float
    audio_energy: float = 0.0
    neighbor_phases: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be a positive finite value")
        if not math.isfinite(self.omega):
            raise ValueError("omega must be finite")
        if not math.isfinite(self.audio_energy) or self.audio_energy < 0:
            raise ValueError("audio_energy must be finite and non-negative")
        for phase in self.neighbor_phases:
            if not math.isfinite(phase):
                raise ValueError("neighbor phases must be finite")


class CSTState:
    """Bounded adaptive state with memory, phase, energy, and entropy signals."""

    def __init__(
        self,
        seed: int = 0,
        params: CSTParameters | None = None,
        *,
        x12: float = 0.0,
        m12: float = 0.0,
        omega: float = 0.0,
        phase: float | None = None,
        energy: float = 0.0,
        entropy: float = 1.0,
        step_index: int = 0,
    ) -> None:
        self.seed = int(seed)
        self.params = params or CSTParameters()
        self._rng = np.random.default_rng(self.seed)

        self.x12 = float(np.clip(x12, -1.0, 1.0))
        self.m12 = float(m12)
        self.omega = float(omega)
        self.phase = (
            float(self._rng.uniform(0.0, 2.0 * math.pi))
            if phase is None
            else float(phase) % (2.0 * math.pi)
        )
        self.energy = max(0.0, float(energy))
        self.entropy = float(np.clip(entropy, 0.0, 1.0))
        self.step_index = int(step_index)
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")

    def step(
        self,
        *,
        dt: float,
        omega: float,
        audio_energy: float = 0.0,
        neighbor_phases: Sequence[float] = (),
    ) -> dict[str, float | int]:
        """Advance one deterministic state frame.

        The update intentionally mirrors the historical bounded adaptive form:
        ``dx12/dt = k*Omega - gamma*x12`` and
        ``dm12/dt = alpha*(x12-m12)``. Audio energy is an explicit software
        input gain, not a claim about a physical extra dimension.
        """

        event = CSTEvent(
            dt=float(dt),
            omega=float(omega),
            audio_energy=float(audio_energy),
            neighbor_phases=tuple(float(value) for value in neighbor_phases),
        )
        effective_omega = event.omega + self.params.audio_gain * event.audio_energy

        dx12 = (
            self.params.k * effective_omega - self.params.gamma * self.x12
        ) * event.dt
        self.x12 = float(np.clip(self.x12 + dx12, -1.0, 1.0))

        dm12 = self.params.alpha * (self.x12 - self.m12) * event.dt
        self.m12 = float(self.m12 + dm12)

        if event.neighbor_phases:
            coupling = sum(
                math.sin(phase - self.phase) for phase in event.neighbor_phases
            ) / len(event.neighbor_phases)
        else:
            coupling = 0.0
        phase_velocity = (
            self.params.natural_frequency + self.params.sync_strength * coupling
        )
        self.phase = float(
            (self.phase + event.dt * phase_velocity) % (2.0 * math.pi)
        )

        self.omega = float(effective_omega)
        # This is a diagnostic computational energy, not physical energy.
        self.energy = float(
            0.5 * (self.x12 * self.x12 + self.m12 * self.m12 + self.omega * self.omega)
            + event.audio_energy
        )
        self.entropy = self._binary_state_entropy(self.x12)
        self.step_index += 1
        return self.snapshot()

    @staticmethod
    def _binary_state_entropy(x12: float) -> float:
        probability = float(np.clip((x12 + 1.0) * 0.5, 0.0, 1.0))
        if probability in (0.0, 1.0):
            return 0.0
        other = 1.0 - probability
        value = -(
            probability * math.log2(probability) + other * math.log2(other)
        )
        return float(np.clip(value, 0.0, 1.0))

    def snapshot(self) -> dict[str, float | int]:
        """Return the public active-state contract used by replay/tests."""

        return {
            "x12": self.x12,
            "m12": self.m12,
            "omega": self.omega,
            "phase": self.phase,
            "energy": self.energy,
            "entropy": self.entropy,
            "step": self.step_index,
        }

    def to_dict(self) -> dict:
        return {
            "version": _STATE_VERSION,
            "seed": self.seed,
            "params": asdict(self.params),
            "state": self.snapshot(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict) -> "CSTState":
        version = int(data.get("version", 0))
        if version != _STATE_VERSION:
            raise ValueError(f"Unsupported CST state version: {version}")
        state = data["state"]
        params = CSTParameters(**data["params"])
        return cls(
            seed=int(data["seed"]),
            params=params,
            x12=float(state["x12"]),
            m12=float(state["m12"]),
            omega=float(state["omega"]),
            phase=float(state["phase"]),
            energy=float(state["energy"]),
            entropy=float(state["entropy"]),
            step_index=int(state["step"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> "CSTState":
        return cls.from_dict(json.loads(payload))


class CSTEngine:
    """Deterministic event/replay wrapper around :class:`CSTState`."""

    def __init__(self, seed: int = 0, params: CSTParameters | None = None) -> None:
        self.seed = int(seed)
        self.params = params or CSTParameters()
        self.state = CSTState(seed=self.seed, params=self.params)

    def reset(self) -> CSTState:
        self.state = CSTState(seed=self.seed, params=self.params)
        return self.state

    def replay(self, events: Iterable[CSTEvent]) -> list[dict[str, float | int]]:
        """Replay inputs from a fresh seeded state and return every snapshot."""

        self.reset()
        snapshots: list[dict[str, float | int]] = []
        for event in events:
            if not isinstance(event, CSTEvent):
                raise TypeError("replay expects CSTEvent instances")
            snapshots.append(
                self.state.step(
                    dt=event.dt,
                    omega=event.omega,
                    audio_energy=event.audio_energy,
                    neighbor_phases=event.neighbor_phases,
                )
            )
        return snapshots
