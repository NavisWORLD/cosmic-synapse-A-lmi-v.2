import json
import os
from pathlib import Path
import subprocess

import pytest

from a_lmi.continuity import initialize_workspace
from cosmic_synapse.cst_state import CSTEvent, CSTState


NATIVE = os.environ.get("ALMI_NATIVE_CLI")


def _native(*args: str) -> dict:
    assert NATIVE, "ALMI_NATIVE_CLI must point to the built native CLI"
    result = subprocess.run([NATIVE, "--json", *args], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_python_persisted_cst_state_replays_identically_in_rust(tmp_path: Path):
    workspace = tmp_path / "python-state"
    initialize_workspace(workspace, name="CST Oracle", seed=4242)
    state_path = workspace / "state" / "cst.json"
    initial = json.loads(state_path.read_text(encoding="utf-8"))

    events = [
        {"dt": 0.1, "omega": 2.0, "audio_energy": 0.4, "neighbor_phases": [0.2, 0.8]},
        {"dt": 0.05, "omega": -0.3, "audio_energy": 0.0, "neighbor_phases": []},
        {"dt": 0.2, "omega": 0.7, "audio_energy": 0.9, "neighbor_phases": [1.1, 2.2, 3.3]},
    ]
    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps(events), encoding="utf-8")

    python_state = CSTState.from_dict(initial)
    expected = []
    for raw in events:
        event = CSTEvent(
            dt=raw["dt"],
            omega=raw["omega"],
            audio_energy=raw["audio_energy"],
            neighbor_phases=tuple(raw["neighbor_phases"]),
        )
        expected.append(
            python_state.step(
                dt=event.dt,
                omega=event.omega,
                audio_energy=event.audio_energy,
                neighbor_phases=event.neighbor_phases,
            )
        )

    result = _native("state", "replay", str(state_path), str(events_path))
    actual = result["snapshots"]
    assert len(actual) == len(expected)
    for rust_snapshot, python_snapshot in zip(actual, expected, strict=True):
        assert rust_snapshot["step"] == python_snapshot["step"]
        for key in ("x12", "m12", "omega", "phase", "energy", "entropy"):
            assert rust_snapshot[key] == pytest.approx(python_snapshot[key], rel=0.0, abs=1e-12)
