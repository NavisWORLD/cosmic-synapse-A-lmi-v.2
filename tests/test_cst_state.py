import json
import math

from cosmic_synapse.cst_state import CSTParameters, CSTState


def test_cst_state_is_bounded_and_memory_tracks_adaptive_state():
    params = CSTParameters(k=0.7, gamma=0.25, alpha=0.4, sync_strength=0.15)
    state = CSTState(seed=7, params=params)
    initial_memory = state.m12

    for _ in range(500):
        state.step(dt=0.01, omega=3.0, audio_energy=0.8, neighbor_phases=[0.3, 1.1])

    assert -1.0 <= state.x12 <= 1.0
    assert state.m12 != initial_memory
    assert abs(state.m12 - state.x12) < 1.0
    assert 0.0 <= state.phase < 2 * math.pi
    assert state.energy >= 0.0
    assert 0.0 <= state.entropy <= 1.0


def test_cst_state_serialization_round_trip_preserves_internal_state():
    state = CSTState(seed=11)
    state.step(dt=0.02, omega=0.9, audio_energy=0.25, neighbor_phases=[0.1])
    encoded = state.to_json()
    restored = CSTState.from_json(encoded)

    assert json.loads(restored.to_json()) == json.loads(encoded)
    assert restored.params == state.params
