from cosmic_synapse.cst_state import CSTEngine, CSTEvent


def events():
    return [
        CSTEvent(dt=0.01, omega=0.2, audio_energy=0.0, neighbor_phases=(0.1, 0.2)),
        CSTEvent(dt=0.01, omega=0.5, audio_energy=0.4, neighbor_phases=(0.3, 0.6)),
        CSTEvent(dt=0.02, omega=-0.1, audio_energy=0.9, neighbor_phases=(1.2,)),
        CSTEvent(dt=0.01, omega=0.8, audio_energy=0.2, neighbor_phases=()),
    ]


def test_same_seed_and_events_produce_identical_replay():
    first = CSTEngine(seed=2026).replay(events())
    second = CSTEngine(seed=2026).replay(events())
    assert first == second


def test_different_seed_changes_seeded_initial_phase_but_not_contract_shape():
    first = CSTEngine(seed=1).replay(events())
    second = CSTEngine(seed=2).replay(events())
    assert first != second
    assert set(first[-1]) == {"x12", "m12", "omega", "phase", "energy", "entropy", "step"}
