use almi_core::{CSTStateEnvelope, CstParameters, CstSnapshot};
use almi_state::{CstEvent, NativeCstState};

fn fixture() -> CSTStateEnvelope {
    CSTStateEnvelope {
        version: 1,
        seed: 7,
        params: CstParameters::default(),
        state: CstSnapshot { x12: 0.0, m12: 0.0, omega: 0.0, phase: 1.25, energy: 0.0, entropy: 1.0, step: 0 },
    }
}

#[test]
fn replay_matches_python_equations_from_canonical_envelope() {
    let mut state = NativeCstState::from_envelope(fixture()).unwrap();
    let snapshot = state.step(&CstEvent { dt: 0.1, omega: 2.0, audio_energy: 0.4, neighbor_phases: vec![0.2, 0.8] }).unwrap();
    let expected_x12 = 0.105;
    let expected_m12 = 0.00315;
    assert!((snapshot.x12 - expected_x12).abs() < 1e-12);
    assert!((snapshot.m12 - expected_m12).abs() < 1e-12);
    assert_eq!(snapshot.step, 1);
    assert!((0.0..=1.0).contains(&snapshot.entropy));
}

#[test]
fn unknown_state_version_fails_closed() {
    let mut env = fixture();
    env.version = 99;
    assert!(NativeCstState::from_envelope(env).is_err());
}
