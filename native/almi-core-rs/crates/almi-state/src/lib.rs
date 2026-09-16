//! Canonical native CST computational-state adapter.
//! `12D` is retained only as historical/project terminology; these are bounded
//! software state variables and this crate makes no new-physics claim.

use almi_core::{AlmiError, CSTStateEnvelope, CstParameters, CstSnapshot, Result, CST_STATE_VERSION};
use serde::Deserialize;
use std::f64::consts::TAU;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CstEvent {
    pub dt: f64,
    pub omega: f64,
    #[serde(default)]
    pub audio_energy: f64,
    #[serde(default)]
    pub neighbor_phases: Vec<f64>,
}

impl CstEvent {
    pub fn validate(&self) -> Result<()> {
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(AlmiError::InvalidInput("dt must be a positive finite value".into()));
        }
        if !self.omega.is_finite() {
            return Err(AlmiError::InvalidInput("omega must be finite".into()));
        }
        if !self.audio_energy.is_finite() || self.audio_energy < 0.0 {
            return Err(AlmiError::InvalidInput("audio_energy must be finite and non-negative".into()));
        }
        if self.neighbor_phases.iter().any(|v| !v.is_finite()) {
            return Err(AlmiError::InvalidInput("neighbor phases must be finite".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NativeCstState {
    envelope: CSTStateEnvelope,
}

impl NativeCstState {
    /// Deterministic native initialization. Exact cross-language replay parity is
    /// anchored by persisted canonical envelopes, so provider/language changes
    /// never regenerate an existing phase from seed.
    pub fn new(seed: i64) -> Self {
        let phase = phase_from_seed(seed);
        Self {
            envelope: CSTStateEnvelope {
                version: CST_STATE_VERSION,
                seed,
                params: CstParameters::default(),
                state: CstSnapshot {
                    x12: 0.0,
                    m12: 0.0,
                    omega: 0.0,
                    phase,
                    energy: 0.0,
                    entropy: 1.0,
                    step: 0,
                },
            },
        }
    }

    pub fn from_envelope(envelope: CSTStateEnvelope) -> Result<Self> {
        envelope.validate_version()?;
        validate_parameters(&envelope.params)?;
        validate_snapshot(&envelope.state)?;
        Ok(Self { envelope })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path)?;
        let envelope: CSTStateEnvelope = serde_json::from_slice(&bytes)?;
        Self::from_envelope(envelope)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, almi_core::canonical_json_bytes(&self.envelope)?)?;
        Ok(())
    }

    pub fn envelope(&self) -> &CSTStateEnvelope {
        &self.envelope
    }

    pub fn into_envelope(self) -> CSTStateEnvelope {
        self.envelope
    }

    pub fn snapshot(&self) -> CstSnapshot {
        self.envelope.state.clone()
    }

    pub fn step(&mut self, event: &CstEvent) -> Result<CstSnapshot> {
        event.validate()?;
        let params = &self.envelope.params;
        let state = &mut self.envelope.state;
        let effective_omega = event.omega + params.audio_gain * event.audio_energy;

        let dx12 = (params.k * effective_omega - params.gamma * state.x12) * event.dt;
        state.x12 = (state.x12 + dx12).clamp(-1.0, 1.0);
        let dm12 = params.alpha * (state.x12 - state.m12) * event.dt;
        state.m12 += dm12;

        let coupling = if event.neighbor_phases.is_empty() {
            0.0
        } else {
            event.neighbor_phases.iter().map(|phase| (phase - state.phase).sin()).sum::<f64>()
                / event.neighbor_phases.len() as f64
        };
        let phase_velocity = params.natural_frequency + params.sync_strength * coupling;
        state.phase = (state.phase + event.dt * phase_velocity).rem_euclid(TAU);
        state.omega = effective_omega;
        state.energy = 0.5 * (state.x12 * state.x12 + state.m12 * state.m12 + state.omega * state.omega)
            + event.audio_energy;
        state.entropy = binary_state_entropy(state.x12);
        state.step = state.step.checked_add(1).ok_or_else(|| AlmiError::Integrity("CST step counter overflow".into()))?;
        Ok(state.clone())
    }

    pub fn replay(&mut self, events: &[CstEvent]) -> Result<Vec<CstSnapshot>> {
        let mut out = Vec::with_capacity(events.len());
        for event in events {
            out.push(self.step(event)?);
        }
        Ok(out)
    }

    pub fn equivalent(&self, other: &Self, tolerance: f64) -> bool {
        if self.envelope.version != other.envelope.version || self.envelope.seed != other.envelope.seed {
            return false;
        }
        let a = &self.envelope.state;
        let b = &other.envelope.state;
        a.step == b.step
            && [a.x12-b.x12,a.m12-b.m12,a.omega-b.omega,a.phase-b.phase,a.energy-b.energy,a.entropy-b.entropy]
                .into_iter().all(|v| v.abs() <= tolerance)
    }
}

fn validate_parameters(params: &CstParameters) -> Result<()> {
    let values = [params.k, params.gamma, params.alpha, params.sync_strength, params.audio_gain, params.natural_frequency];
    if values.into_iter().any(|v| !v.is_finite()) {
        return Err(AlmiError::Integrity("CST parameters must be finite".into()));
    }
    if params.gamma < 0.0 || params.alpha < 0.0 {
        return Err(AlmiError::Integrity("CST gamma and alpha must be non-negative".into()));
    }
    Ok(())
}

fn validate_snapshot(state: &CstSnapshot) -> Result<()> {
    if [state.x12,state.m12,state.omega,state.phase,state.energy,state.entropy].into_iter().any(|v| !v.is_finite()) {
        return Err(AlmiError::Integrity("CST state values must be finite".into()));
    }
    if !(-1.0..=1.0).contains(&state.x12) || !(0.0..=1.0).contains(&state.entropy) || state.energy < 0.0 {
        return Err(AlmiError::Integrity("CST state is outside canonical bounds".into()));
    }
    Ok(())
}

fn binary_state_entropy(x12: f64) -> f64 {
    let probability = ((x12 + 1.0) * 0.5).clamp(0.0, 1.0);
    if probability == 0.0 || probability == 1.0 {
        return 0.0;
    }
    let other = 1.0 - probability;
    (-(probability * probability.log2() + other * other.log2())).clamp(0.0, 1.0)
}

fn phase_from_seed(seed: i64) -> f64 {
    // Stable local initializer for *new Rust-created* workspaces. Existing
    // Python-created workspaces carry their canonical persisted phase.
    let mut x = (seed as u64) ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let unit = (x as f64) / (u64::MAX as f64);
    unit * TAU
}
