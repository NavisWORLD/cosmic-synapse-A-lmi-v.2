//! Native LightToken spectral computation and similarity semantics.
//!
//! Spectral bins are a software transform over embedding coordinates. They are
//! not physical frequencies unless an external mapping is separately defined.

use lighttoken_core::{
    ComplexBin, LightTokenRecord, SimilarityMethod, EMBEDDING_DIMENSION, SPECTRAL_DIMENSION,
};
use rustfft::{num_complex::Complex, FftPlanner};
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SpectrumError {
    #[error("invalid spectral input: {0}")]
    InvalidInput(String),
    #[error("LightToken {0} has no spectral signature")]
    MissingSpectrum(String),
}

pub type Result<T> = std::result::Result<T, SpectrumError>;

fn validate_finite(values: &[f32], label: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SpectrumError::InvalidInput(format!(
            "{label} contains a non-finite value"
        )));
    }
    Ok(())
}

/// Compute the active 769-bin one-sided forward FFT for a 1536-value embedding.
pub fn rfft_embedding(input: &[f32]) -> Result<Vec<ComplexBin>> {
    if input.len() != EMBEDDING_DIMENSION {
        return Err(SpectrumError::InvalidInput(format!(
            "embedding must contain {EMBEDDING_DIMENSION} values, got {}",
            input.len()
        )));
    }
    validate_finite(input, "embedding")?;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(EMBEDDING_DIMENSION);
    let mut buffer: Vec<Complex<f32>> = input
        .iter()
        .copied()
        .map(|real| Complex { real, im: 0.0 })
        .collect();
    fft.process(&mut buffer);

    Ok(buffer
        .into_iter()
        .take(SPECTRAL_DIMENSION)
        .map(|bin| ComplexBin {
            real: bin.re,
            imag: bin.im,
        })
        .collect())
}

/// Return magnitude of each complex spectral bin as finite `f32` values.
pub fn spectral_power(bins: &[ComplexBin]) -> Result<Vec<f32>> {
    if bins.is_empty() {
        return Err(SpectrumError::InvalidInput(
            "spectral payload must not be empty".into(),
        ));
    }
    let mut power = Vec::with_capacity(bins.len());
    for (index, bin) in bins.iter().enumerate() {
        if !bin.real.is_finite() || !bin.imag.is_finite() {
            return Err(SpectrumError::InvalidInput(format!(
                "spectral bin {index} contains a non-finite value"
            )));
        }
        let magnitude = bin.real.hypot(bin.imag);
        if !magnitude.is_finite() {
            return Err(SpectrumError::InvalidInput(format!(
                "spectral magnitude {index} is non-finite"
            )));
        }
        power.push(magnitude);
    }
    Ok(power)
}

/// Return the first maximum bin index and its magnitude, matching NumPy argmax.
pub fn dominant_bin(power: &[f32]) -> Result<(usize, f32)> {
    if power.is_empty() {
        return Err(SpectrumError::InvalidInput(
            "spectral power must not be empty".into(),
        ));
    }
    validate_finite(power, "spectral power")?;

    let mut best_index = 0usize;
    let mut best_value = power[0];
    for (index, &value) in power.iter().enumerate().skip(1) {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    Ok((best_index, best_value))
}

fn arrays_equal(left: &[f32], right: &[f32]) -> bool {
    left == right
}

fn vector_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&value| {
            let value = f64::from(value);
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

/// Compare two spectral-power arrays using the preserved Python semantics.
pub fn similarity(left: &[f32], right: &[f32], method: SimilarityMethod) -> Result<f32> {
    if left.len() != right.len() {
        return Err(SpectrumError::InvalidInput(format!(
            "spectral shapes must match, got {} and {}",
            left.len(),
            right.len()
        )));
    }
    if left.is_empty() {
        return Err(SpectrumError::InvalidInput(
            "spectral power arrays must not be empty".into(),
        ));
    }
    validate_finite(left, "left spectral power")?;
    validate_finite(right, "right spectral power")?;

    let score = match method {
        SimilarityMethod::PowerCorrelation => {
            let count = left.len() as f64;
            let mean_left = left.iter().map(|&value| f64::from(value)).sum::<f64>() / count;
            let mean_right = right.iter().map(|&value| f64::from(value)).sum::<f64>() / count;

            let mut numerator = 0.0f64;
            let mut square_left = 0.0f64;
            let mut square_right = 0.0f64;
            for (&left_value, &right_value) in left.iter().zip(right.iter()) {
                let centered_left = f64::from(left_value) - mean_left;
                let centered_right = f64::from(right_value) - mean_right;
                numerator += centered_left * centered_right;
                square_left += centered_left * centered_left;
                square_right += centered_right * centered_right;
            }
            if square_left == 0.0 || square_right == 0.0 {
                if arrays_equal(left, right) {
                    1.0
                } else {
                    0.0
                }
            } else {
                numerator / (square_left.sqrt() * square_right.sqrt())
            }
        }
        SimilarityMethod::Cosine => {
            let numerator = left
                .iter()
                .zip(right.iter())
                .map(|(&left_value, &right_value)| f64::from(left_value) * f64::from(right_value))
                .sum::<f64>();
            let denominator = vector_norm(left) * vector_norm(right);
            if denominator == 0.0 {
                if arrays_equal(left, right) {
                    1.0
                } else {
                    0.0
                }
            } else {
                numerator / denominator
            }
        }
        SimilarityMethod::Euclidean => {
            let distance = left
                .iter()
                .zip(right.iter())
                .map(|(&left_value, &right_value)| {
                    let delta = f64::from(left_value) - f64::from(right_value);
                    delta * delta
                })
                .sum::<f64>()
                .sqrt();
            let max_distance = vector_norm(left) + vector_norm(right);
            if max_distance == 0.0 {
                1.0
            } else {
                1.0 - (distance / max_distance)
            }
        }
    };

    if !score.is_finite() {
        return Err(SpectrumError::InvalidInput(
            "similarity computation produced a non-finite score".into(),
        ));
    }
    Ok(score as f32)
}

/// Compare the stored spectra of two LightTokens.
pub fn token_similarity(
    left: &LightTokenRecord,
    right: &LightTokenRecord,
    method: SimilarityMethod,
) -> Result<f32> {
    let left_bins = left
        .spectral_signature
        .as_deref()
        .ok_or_else(|| SpectrumError::MissingSpectrum(left.token_id.clone()))?;
    let right_bins = right
        .spectral_signature
        .as_deref()
        .ok_or_else(|| SpectrumError::MissingSpectrum(right.token_id.clone()))?;
    let left_power = spectral_power(left_bins)?;
    let right_power = spectral_power(right_bins)?;
    similarity(&left_power, &right_power, method)
}
