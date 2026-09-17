//! Native LightToken schema and Python-compatible wire contract.
//!
//! The preserved Python implementation in `a_lmi/core/light_token.py` remains
//! the compatibility oracle. This crate provides strict native parsing,
//! validation, and canonical active-form serialization without assigning any
//! physical meaning to embedding spectral bins.

use almi_core::normalize_json;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use std::collections::BTreeMap;
use thiserror::Error;

pub const LIGHTTOKEN_SCHEMA_VERSION: u32 = 1;
pub const EMBEDDING_DIMENSION: usize = 1536;
pub const SPECTRAL_DIMENSION: usize = EMBEDDING_DIMENSION / 2 + 1;
pub const SPECTRAL_TRANSFORM: &str = "embedding_rfft";
pub const LEGACY_SPECTRAL_TRANSFORM: &str = "legacy_embedding_fft";
pub const LEGACY_FULL_FFT_DIMENSION: usize = EMBEDDING_DIMENSION;

#[derive(Debug, Error)]
pub enum LightTokenError {
    #[error("invalid LightToken: {0}")]
    Invalid(String),
    #[error("unsupported LightToken representation: {0}")]
    Unsupported(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, LightTokenError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityMethod {
    PowerCorrelation,
    Cosine,
    Euclidean,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComplexBin {
    pub real: f32,
    pub imag: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LightTokenRecord {
    pub token_id: String,
    pub timestamp: String,
    pub source_uri: String,
    pub modality: String,
    pub raw_data_ref: String,
    pub content_text: Option<String>,
    pub metadata: BTreeMap<String, Value>,
    pub perceptual_hash: Option<String>,
    pub joint_embedding: Option<Vec<f32>>,
    pub spectral_signature: Option<Vec<ComplexBin>>,
}

#[derive(Debug, Deserialize)]
struct WireToken {
    token_id: String,
    timestamp: String,
    source_uri: String,
    modality: String,
    raw_data_ref: String,
    #[serde(default)]
    content_text: Option<String>,
    #[serde(default)]
    metadata: BTreeMap<String, Value>,
    #[serde(default)]
    perceptual_hash: Option<String>,
    #[serde(default)]
    joint_embedding: Option<Vec<f64>>,
    #[serde(default)]
    spectral_signature_real: Option<Vec<f64>>,
    #[serde(default)]
    spectral_signature_imag: Option<Vec<f64>>,
    #[serde(default)]
    spectral_signature_magnitude: Option<Vec<f64>>,
    #[serde(default)]
    spectral_signature_phase: Option<Vec<f64>>,
}

fn finite_f32(value: f64, label: &str) -> Result<f32> {
    if !value.is_finite() {
        return Err(LightTokenError::Invalid(format!(
            "{label} contains a non-finite value"
        )));
    }
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(LightTokenError::Invalid(format!(
            "{label} value is outside finite float32 range"
        )));
    }
    Ok(narrowed)
}

fn finite_vec(values: Vec<f64>, label: &str) -> Result<Vec<f32>> {
    values
        .into_iter()
        .map(|value| finite_f32(value, label))
        .collect()
}

fn label_spectral_shape(metadata: &mut BTreeMap<String, Value>, size: usize) {
    let (transform, dimension) = if size == SPECTRAL_DIMENSION {
        (SPECTRAL_TRANSFORM, SPECTRAL_DIMENSION)
    } else {
        (LEGACY_SPECTRAL_TRANSFORM, size)
    };
    metadata
        .entry("spectral_transform".into())
        .or_insert_with(|| Value::String(transform.into()));
    metadata
        .entry("spectral_dimension".into())
        .or_insert_with(|| Value::Number(Number::from(dimension as u64)));
}

fn parse_spectrum(wire: &WireToken, metadata: &mut BTreeMap<String, Value>) -> Result<Option<Vec<ComplexBin>>> {
    let has_active = wire.spectral_signature_real.is_some() || wire.spectral_signature_imag.is_some();
    let has_legacy = wire.spectral_signature_magnitude.is_some() || wire.spectral_signature_phase.is_some();

    if has_active && has_legacy {
        return Err(LightTokenError::Invalid(
            "spectral payload cannot contain both active real/imag and historical magnitude/phase forms".into(),
        ));
    }

    if has_active {
        let real = wire.spectral_signature_real.as_ref().ok_or_else(|| {
            LightTokenError::Invalid("spectral real/imag components are incomplete".into())
        })?;
        let imag = wire.spectral_signature_imag.as_ref().ok_or_else(|| {
            LightTokenError::Invalid("spectral real/imag components are incomplete".into())
        })?;
        if real.len() != imag.len() {
            return Err(LightTokenError::Invalid(
                "spectral real/imag component lengths do not match".into(),
            ));
        }
        if real.len() != SPECTRAL_DIMENSION {
            return Err(LightTokenError::Invalid(format!(
                "active spectral payload must contain {SPECTRAL_DIMENSION} bins, got {}",
                real.len()
            )));
        }
        let mut bins = Vec::with_capacity(real.len());
        for (index, (&r, &i)) in real.iter().zip(imag.iter()).enumerate() {
            bins.push(ComplexBin {
                real: finite_f32(r, &format!("spectral real component {index}"))?,
                imag: finite_f32(i, &format!("spectral imag component {index}"))?,
            });
        }
        label_spectral_shape(metadata, bins.len());
        return Ok(Some(bins));
    }

    if has_legacy {
        let magnitude = wire.spectral_signature_magnitude.as_ref().ok_or_else(|| {
            LightTokenError::Invalid("historical spectral magnitude/phase components are incomplete".into())
        })?;
        let phase = wire.spectral_signature_phase.as_ref().ok_or_else(|| {
            LightTokenError::Invalid("historical spectral magnitude/phase components are incomplete".into())
        })?;
        if magnitude.len() != phase.len() {
            return Err(LightTokenError::Invalid(
                "historical spectral magnitude/phase lengths do not match".into(),
            ));
        }
        if magnitude.len() != SPECTRAL_DIMENSION && magnitude.len() != LEGACY_FULL_FFT_DIMENSION {
            return Err(LightTokenError::Unsupported(format!(
                "historical spectral payload length {}; supported lengths are {SPECTRAL_DIMENSION} and {LEGACY_FULL_FFT_DIMENSION}",
                magnitude.len()
            )));
        }
        let mut bins = Vec::with_capacity(magnitude.len());
        for (index, (&magnitude, &phase)) in magnitude.iter().zip(phase.iter()).enumerate() {
            if !magnitude.is_finite() || !phase.is_finite() {
                return Err(LightTokenError::Invalid(format!(
                    "historical spectral component {index} contains a non-finite value"
                )));
            }
            let real = finite_f32(magnitude * phase.cos(), "historical spectral real component")?;
            let imag = finite_f32(magnitude * phase.sin(), "historical spectral imag component")?;
            bins.push(ComplexBin { real, imag });
        }
        label_spectral_shape(metadata, bins.len());
        return Ok(Some(bins));
    }

    Ok(None)
}

pub fn from_json_bytes(bytes: &[u8]) -> Result<LightTokenRecord> {
    let wire: WireToken = serde_json::from_slice(bytes)?;
    let mut metadata = wire.metadata.clone();
    let joint_embedding = match wire.joint_embedding.clone() {
        Some(values) => Some(finite_vec(values, "joint_embedding")?),
        None => None,
    };
    let spectral_signature = parse_spectrum(&wire, &mut metadata)?;

    let token = LightTokenRecord {
        token_id: wire.token_id,
        timestamp: wire.timestamp,
        source_uri: wire.source_uri,
        modality: wire.modality,
        raw_data_ref: wire.raw_data_ref,
        content_text: wire.content_text,
        metadata,
        perceptual_hash: wire.perceptual_hash,
        joint_embedding,
        spectral_signature,
    };
    token.validate()?;
    Ok(token)
}

impl LightTokenRecord {
    pub fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("token_id", self.token_id.as_str()),
            ("timestamp", self.timestamp.as_str()),
            ("source_uri", self.source_uri.as_str()),
            ("modality", self.modality.as_str()),
            ("raw_data_ref", self.raw_data_ref.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(LightTokenError::Invalid(format!(
                    "{label} must not be empty"
                )));
            }
        }

        if let Some(embedding) = &self.joint_embedding {
            if embedding.len() != EMBEDDING_DIMENSION {
                return Err(LightTokenError::Invalid(format!(
                    "joint_embedding must contain {EMBEDDING_DIMENSION} values, got {}",
                    embedding.len()
                )));
            }
            if embedding.iter().any(|value| !value.is_finite()) {
                return Err(LightTokenError::Invalid(
                    "joint_embedding contains a non-finite value".into(),
                ));
            }
        }

        if let Some(spectrum) = &self.spectral_signature {
            if spectrum.len() != SPECTRAL_DIMENSION && spectrum.len() != LEGACY_FULL_FFT_DIMENSION {
                return Err(LightTokenError::Invalid(format!(
                    "spectral payload has unsupported length {}",
                    spectrum.len()
                )));
            }
            if spectrum
                .iter()
                .any(|bin| !bin.real.is_finite() || !bin.imag.is_finite())
            {
                return Err(LightTokenError::Invalid(
                    "spectral payload contains a non-finite value".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn spectral_transform(&self) -> &str {
        self.metadata
            .get("spectral_transform")
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                if self
                    .spectral_signature
                    .as_ref()
                    .is_some_and(|bins| bins.len() == SPECTRAL_DIMENSION)
                {
                    SPECTRAL_TRANSFORM
                } else {
                    LEGACY_SPECTRAL_TRANSFORM
                }
            })
    }

    pub fn spectral_dimension(&self) -> Option<usize> {
        self.spectral_signature.as_ref().map(Vec::len)
    }

    /// Serialize the active Python-compatible LightToken representation.
    ///
    /// Historical full-FFT records are deliberately read-only: writers only
    /// emit the active 769-bin real/imag representation.
    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        if self
            .spectral_signature
            .as_ref()
            .is_some_and(|bins| bins.len() != SPECTRAL_DIMENSION)
        {
            return Err(LightTokenError::Unsupported(
                "historical spectral records are read-only; active writers emit 769-bin embedding_rfft only".into(),
            ));
        }

        let mut object = Map::new();
        object.insert("token_id".into(), Value::String(self.token_id.clone()));
        object.insert("timestamp".into(), Value::String(self.timestamp.clone()));
        object.insert("source_uri".into(), Value::String(self.source_uri.clone()));
        object.insert("modality".into(), Value::String(self.modality.clone()));
        object.insert("raw_data_ref".into(), Value::String(self.raw_data_ref.clone()));
        object.insert(
            "content_text".into(),
            self.content_text
                .as_ref()
                .map(|value| Value::String(value.clone()))
                .unwrap_or(Value::Null),
        );
        object.insert("metadata".into(), serde_json::to_value(&self.metadata)?);
        object.insert(
            "perceptual_hash".into(),
            self.perceptual_hash
                .as_ref()
                .map(|value| Value::String(value.clone()))
                .unwrap_or(Value::Null),
        );

        if let Some(embedding) = &self.joint_embedding {
            object.insert(
                "joint_embedding".into(),
                Value::Array(
                    embedding
                        .iter()
                        .map(|&value| float_value(value))
                        .collect::<Result<Vec<_>>>()?,
                ),
            );
        }

        if let Some(spectrum) = &self.spectral_signature {
            object.insert(
                "spectral_signature_real".into(),
                Value::Array(
                    spectrum
                        .iter()
                        .map(|bin| float_value(bin.real))
                        .collect::<Result<Vec<_>>>()?,
                ),
            );
            object.insert(
                "spectral_signature_imag".into(),
                Value::Array(
                    spectrum
                        .iter()
                        .map(|bin| float_value(bin.imag))
                        .collect::<Result<Vec<_>>>()?,
                ),
            );
        }

        let normalized = normalize_json(Value::Object(object));
        let mut bytes = serde_json::to_vec(&normalized)?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

fn float_value(value: f32) -> Result<Value> {
    if !value.is_finite() {
        return Err(LightTokenError::Invalid(
            "cannot serialize non-finite float".into(),
        ));
    }
    let number = Number::from_f64(value as f64).ok_or_else(|| {
        LightTokenError::Invalid("cannot encode float as JSON number".into())
    })?;
    Ok(Value::Number(number))
}
