//! Append-safe, order-preserving native memory ledger.

use almi_core::{canonical_json_bytes, AlmiError, MemoryRecord, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendResult {
    pub index: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerVerification {
    pub records: usize,
    pub digests: Vec<String>,
}

pub fn append_record(path: impl AsRef<Path>, record: &MemoryRecord) -> Result<AppendResult> {
    record.validate()?;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let line = canonical_json_bytes(record)?;
    let index = if path.exists() { count_records(path)? } else { 0 };
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(&line)?;
    file.flush()?;
    Ok(AppendResult { index, sha256: digest(&line) })
}

pub fn verify_ledger(path: impl AsRef<Path>) -> Result<LedgerVerification> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| AlmiError::Integrity("memory ledger is not valid UTF-8".into()))?;
    let mut records = 0usize;
    let mut digests = Vec::new();
    for raw in text.lines() {
        if raw.trim().is_empty() { continue; }
        let record: MemoryRecord = serde_json::from_str(raw).map_err(|e| AlmiError::Integrity(format!("memory ledger contains invalid JSONL: {e}")))?;
        record.validate()?;
        let mut original = raw.as_bytes().to_vec();
        original.push(b'\n');
        digests.push(digest(&original));
        records += 1;
    }
    Ok(LedgerVerification { records, digests })
}

pub fn read_records(path: impl AsRef<Path>) -> Result<Vec<MemoryRecord>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for raw in text.lines() {
        if raw.trim().is_empty() { continue; }
        let record: MemoryRecord = serde_json::from_str(raw).map_err(|e| AlmiError::Integrity(format!("memory ledger contains invalid JSONL: {e}")))?;
        record.validate()?;
        out.push(record);
    }
    Ok(out)
}

fn count_records(path: &Path) -> Result<usize> {
    Ok(verify_ledger(path)?.records)
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
