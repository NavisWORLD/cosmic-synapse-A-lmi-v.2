use almi_continuity::initialize_workspace;
use almi_core::{canonical_json_bytes, MemoryRecord};
use almi_cosmos::{export_bundle, import_bundle, verify_bundle};
use almi_memory::{append_record, verify_ledger};
use almi_state::NativeCstState;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn percentile(values: &[f64], percentile: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| a.total_cmp(b));
    if ordered.is_empty() {
        return 0.0;
    }
    let rank = ((percentile * ordered.len() as f64).ceil() as usize).saturating_sub(1);
    ordered[rank.min(ordered.len() - 1)]
}

fn summarize(samples: Vec<f64>) -> Value {
    let mut ordered = samples.clone();
    ordered.sort_by(|a, b| a.total_cmp(b));
    let median = if ordered.is_empty() {
        0.0
    } else if ordered.len() % 2 == 0 {
        let mid = ordered.len() / 2;
        (ordered[mid - 1] + ordered[mid]) / 2.0
    } else {
        ordered[ordered.len() / 2]
    };
    json!({
        "samples": samples.len(),
        "median_us": median,
        "p95_us": percentile(&samples, 0.95),
    })
}

fn timed<T>(operation: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let result = operation();
    (result, start.elapsed().as_secs_f64() * 1_000_000.0)
}

fn temp_root() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("almi-native-bench-{}-{nonce}", std::process::id()))
}

fn record(index: usize) -> MemoryRecord {
    MemoryRecord {
        version: Some(1),
        role: "user".into(),
        content: format!("synthetic benchmark record {index}"),
        provider: None,
        provenance: None,
        timestamp: None,
        extra: BTreeMap::new(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(25)
        .clamp(3, 200);
    let base = temp_root();
    fs::create_dir_all(&base)?;

    let result = (|| -> Result<Value, Box<dyn std::error::Error>> {
        let mut metrics = serde_json::Map::new();

        let mut samples = Vec::with_capacity(iterations);
        for index in 0..iterations {
            let workspace = base.join(format!("init-{index}"));
            let (outcome, elapsed) =
                timed(|| initialize_workspace(&workspace, "Rust Benchmark", index as i64));
            outcome?;
            samples.push(elapsed);
        }
        metrics.insert("workspace_init".into(), summarize(samples));

        let source = base.join("source");
        initialize_workspace(&source, "Rust Benchmark", 7)?;

        let mut samples = Vec::with_capacity(iterations);
        for index in 0..iterations {
            let bundle = base.join(format!("export-{index}.cosmos"));
            let (outcome, elapsed) = timed(|| export_bundle(&source, &bundle));
            outcome?;
            samples.push(elapsed);
        }
        metrics.insert("bundle_export".into(), summarize(samples));

        let verify_bundle_path = base.join("verify.cosmos");
        export_bundle(&source, &verify_bundle_path)?;
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let (outcome, elapsed) = timed(|| verify_bundle(&verify_bundle_path));
            outcome?;
            samples.push(elapsed);
        }
        metrics.insert("bundle_verify".into(), summarize(samples));

        let mut samples = Vec::with_capacity(iterations);
        for index in 0..iterations {
            let destination = base.join(format!("import-{index}"));
            let (outcome, elapsed) = timed(|| import_bundle(&verify_bundle_path, &destination));
            outcome?;
            samples.push(elapsed);
        }
        metrics.insert("bundle_import".into(), summarize(samples));

        let memory_workspace = base.join("memory");
        initialize_workspace(&memory_workspace, "Memory Benchmark", 8)?;
        let ledger = memory_workspace.join("memory/ledger.jsonl");
        let mut samples = Vec::with_capacity(iterations);
        for index in 0..iterations {
            let item = record(index);
            let (outcome, elapsed) = timed(|| append_record(&ledger, &item));
            outcome?;
            samples.push(elapsed);
        }
        metrics.insert("memory_append".into(), summarize(samples));

        let scan_workspace = base.join("scan");
        initialize_workspace(&scan_workspace, "Scan Benchmark", 9)?;
        let scan_ledger = scan_workspace.join("memory/ledger.jsonl");
        for index in 0..100 {
            append_record(&scan_ledger, &record(index))?;
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let (outcome, elapsed) = timed(|| verify_ledger(&scan_ledger));
            outcome?;
            samples.push(elapsed);
        }
        metrics.insert("memory_scan_100".into(), summarize(samples));

        let state = NativeCstState::new(10);
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let (outcome, elapsed) = timed(|| canonical_json_bytes(state.envelope()));
            outcome?;
            samples.push(elapsed);
        }
        metrics.insert("state_serialize".into(), summarize(samples));

        Ok(json!({
            "implementation": "rust-native-library",
            "iterations": iterations,
            "timing_unit": "microseconds",
            "metrics": metrics,
            "methodology": "Release binary; operations timed inside one process; filesystem operations use synthetic temporary data.",
        }))
    })();

    let _ = fs::remove_dir_all(&base);
    println!("{}", serde_json::to_string(&result?)?);
    Ok(())
}
