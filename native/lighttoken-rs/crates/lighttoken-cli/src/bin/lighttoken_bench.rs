//! Descriptive synthetic LightToken measurements. No speedup threshold or claim.
use lighttoken_core::{from_json_bytes, SimilarityMethod};
use lighttoken_spectrum::{
    backend_diagnostics, similarity, similarity_many, spectral_power, token_similarity,
};
use serde_json::{json, Map, Value};
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

fn summarize(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let median = if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    };
    let p95 = values[(95 * n).div_ceil(100).saturating_sub(1).min(n - 1)];
    json!({"samples": n, "median_us": median, "p95_us": p95})
}

fn measure(mut operation: impl FnMut() -> f32, iterations: usize) -> Value {
    black_box(operation());
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(operation());
        samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    summarize(samples)
}

fn run() -> Result<Value, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: lighttoken_bench <synthetic-fixture-directory> <iterations>".into());
    }
    let iterations: usize = args[2].parse()?;
    if !(3..=200).contains(&iterations) {
        return Err("iterations must be between 3 and 200".into());
    }
    let dir = Path::new(&args[1]);
    let left = from_json_bytes(&fs::read(dir.join("active_random.json"))?)?;
    let right = from_json_bytes(&fs::read(dir.join("active_sinusoid.json"))?)?;
    let query = spectral_power(left.spectral_signature.as_deref().ok_or("missing spectrum")?)?;
    let candidate =
        spectral_power(right.spectral_signature.as_deref().ok_or("missing spectrum")?)?;
    let candidates = vec![candidate.clone()];
    let diagnostics = backend_diagnostics();
    let mut methods = Map::new();
    for (name, method) in [
        ("power_correlation", SimilarityMethod::PowerCorrelation),
        ("cosine", SimilarityMethod::Cosine),
        ("euclidean", SimilarityMethod::Euclidean),
    ] {
        let token_score = token_similarity(&left, &right, method)?;
        let scalar_score = similarity(&query, &candidate, method)?;
        if (token_score - scalar_score).abs() > 0.0001 {
            return Err(format!("{name}: token/scalar parity failed").into());
        }
        let cpp_score = if diagnostics.cpp_available {
            let value = similarity_many(&query, &candidates, method)?[0];
            if (value - scalar_score).abs() > 0.0001 {
                return Err(format!("{name}: C++/Rust parity failed").into());
            }
            Some(value)
        } else {
            None
        };
        let rust_token = measure(
            || token_similarity(&left, &right, method).expect("validated token"),
            iterations,
        );
        let rust_scalar = measure(
            || similarity(&query, &candidate, method).expect("validated power"),
            iterations,
        );
        let cpp_dispatch = if cpp_score.is_some() {
            Some(measure(
                || similarity_many(&query, &candidates, method).expect("self-tested library")[0],
                iterations,
            ))
        } else {
            None
        };
        methods.insert(
            name.into(),
            json!({
                "score": token_score,
                "rust_token": rust_token,
                "rust_scalar_kernel": rust_scalar,
                "cpp_load_and_dispatch": cpp_dispatch,
                "cpp_score": cpp_score,
            }),
        );
    }
    Ok(json!({
        "schema_version": 1,
        "input": "synthetic_python_oracle_fixture",
        "iterations": iterations,
        "cpu_parallelism": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "rust_version": env!("CARGO_PKG_VERSION"),
        "cpp_available": diagnostics.cpp_available,
        "cpp_detail": diagnostics.detail,
        "methods": methods,
        "interpretation": "Per-run descriptive timings only. Rust token-level versus Rust scalar-kernel and C++ dynamic-load/dispatch have distinct operation boundaries."
    }))
}

fn main() {
    match run() {
        Ok(value) => println!("{}", serde_json::to_string_pretty(&value).expect("serialize benchmark")),
        Err(error) => {
            eprintln!("lighttoken benchmark failed: {error}");
            std::process::exit(1);
        }
    }
}
