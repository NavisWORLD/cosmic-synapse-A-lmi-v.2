# Native Benchmarking

Performance results are evidence about a specific machine and run, not a product ranking or a claim that Rust is universally faster than Python.

## Harness

CI builds `almi_bench` in release mode and runs `scripts/benchmark_native.py`. Both sides time library operations inside one process so CLI/process startup is excluded.

The synthetic workload covers:

- workspace initialization
- `.cosmos` bundle export
- `.cosmos` bundle verification
- `.cosmos` bundle import
- memory append
- validation/scan of a 100-record memory ledger
- CST state serialization

Each operation records sample count, median microseconds, and p95 microseconds. CI also records platform, machine architecture, Python version, CPU count, and iteration count.

## Reproduce

```text
python -m pip install -e .
cargo build --locked --release --manifest-path native/almi-core-rs/Cargo.toml -p almi-cli --bin almi_bench
python scripts/benchmark_native.py \
  --rust-bench native/almi-core-rs/target/release/almi_bench \
  --iterations 25 \
  --output native/almi-core-rs/dist/benchmarks/local.json
```

Windows users may substitute the `.exe` path.

## Interpretation limits

Filesystem caches, runner load, CPU scaling, antivirus, storage, Python/Rust toolchain versions, and input sizes can materially change results. CI therefore publishes the raw JSON artifact and does not enforce a performance threshold.

`memory_scan_100` validates the same synthetic JSONL corpus on both sides but is not a microarchitectural equivalence proof. The benchmark intentionally avoids manipulating iteration counts or inputs to make either implementation look better.
