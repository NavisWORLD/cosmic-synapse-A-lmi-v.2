# LightToken native benchmark methodology

The authoritative numerical oracle remains the preserved Python LightToken
implementation. Rust owns native validation, scoring, and orchestration; C++
provides optional self-tested acceleration. These measurements compare
implementations on synthetic 1536-element embeddings with 769 embedding-spectrum
bins. They do **not** measure physical frequency or provide general performance
rankings.

## Reproduce

```bash
python -m pip install -e .
python scripts/generate_lighttoken_fixtures.py --output tests/fixtures/lighttoken
cmake -S native/lighttoken-cpp -B native/lighttoken-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/lighttoken-cpp/build --config Release
ctest --test-dir native/lighttoken-cpp/build --output-on-failure
cargo build --release --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-cli --bin lighttoken_bench
LIGHTTOKEN_CPP_LIB="$PWD/native/lighttoken-cpp/build/liblighttoken_accel.so" \
python scripts/benchmark_lighttoken_native.py \
  --rust-bench native/lighttoken-rs/target/release/lighttoken_bench \
  --fixtures tests/fixtures/lighttoken \
  --iterations 25 \
  --output dist/lighttoken/benchmarks/local.json
```

For Windows and macOS, substitute the correct executable and accelerator library
names. Missing or failed C++ self-test should be reported as Rust fallback, not
as an accelerated result.

## Measurement boundaries

Python and Rust token-level calls both derive spectral power and compute a score.
Rust scalar kernel uses precomputed spectral power. C++ dispatch also receives
precomputed power and includes dynamic-library loading and ABI/self-test checking
in its per-call measurement. Timings are not identical operation boundaries;
compare methods only with those caveats.

One warmup precedes each series. The report records iteration count, median and
p95 microseconds, platform/CPU metadata, score parity, and C++ availability.
The output is descriptive for that runner and synthetic input, not a universal
superiority or production-throughput claim. Raw JSON evidence is retained by CI.
