# LightToken Native Workstation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real cross-language LightToken workstation where Python remains the compatibility oracle, Rust owns canonical native behavior, C++ optionally accelerates batch similarity, JavaFX provides the functional desktop application, SQLite stores rebuildable workstation metadata, and Windows is a first-class packaged lifecycle target.

**Architecture:** The implementation follows `JavaFX desktop → JNI → Rust LightToken engine → optional C++ acceleration`, with the existing Python `a_lmi/core/light_token.py` preserved as the oracle. Rust reuses the already-verified A-LMI native continuity/`.cosmos` crates by path dependency; Java never owns the token wire format and C++ never owns persistence or archive parsing.

**Tech Stack:** Python 3.11+, Rust edition 2021 / rust-version 1.82+, `rustfft = "6"`, Serde, Clap, JNI crate 0.21, C++17 + CMake 3.24+, JDK 21, OpenJFX 21.0.9, Gradle 8.14.5, Xerial SQLite JDBC 3.53.4.0, JUnit 5, Java `jlink`/`jpackage`, GitHub Actions Linux/macOS/Windows.

**Spec:** `docs/superpowers/specs/2026-09-17-lighttoken-native-workstation-design.md`

## Global Constraints

- Preserve `a_lmi/core/light_token.py`; it is the compatibility/reference oracle.
- Active embedding dimension is exactly `1536`; active one-sided real FFT dimension is exactly `769`.
- Active transform label is `embedding_rfft`; spectral-bin indices are software-domain bins, not physical frequencies.
- New writes reject NaN and infinity.
- Python, Rust, C++, and Java must agree on supported similarity semantics within committed measured tolerances.
- C++ is optional; every accelerated operation has a Rust fallback and backend self-test.
- Java calls Rust through JNI; Java must never load a C++ path selected from token metadata.
- Source tokens are read-only by default; SQLite stores rebuildable workstation metadata/cache only.
- `.cosmos` verification/import reuses `native/almi-core-rs` verified crates/surfaces; no second archive parser.
- Windows x86_64 is the first packaged desktop target; normal install is user-local and requires no administrator privileges.
- Update/uninstall preserve workspaces, `.cosmos` bundles, source tokens, exports, SQLite user data, and backups by default.
- No consciousness, biological, physical-resonance, quantum, security-certification, or unmeasured performance claims.
- No later repo project (multimodal, HRCS, God Music, Unity) is mixed into this implementation branch.

## File Structure

### Existing files modified

- `tests/test_light_token_contract.py` — extend oracle contract coverage without changing existing semantics.
- `.gitignore` — ignore new Rust/C++/Gradle/Java packaging build products while keeping fixture/evidence files tracked.
- `README.md` — add the final LightToken workstation entry point only after executable surfaces are proven.
- `TESTING.md` — document exact LightToken CI gates after they exist.
- `docs/CLAIMS_AND_LIMITATIONS.md` — record software-only LightToken claims/boundaries.

### Python oracle and fixtures

- `scripts/generate_lighttoken_fixtures.py` — deterministic synthetic fixture generator using the preserved Python implementation.
- `tests/fixtures/lighttoken/manifest.json` — fixture names, hashes, dimensions, tolerance provenance.
- `tests/fixtures/lighttoken/*.json` — fixed active and historical-form token fixtures.
- `tests/fixtures/lighttoken/similarity.json` — Python oracle pairwise expected scores.
- `tests/test_lighttoken_native_interop.py` — Python↔Rust CLI parity and golden-vector checks.

### Rust

- `native/lighttoken-rs/Cargo.toml` / `Cargo.lock` — isolated LightToken Rust workspace.
- `native/lighttoken-rs/crates/lighttoken-core/` — schema, validation, canonical serialization.
- `native/lighttoken-rs/crates/lighttoken-spectrum/` — FFT, spectral power, similarity, backend dispatch.
- `native/lighttoken-rs/crates/lighttoken-index/` — deterministic collection/search/index contract.
- `native/lighttoken-rs/crates/lighttoken-io/` — JSON/JSONL/workspace/verified `.cosmos` read adapters.
- `native/lighttoken-rs/crates/lighttoken-ffi/` — stable C ABI plus JNI adapter.
- `native/lighttoken-rs/crates/lighttoken-cli/` — automation CLI and benchmark binary.
- `native/lighttoken-rs/include/lighttoken.h` — stable public C header.
- `native/lighttoken-rs/fuzz/` — JSON/collection fuzz targets.

### C++

- `native/lighttoken-cpp/CMakeLists.txt` — portable build.
- `native/lighttoken-cpp/include/lighttoken_accel.h` — narrow C-compatible kernel ABI.
- `native/lighttoken-cpp/src/lighttoken_accel.cpp` — scalar/batched numerical implementation.
- `native/lighttoken-cpp/tests/` — known-vector and malformed-input tests.

### Java

- `apps/lighttoken-workstation-java/settings.gradle`
- `apps/lighttoken-workstation-java/build.gradle`
- `apps/lighttoken-workstation-java/gradle/wrapper/*`
- `apps/lighttoken-workstation-java/src/main/java/world/navis/lighttoken/...` — native bridge, repository, services, JavaFX UI.
- `apps/lighttoken-workstation-java/src/main/resources/world/navis/lighttoken/...` — FXML/CSS if used.
- `apps/lighttoken-workstation-java/src/test/java/world/navis/lighttoken/...` — headless and controller/view-model tests.

### Windows + CI + docs

- `INSTALL_LIGHTTOKEN_WINDOWS.bat`, `BUILD_LIGHTTOKEN_WINDOWS.bat`, `RUN_LIGHTTOKEN_WINDOWS.bat`, `TEST_LIGHTTOKEN_WINDOWS.bat`, `VERIFY_LIGHTTOKEN_WINDOWS.bat`, `UPDATE_LIGHTTOKEN_WINDOWS.bat`, `UNINSTALL_LIGHTTOKEN_WINDOWS.bat`.
- `scripts/windows/lighttoken/common.ps1`, `build.ps1`, `install.ps1`, `run.ps1`, `test.ps1`, `verify.ps1`, `update.ps1`, `uninstall.ps1`.
- `.github/workflows/lighttoken-native-ci.yml`.
- `scripts/benchmark_lighttoken_native.py`.
- `docs/lighttoken/ARCHITECTURE.md`, `INTEROPERABILITY.md`, `WINDOWS_INSTALL.md`, `SECURITY.md`, `BENCHMARKS.md`, `VERIFICATION.md`.
- `LIGHTTOKEN_NATIVE_WORKSTATION_FINAL_REPORT.md` after definitive merge evidence exists.

---

### Task 1: Freeze the Python oracle and generate RED golden fixtures

**Files:**
- Modify: `tests/test_light_token_contract.py`
- Create: `scripts/generate_lighttoken_fixtures.py`
- Create: `tests/fixtures/lighttoken/manifest.json`
- Create: `tests/fixtures/lighttoken/active_zero.json`
- Create: `tests/fixtures/lighttoken/active_constant.json`
- Create: `tests/fixtures/lighttoken/active_impulse.json`
- Create: `tests/fixtures/lighttoken/active_sinusoid.json`
- Create: `tests/fixtures/lighttoken/active_random.json`
- Create: `tests/fixtures/lighttoken/historical_magnitude_phase.json`
- Create: `tests/fixtures/lighttoken/similarity.json`
- Create: `tests/test_lighttoken_native_interop.py`

**Interfaces:**
- Consumes: `a_lmi.core.light_token.LightToken`, `spectral_similarity`, `resonance_match`.
- Produces: immutable synthetic fixture corpus keyed by fixed token IDs and `similarity.json` expected values; `LIGHTTOKEN_NATIVE_CLI` test environment variable.

- [ ] **Step 1: Extend the Python contract tests for all three similarity methods and degenerate behavior**

```python
from a_lmi.core.light_token import LightToken, spectral_similarity


def test_similarity_degenerate_contract():
    a = make_token()
    b = make_token()
    zeros = np.zeros(1536, dtype=np.float32)
    a.set_embedding(zeros)
    b.set_embedding(zeros)
    assert spectral_similarity(a, b, "power_correlation") == 1.0
    assert spectral_similarity(a, b, "cosine") == 1.0
    assert spectral_similarity(a, b, "euclidean") == 1.0
```

- [ ] **Step 2: Run the Python oracle test and verify it is green before native work**

Run: `python -m pytest -q tests/test_light_token_contract.py`

Expected: PASS. If it fails, stop and resolve the oracle contradiction before generating native fixtures.

- [ ] **Step 3: Implement the deterministic synthetic fixture generator**

Use fixed UUID strings and timestamp `2000-01-01T00:00:00+00:00`. Generate arrays with NumPy `float32`: zero, constant `0.25`, impulse at index 17, `sin(2π*index/64)`, and `np.random.default_rng(424242).standard_normal(1536).astype(np.float32)`. Override `token_id` and `timestamp` before serialization. Write Python canonical JSON using `token.to_json()` plus a newline. Include SHA-256 for each fixture in `manifest.json`.

- [ ] **Step 4: Generate fixtures and verify they contain no user/private data**

Run: `python scripts/generate_lighttoken_fixtures.py --output tests/fixtures/lighttoken`

Expected: files above produced; `manifest.json` reports embedding dimension 1536 and spectral dimension 769.

- [ ] **Step 5: Write RED Python↔Rust CLI interop tests**

```python
def test_rust_cli_validates_python_fixture(native_cli, fixture_dir):
    result = subprocess.run(
        [native_cli, "validate", str(fixture_dir / "active_random.json"), "--json"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["embedding_dimension"] == 1536
    assert payload["spectral_dimension"] == 769
```

Also add RED checks for inspect, spectrum, compare, historical reader compatibility, and canonical JSON round-trip hash.

- [ ] **Step 6: Run the native interop test and verify expected RED failure**

Run: `LIGHTTOKEN_NATIVE_CLI=/nonexistent/lighttoken python -m pytest -q tests/test_lighttoken_native_interop.py`

Expected: FAIL because the native CLI does not exist.

- [ ] **Step 7: Commit the oracle lock and RED evidence**

```bash
git add tests/test_light_token_contract.py scripts/generate_lighttoken_fixtures.py tests/fixtures/lighttoken tests/test_lighttoken_native_interop.py
git commit -m "test(lighttoken): freeze Python oracle and native RED fixtures"
```

---

### Task 2: Create the Rust workspace and canonical LightToken schema

**Files:**
- Create: `native/lighttoken-rs/Cargo.toml`
- Create: `native/lighttoken-rs/crates/lighttoken-core/Cargo.toml`
- Create: `native/lighttoken-rs/crates/lighttoken-core/src/lib.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-core/tests/python_fixtures.rs`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `almi-core::canonical_json_bytes` by path dependency `../almi-core-rs/crates/almi-core`.
- Produces:
  - `pub const LIGHTTOKEN_SCHEMA_VERSION: u32 = 1;`
  - `pub const EMBEDDING_DIMENSION: usize = 1536;`
  - `pub const SPECTRAL_DIMENSION: usize = 769;`
  - `pub struct ComplexBin { pub real: f32, pub imag: f32 }`
  - `pub struct LightTokenRecord`
  - `pub enum SimilarityMethod { PowerCorrelation, Cosine, Euclidean }`
  - `impl LightTokenRecord { pub fn validate(&self) -> Result<()>; pub fn canonical_json_bytes(&self) -> Result<Vec<u8>>; }`

- [ ] **Step 1: Write Rust fixture parsing tests before implementing the schema**

```rust
#[test]
fn parses_active_python_fixture() {
    let token = lighttoken_core::from_json_bytes(include_bytes!(
        "../../../../../tests/fixtures/lighttoken/active_random.json"
    ))
    .unwrap();
    assert_eq!(token.joint_embedding.as_ref().unwrap().len(), EMBEDDING_DIMENSION);
    assert_eq!(token.spectral_signature.as_ref().unwrap().len(), SPECTRAL_DIMENSION);
    token.validate().unwrap();
}
```

- [ ] **Step 2: Run the crate test and verify RED compile failure**

Run: `cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-core`

Expected: FAIL because types/functions are not defined.

- [ ] **Step 3: Implement the workspace and schema**

Use `BTreeMap<String, serde_json::Value>` for metadata so canonical serialization is stable. Model real/imag components as separate serialized vectors matching Python keys `spectral_signature_real` and `spectral_signature_imag`; expose them internally as `Vec<ComplexBin>` after deserialization.

Validation rules:
- source/modality/raw reference non-empty;
- embedding length exactly 1536 if present;
- active real/imag lengths exactly 769 and equal;
- legacy magnitude/phase lengths equal and finite;
- all floats finite;
- unsupported/incomplete spectral forms fail closed.

- [ ] **Step 4: Re-run Rust fixture tests**

Run: `cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-core`

Expected: PASS.

- [ ] **Step 5: Add canonical serialization hash parity against Python fixture bytes**

The test must parse `active_random.json`, write canonical JSON using `almi_core::normalize_json`/`canonical_json_bytes`, and compare the exact SHA-256 from fixture `manifest.json`.

- [ ] **Step 6: Run format, Clippy, and tests**

```bash
cargo fmt --manifest-path native/lighttoken-rs/Cargo.toml --all -- --check
cargo clippy --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-core --all-targets -- -D warnings
cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-core
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add native/lighttoken-rs .gitignore
git commit -m "feat(lighttoken): add native Rust schema contract"
```

---

### Task 3: Implement Rust FFT, spectral power, and Python similarity parity

**Files:**
- Create: `native/lighttoken-rs/crates/lighttoken-spectrum/Cargo.toml`
- Create: `native/lighttoken-rs/crates/lighttoken-spectrum/src/lib.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-spectrum/src/rust_backend.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-spectrum/tests/python_parity.rs`
- Modify: `native/lighttoken-rs/Cargo.toml`

**Interfaces:**
- Consumes: `LightTokenRecord`, `ComplexBin`, `SimilarityMethod`.
- Produces:
  - `pub fn rfft_embedding(input: &[f32]) -> Result<Vec<ComplexBin>>`
  - `pub fn spectral_power(bins: &[ComplexBin]) -> Result<Vec<f32>>`
  - `pub fn dominant_bin(power: &[f32]) -> Result<(usize, f32)>`
  - `pub fn similarity(a: &[f32], b: &[f32], method: SimilarityMethod) -> Result<f32>`
  - `pub fn token_similarity(a: &LightTokenRecord, b: &LightTokenRecord, method: SimilarityMethod) -> Result<f32>`

- [ ] **Step 1: Write Python-fixture FFT parity tests**

Use fixture real/imag arrays. Start with tolerances `abs <= 2e-4`, `rel <= 2e-5`; measure actual differences and only tighten or justify widening with a committed fixture/tolerance update.

- [ ] **Step 2: Run tests and verify RED**

Run: `cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-spectrum`

Expected: FAIL due to missing implementation.

- [ ] **Step 3: Implement `rustfft` RFFT behavior**

Construct a `Vec<Complex<f32>>` from 1536 real values with zero imaginary part, run forward FFT, and retain bins `0..=1536/2`. Do not normalize, matching `numpy.fft.rfft` forward convention.

- [ ] **Step 4: Implement exact Python degenerate similarity semantics**

Power correlation computes mean/std/covariance; if either standard deviation is zero, return 1 only for exact equal arrays, else 0. Cosine and normalized euclidean must follow the same special cases as the Python oracle.

- [ ] **Step 5: Compare every fixed fixture and expected score**

Run: `cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-spectrum -- --nocapture`

Expected: PASS and printed measured max absolute/relative differences remain within committed tolerances.

- [ ] **Step 6: Add property tests**

Properties: identical finite vectors produce similarity 1; cosine symmetric; euclidean similarity symmetric and within expected numerical bounds for finite fixtures; RFFT output length always 769 for valid embeddings.

- [ ] **Step 7: Commit**

```bash
git add native/lighttoken-rs
git commit -m "feat(lighttoken): add Rust spectral and similarity parity"
```

---

### Task 4: Implement deterministic collection index, search, JSON/JSONL IO, and A-LMI read adapters

**Files:**
- Create: `native/lighttoken-rs/crates/lighttoken-index/Cargo.toml`
- Create: `native/lighttoken-rs/crates/lighttoken-index/src/lib.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-io/Cargo.toml`
- Create: `native/lighttoken-rs/crates/lighttoken-io/src/lib.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-io/src/collection.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-io/src/almi.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-index/tests/search.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-io/tests/io.rs`
- Modify: `native/lighttoken-rs/Cargo.toml`

**Interfaces:**
- Produces:
  - `pub struct TokenCollection { ... }`
  - `pub struct SearchHit { pub token_id: String, pub score: f32, pub rank: usize }`
  - `pub struct SearchRequest { pub method: SimilarityMethod, pub top_k: Option<usize>, pub threshold: Option<f32>, pub modality: Option<String>, pub source_prefix: Option<String> }`
  - `pub fn search(&self, query: &LightTokenRecord, request: &SearchRequest) -> Result<Vec<SearchHit>>`
  - `pub fn load_token_file(path: &Path) -> Result<LightTokenRecord>`
  - `pub fn load_collection(path: &Path) -> Result<Vec<LightTokenRecord>>`
  - `pub fn discover_almi_workspace(path: &Path) -> Result<Vec<LightTokenSource>>`
  - `pub fn import_verified_cosmos(bundle: &Path, temp_dest: &Path) -> Result<Vec<LightTokenSource>>`

- [ ] **Step 1: Write deterministic ranking tests**

Tie-break equal scores by `token_id` ascending after score descending. Include top-K, threshold, modality, and source-prefix tests.

- [ ] **Step 2: Run tests and verify RED**

Run: `cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-index -p lighttoken-io`

Expected: FAIL due to missing crates/functions.

- [ ] **Step 3: Implement collection/search with hard safe defaults**

Default maximum collection size: 100,000 tokens per load. Reject larger collections unless the CLI/application explicitly lowers/raises a configured bound within a compiled hard ceiling of 1,000,000.

- [ ] **Step 4: Implement JSON/JSONL parsing with file-size bounds**

Default single token JSON maximum 16 MiB; collection JSONL maximum 512 MiB. Read incrementally for JSONL and reject oversized individual lines before parsing.

- [ ] **Step 5: Reuse A-LMI `.cosmos` crates**

Add path dependencies to `../almi-core-rs/crates/almi-cosmos` and `../almi-core-rs/crates/almi-continuity`. `import_verified_cosmos` first calls the existing verifier/import surface into a fresh temporary directory; it must never unzip independently.

- [ ] **Step 6: Test unresolved raw references explicitly**

`LightTokenSource` must distinguish `ReferenceOnly` from `Resolved { path, sha256 }`; tests assert missing referenced files stay unresolved instead of being fabricated.

- [ ] **Step 7: Run tests and commit**

```bash
cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-index -p lighttoken-io
git add native/lighttoken-rs
git commit -m "feat(lighttoken): add deterministic search and safe IO"
```

---

### Task 5: Deliver the native CLI and make Python↔Rust interop GREEN

**Files:**
- Create: `native/lighttoken-rs/crates/lighttoken-cli/Cargo.toml`
- Create: `native/lighttoken-rs/crates/lighttoken-cli/src/main.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-cli/src/output.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-cli/src/bin/lighttoken_bench.rs`
- Modify: `native/lighttoken-rs/Cargo.toml`
- Modify: `tests/test_lighttoken_native_interop.py`

**Interfaces:**
- Binary: `lighttoken`.
- Commands exactly as specified in design: `doctor`, `version`, `inspect`, `validate`, `spectrum`, `compare`, `index build`, `search`, `backend`.
- `--json` returns one compact JSON object/array on stdout; diagnostics go to stderr.

- [ ] **Step 1: Add CLI parser tests for every command**

Use Clap derive and `Cli::try_parse_from` unit tests.

- [ ] **Step 2: Run CLI tests RED**

Run: `cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-cli`

- [ ] **Step 3: Implement minimal commands over core crates**

`doctor --json` reports schema version, dimensions, rust backend availability, optional C++ backend status, and A-LMI integration availability. `compare` must emit token IDs, method, score, and backend.

- [ ] **Step 4: Build debug CLI and run the Python interop suite**

```bash
cargo build --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-cli
LIGHTTOKEN_NATIVE_CLI=$PWD/native/lighttoken-rs/target/debug/lighttoken python -m pytest -q tests/test_lighttoken_native_interop.py
```

Expected: PASS for active fixtures, historical read, spectrum, similarity, and canonical serialization checks.

- [ ] **Step 5: Preserve exact failure categories**

Add tests asserting malformed dimensions/non-finite/partial spectral fields produce non-zero exit and JSON error `category` values rather than panics.

- [ ] **Step 6: Commit**

```bash
git add native/lighttoken-rs tests/test_lighttoken_native_interop.py
git commit -m "feat(lighttoken): add native CLI and Python interoperability"
```

---

### Task 6: Add stable C ABI and JNI-safe Rust engine boundary

**Files:**
- Create: `native/lighttoken-rs/crates/lighttoken-ffi/Cargo.toml`
- Create: `native/lighttoken-rs/crates/lighttoken-ffi/src/lib.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-ffi/src/c_api.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-ffi/src/jni_api.rs`
- Create: `native/lighttoken-rs/include/lighttoken.h`
- Create: `native/lighttoken-rs/tests/c_abi_smoke.c`
- Create: `native/lighttoken-rs/crates/lighttoken-ffi/tests/lifecycle.rs`
- Modify: `native/lighttoken-rs/Cargo.toml`

**Interfaces:**
- `LIGHTTOKEN_ABI_VERSION = 1`.
- Opaque `lighttoken_context`.
- C functions:
  - `lighttoken_abi_version()`
  - `lighttoken_context_new()` / `lighttoken_context_free()`
  - `lighttoken_validate_json()`
  - `lighttoken_compare_json()`
  - `lighttoken_search_json()`
  - `lighttoken_backend_json()`
  - `lighttoken_string_free()`.
- JNI class: `world.navis.lighttoken.nativebridge.NativeEngine` with native `abiVersion`, `createContext`, `freeContext`, `validateJson`, `compareJson`, `searchJson`, `backendJson`.

- [ ] **Step 1: Write C caller and Rust lifecycle tests first**

C smoke must create context, validate `active_random.json`, compare two fixtures, free returned strings/context, and exit zero.

- [ ] **Step 2: Verify RED link/compile**

Run: `cargo build --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-ffi`

Expected: missing symbols/tests fail until implementation exists.

- [ ] **Step 3: Implement panic containment and deterministic status codes**

Every exported FFI function wraps implementation in `catch_unwind`. Public error codes: `0 OK`, `1 INVALID_ARGUMENT`, `2 INVALID_TOKEN`, `3 IO`, `4 UNSUPPORTED_VERSION`, `5 BACKEND`, `255 PANIC_CONTAINED`.

- [ ] **Step 4: Implement JNI wrappers over the same engine functions**

JNI accepts UTF-8 JSON strings and returns compact JSON strings for the first integration increment; large embedding arrays are returned through dedicated `float[]` methods when the Java UI needs chart data, avoiding giant JSON copies for charts.

- [ ] **Step 5: Run Rust lifecycle tests and external C caller**

```bash
cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-ffi
cargo build --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-ffi
cc native/lighttoken-rs/tests/c_abi_smoke.c -I native/lighttoken-rs/include -L native/lighttoken-rs/target/debug -Wl,-rpath,$PWD/native/lighttoken-rs/target/debug -llighttoken_ffi -o /tmp/lighttoken-c-smoke
/tmp/lighttoken-c-smoke tests/fixtures/lighttoken/active_random.json
```

Expected: PASS on Linux CI.

- [ ] **Step 6: Commit**

```bash
git add native/lighttoken-rs
git commit -m "feat(lighttoken): add stable C ABI and JNI boundary"
```

---

### Task 7: Add optional C++ acceleration with self-test and Rust fallback

**Files:**
- Create: `native/lighttoken-cpp/CMakeLists.txt`
- Create: `native/lighttoken-cpp/include/lighttoken_accel.h`
- Create: `native/lighttoken-cpp/src/lighttoken_accel.cpp`
- Create: `native/lighttoken-cpp/tests/test_accel.cpp`
- Create: `native/lighttoken-rs/crates/lighttoken-spectrum/src/cpp_backend.rs`
- Create: `native/lighttoken-rs/crates/lighttoken-spectrum/tests/cpp_parity.rs`
- Modify: `native/lighttoken-rs/crates/lighttoken-spectrum/Cargo.toml`

**Interfaces:**
- C++ C ABI functions:
  - `lt_accel_abi_version()`
  - `lt_accel_self_test()`
  - `lt_accel_spectral_power(...)`
  - `lt_accel_cosine_many(...)`
  - `lt_accel_correlation_many(...)`
  - `lt_accel_euclidean_many(...)`
  - `lt_accel_top_k(...)`.
- Rust `BackendKind::{Rust, Cpp}` and `BackendDiagnostics`.

- [ ] **Step 1: Write C++ known-vector tests and Rust fallback tests**

Fallback test sets `LIGHTTOKEN_DISABLE_CPP=1` and asserts backend is Rust with identical scores.

- [ ] **Step 2: Configure CMake C++17 portable scalar build**

Run: `cmake -S native/lighttoken-cpp -B native/lighttoken-cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build native/lighttoken-cpp/build --config Release`

Expected initially: test/link failures until functions are implemented.

- [ ] **Step 3: Implement scalar C++ kernels first**

Do not add SIMD intrinsics until scalar parity is green. Validate null pointers/lengths and return explicit integer status codes.

- [ ] **Step 4: Load C++ dynamically from Rust using application-owned paths only**

Search order: explicit trusted process configuration set by installer/CI, then native library directory beside Rust JNI/CLI binary. Never read library paths from token metadata/source fields.

- [ ] **Step 5: Require self-test before enabling C++**

`lt_accel_self_test()` computes fixed known vectors. Any ABI mismatch/self-test error leaves the process operational on Rust fallback and exposes reason in `BackendDiagnostics`.

- [ ] **Step 6: Run C++ tests and Rust parity suite**

```bash
ctest --test-dir native/lighttoken-cpp/build --output-on-failure
LIGHTTOKEN_CPP_LIB=<built-library> cargo test --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-spectrum cpp_parity -- --nocapture
```

Expected: all scores within committed tolerances.

- [ ] **Step 7: Commit**

```bash
git add native/lighttoken-cpp native/lighttoken-rs
git commit -m "feat(lighttoken): add optional C++ acceleration backend"
```

---

### Task 8: Bootstrap the Java 21 workstation, JNI bridge, and SQLite library

**Files:**
- Create: `apps/lighttoken-workstation-java/settings.gradle`
- Create: `apps/lighttoken-workstation-java/build.gradle`
- Create: `apps/lighttoken-workstation-java/gradle.properties`
- Create: `apps/lighttoken-workstation-java/gradle/wrapper/gradle-wrapper.properties`
- Create: wrapper scripts/jar through Gradle wrapper generation.
- Create: `apps/lighttoken-workstation-java/src/main/java/world/navis/lighttoken/nativebridge/NativeEngine.java`
- Create: `.../nativebridge/JniNativeEngine.java`
- Create: `.../model/TokenSummary.java`, `SearchHit.java`, `BackendInfo.java`
- Create: `.../library/LibraryDatabase.java`, `LibraryMigrations.java`, `TokenRepository.java`
- Create: `apps/lighttoken-workstation-java/src/test/java/world/navis/lighttoken/...`

**Interfaces:**
- `NativeEngine extends AutoCloseable` with `validateJson`, `compareJson`, `searchJson`, `backendInfo`, chart-array accessors.
- `TokenRepository` operations: `upsertSource`, `listTokens`, `recordQuery`, `listQueries`, `rebuildCache`.
- SQLite schema version 1 tables: `schema_version`, `sources`, `tokens`, `queries`, `query_hits`.

- [ ] **Step 1: Pin build dependencies**

Use Gradle wrapper `8.14.5`, Java toolchain 21, OpenJFX `21.0.9`, Xerial `org.xerial:sqlite-jdbc:3.53.4.0`, JUnit Jupiter. Enable dependency locking with `dependencyLocking { lockAllConfigurations() }` and commit lockfiles.

- [ ] **Step 2: Write RED Java ABI and database migration tests**

```java
@Test
void nativeAbiIsVersionOne() {
    try (NativeEngine engine = JniNativeEngine.load(testNativeLibrary())) {
        assertEquals(1, engine.abiVersion());
    }
}
```

SQLite test opens a temp DB, migrates to schema 1, inserts a synthetic source/token summary, closes/reopens, and asserts persistence.

- [ ] **Step 3: Run Java tests and verify expected RED native-load failure**

Run: `cd apps/lighttoken-workstation-java && ./gradlew test`

Expected: pure SQLite/model tests PASS; JNI integration test RED until native library path is supplied/built.

- [ ] **Step 4: Implement deterministic native library loader**

Accept only an installer/CI system property `lighttoken.native.dir` or application-owned runtime directory. Reject token/source-derived library paths.

- [ ] **Step 5: Implement transactional SQLite migrations with prepared statements**

Before any destructive migration, copy application-owned `library.db` to `backups/library-v<old>-<timestamp>.db`; schema 1 creation itself is non-destructive.

- [ ] **Step 6: Run tests with built JNI library**

Linux example:

```bash
cargo build --manifest-path native/lighttoken-rs/Cargo.toml -p lighttoken-ffi
cd apps/lighttoken-workstation-java
./gradlew test -Dlighttoken.native.dir=../../native/lighttoken-rs/target/debug
```

Expected: JNI ABI + SQLite tests PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/lighttoken-workstation-java
git commit -m "feat(lighttoken): add Java JNI bridge and SQLite library"
```

---

### Task 9: Implement Java application services for real load/search/export

**Files:**
- Create: `apps/lighttoken-workstation-java/src/main/java/world/navis/lighttoken/service/CollectionService.java`
- Create: `.../service/SearchService.java`
- Create: `.../service/ExportService.java`
- Create: `.../service/WorkspaceService.java`
- Create: `.../model/TokenDetail.java`, `QueryResult.java`, `SimilarityMethod.java`
- Create tests under matching `src/test/java` packages.

**Interfaces:**
- `CollectionService.open(Path)` loads via native engine and records source metadata in SQLite.
- `SearchService.search(tokenId, method, topK, threshold, filters)` returns computed native hits and records deterministic query artifact metadata.
- `ExportService.exportToken` and `exportQueryResults` write canonical source/result artifacts only to explicit destinations.
- `WorkspaceService.openWorkspace/openCosmos` are read-only source discovery operations.

- [ ] **Step 1: Write service tests against golden fixture directory**

Tests must use real JNI calls, not mocked scores. Assert the known top result from `similarity.json`, method switching changes/retains ranking according to oracle expectations, and exports include source token IDs/method/scores/backend.

- [ ] **Step 2: Run RED service tests**

Run: `./gradlew test --tests '*ServiceTest' -Dlighttoken.native.dir=...`

- [ ] **Step 3: Implement services with bounded asynchronous execution**

All native collection/search calls run off the JavaFX application thread through a fixed executor sized to `max(1, min(4, availableProcessors/2))`; service objects expose cancellable `CompletableFuture` operations.

- [ ] **Step 4: Ensure source remains read-only**

Tests compare fixture directory hashes before/after load/search and assert no source file changed.

- [ ] **Step 5: Commit**

```bash
git add apps/lighttoken-workstation-java
git commit -m "feat(lighttoken): add real workstation application services"
```

---

### Task 10: Build the functional JavaFX Resonance Explorer UI

**Files:**
- Create: `apps/lighttoken-workstation-java/src/main/java/world/navis/lighttoken/LightTokenWorkstationApp.java`
- Create: `.../ui/MainController.java`
- Create: `.../ui/MainViewModel.java`
- Create: `.../ui/SpectrumChartModel.java`
- Create: `.../ui/EmbeddingChartModel.java`
- Create: `.../ui/SearchResultsModel.java`
- Create: resources `main.fxml` and `workstation.css` if FXML chosen.
- Create controller/view-model tests.

**Interfaces:**
- UI actions invoke Task 9 services only; no scoring math in Java controllers.
- Main regions: source/library browser, token inspector, embedding chart, spectrum overlay chart, search controls/results, status/backend bar.

- [ ] **Step 1: Write view-model tests before rendering code**

Test selection, method switch, top-K/threshold validation, unresolved artifact state, backend label, and export enable/disable states using real service result objects.

- [ ] **Step 2: Run view-model tests RED**

Run: `./gradlew test --tests '*ViewModelTest'`

- [ ] **Step 3: Implement JavaFX application shell and data-bound models**

The chart series must be built from native-returned embedding/spectral arrays. UI labels use “Embedding spectrum” and “Spectral bin”; do not label x-axis Hz/frequency.

- [ ] **Step 4: Implement real search interaction**

Selecting a query token + method + top-K executes Task 9 service; result row click fetches candidate chart data and overlays query/candidate spectra. The status bar shows `Backend: Rust` or `Backend: C++` from native diagnostics.

- [ ] **Step 5: Add an automated headless JavaFX smoke**

Create a test/application mode that initializes the view-model/controller, loads fixture collection, executes one search, asserts chart series lengths `1536` and `769`, then exits without showing fake data.

- [ ] **Step 6: Run Java test suite and commit**

```bash
cd apps/lighttoken-workstation-java
./gradlew test
cd ../..
git add apps/lighttoken-workstation-java
git commit -m "feat(lighttoken): add functional JavaFX resonance explorer"
```

---

### Task 11: Complete verified A-LMI workspace and `.cosmos` read integration

**Files:**
- Modify: `native/lighttoken-rs/crates/lighttoken-io/src/almi.rs`
- Modify: `apps/lighttoken-workstation-java/src/main/java/world/navis/lighttoken/service/WorkspaceService.java`
- Create: `tests/fixtures/lighttoken/almi-workspace/` synthetic fixture workspace.
- Create Rust/Python/Java integration tests.

**Interfaces:**
- `WorkspaceService.openCosmos(Path)` delegates to Rust JNI; Rust delegates to existing `almi-cosmos` verifier/importer.
- Returned source descriptors include whether token/artifact payloads are resolvable and verified.

- [ ] **Step 1: Build a synthetic A-LMI workspace fixture using existing Python continuity APIs**

Fixture contains only synthetic data and at least one artifact manifest entry pointing to a LightToken JSON payload with SHA-256.

- [ ] **Step 2: Write tests that corrupted `.cosmos` input is rejected before LightToken discovery**

Corrupt one declared payload byte and assert native/JNI returns an integrity error category, not partial token results.

- [ ] **Step 3: Implement read-only discovery and verified temporary import**

Use existing Rust A-LMI import into a temporary directory owned by the operation; never write into original workspace/bundle.

- [ ] **Step 4: Run Rust + Java integration tests**

Expected: valid synthetic workspace/bundle discovers token; corrupted bundle fails; original inputs unchanged.

- [ ] **Step 5: Commit**

```bash
git add native/lighttoken-rs apps/lighttoken-workstation-java tests/fixtures/lighttoken
git commit -m "feat(lighttoken): integrate verified A-LMI workspace reads"
```

---

### Task 12: Add Windows build/install/run/test/verify/update/uninstall and Java packaging

**Files:**
- Create seven root `*_LIGHTTOKEN_WINDOWS.bat` files.
- Create `scripts/windows/lighttoken/common.ps1`, `build.ps1`, `install.ps1`, `run.ps1`, `test.ps1`, `verify.ps1`, `update.ps1`, `uninstall.ps1`.
- Modify: `.gitignore` for generated `dist/lighttoken`, Gradle `build/`, CMake `build/`.
- Modify Java Gradle build with `jlink`/`jpackage` tasks.

**Interfaces:**
- Install root `%LOCALAPPDATA%\A-LMI\LightToken` unless `LIGHTTOKEN_INSTALL_ROOT`.
- Data root `%LOCALAPPDATA%\A-LMI\LightToken\data`; uninstall preserves it by default.
- Installed command/application image launches workstation and native CLI.

- [ ] **Step 1: Clone the proven native-core PowerShell safety patterns instead of shelling out ad hoc**

Implement LightToken-specific `Assert-Windows`, external-command checking, architecture detection, source-tree root resolution, and application-owned path checks.

- [ ] **Step 2: Implement `BUILD_LIGHTTOKEN_WINDOWS.bat`**

Build order: Rust release CLI/JNI → CMake C++ Release → Java tests → Gradle `jlink`/`jpackage` application image. Supply C++ path to Rust self-test and Rust JNI path to Gradle tests.

- [ ] **Step 3: Implement install/run/verify**

Install copies only application image/native binaries/manifest into install root. Verify creates a temporary synthetic library, loads golden collection, performs Java→JNI→Rust comparison, then repeats with C++ backend if available and compares scores.

- [ ] **Step 4: Implement safe update**

For source installs: refuse dirty tree, `git fetch origin <current-branch>`, `git merge --ff-only`, rebuild/reinstall. Never touch data root.

- [ ] **Step 5: Implement safe uninstall**

Remove binaries/runtime/manifest only. `-RemoveData` requires interactive exact confirmation `DELETE LIGHTTOKEN DATA`; CI default never uses it.

- [ ] **Step 6: Run Windows lifecycle in Actions, not just local assumptions**

The workflow in Task 14 must execute all seven wrappers; failures are blockers.

- [ ] **Step 7: Commit**

```bash
git add '*LIGHTTOKEN_WINDOWS.bat' scripts/windows/lighttoken apps/lighttoken-workstation-java .gitignore
git commit -m "feat(lighttoken): add complete Windows workstation lifecycle"
```

---

### Task 13: Add fuzz/property/security hardening and descriptive benchmarks

**Files:**
- Create: `native/lighttoken-rs/fuzz/Cargo.toml`
- Create: `native/lighttoken-rs/fuzz/fuzz_targets/token_json.rs`
- Create: `native/lighttoken-rs/fuzz/fuzz_targets/collection_jsonl.rs`
- Create: `scripts/benchmark_lighttoken_native.py`
- Add Rust/C++/Java property/stress tests.

**Interfaces:**
- Fuzzers must never panic on arbitrary bounded bytes.
- Benchmark JSON records OS, CPU count, Python/Rust/Java versions, backend, collection size, median/p95; no pass threshold.

- [ ] **Step 1: Add Rust property tests for serialization/search invariants**

Generate finite f32 vectors only; assert canonical parse/write round-trip and deterministic ranking.

- [ ] **Step 2: Add JNI lifecycle stress test**

Create/free 1,000 contexts; compare fixtures 10,000 times inside a bounded test and assert no native error/panic.

- [ ] **Step 3: Add fuzz targets**

CI smoke commands:

```bash
cargo fuzz run token_json -- -runs=512 -max_len=65536
cargo fuzz run collection_jsonl -- -runs=256 -max_len=262144
```

- [ ] **Step 4: Implement benchmark harness**

Measure Python RFFT/similarity, Rust RFFT, Rust scalar batch, C++ batch when enabled, JNI pair/batch overhead, SQLite load/query, and synthetic collections of 100/1,000/10,000 tokens.

- [ ] **Step 5: Run benchmark once and verify schema, not speed**

Expected: JSON contains all requested environment/timing fields; no assertion that C++ or Rust is faster.

- [ ] **Step 6: Commit**

```bash
git add native/lighttoken-rs/fuzz native/lighttoken-rs native/lighttoken-cpp apps/lighttoken-workstation-java scripts/benchmark_lighttoken_native.py
git commit -m "test(lighttoken): add fuzz lifecycle and benchmark evidence"
```

---

### Task 14: Add dedicated cross-platform CI, packaging artifacts, SBOM, and checksums

**Files:**
- Create: `.github/workflows/lighttoken-native-ci.yml`
- Create/modify scripts needed for packaging/checksum manifest.

**Interfaces:**
- Workflow jobs: `quality`, `rust-contracts`, `python-interop`, `cpp-parity`, `c-abi-jni`, `java-tests`, `fuzz-smoke`, `benchmarks`, `sbom`, `portable-build`, `windows-smoke`.
- Artifacts named with `${{ github.sha }}`.

- [ ] **Step 1: Add quality and Rust/Python parity jobs**

Quality runs fmt, Clippy `-D warnings`, and `cargo audit`. Python job installs root package, builds CLI, runs `tests/test_light_token_contract.py` and `tests/test_lighttoken_native_interop.py`.

- [ ] **Step 2: Add C++ matrix job**

Build/test CMake on `ubuntu-latest`, `macos-14`, `windows-latest`; run Rust C++ parity with explicit trusted library path.

- [ ] **Step 3: Add Java/JNI job**

Use `actions/setup-java@v4` Temurin 21, Gradle cache, build Rust JNI library first, then `./gradlew test` with `lighttoken.native.dir`.

- [ ] **Step 4: Add Windows lifecycle job**

Execute exact user wrappers in order: BUILD → TEST → INSTALL → VERIFY → RUN → clean-tree assertion → UPDATE on push → PACKAGE/UPLOAD → UNINSTALL.

- [ ] **Step 5: Generate release candidate checksums and SBOM**

Retain Windows application image ZIP/installer, CLI, JNI DLL, C++ DLL, Java package/runtime image, fixture manifest, benchmark JSON, SPDX JSON SBOM, and `checksums.txt`.

- [ ] **Step 6: Push branch and require first CI run to reveal real failures**

Do not edit tests to conceal failures; debug root causes and preserve material RED/GREEN history.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/lighttoken-native-ci.yml
git commit -m "ci(lighttoken): verify cross-language workstation"
```

---

### Task 15: Documentation, PR closure, merge verification, and final evidence report

**Files:**
- Create: `docs/lighttoken/ARCHITECTURE.md`
- Create: `docs/lighttoken/INTEROPERABILITY.md`
- Create: `docs/lighttoken/WINDOWS_INSTALL.md`
- Create: `docs/lighttoken/SECURITY.md`
- Create: `docs/lighttoken/BENCHMARKS.md`
- Create: `docs/lighttoken/VERIFICATION.md`
- Modify: `README.md`, `TESTING.md`, `docs/CLAIMS_AND_LIMITATIONS.md`
- Create only after definitive evidence: `LIGHTTOKEN_NATIVE_WORKSTATION_FINAL_REPORT.md`

**Interfaces:**
- Documentation links exact executable commands and claim boundaries.
- Final report records exact commits, PRs, run IDs, artifacts, SHA-256s, toolchains, failures/fixes, benchmark environment, and remaining external gates.

- [ ] **Step 1: Write docs from executed behavior only**

Do not describe CLIP/WavLM execution, microphone capture, physical resonance, or production security as delivered.

- [ ] **Step 2: Run full local/CI-equivalent verification available in environment**

At minimum invoke Python oracle tests, Rust workspace tests, C++ tests, Java tests, CLI golden flow, and packaging checks before PR.

- [ ] **Step 3: Open implementation PR from the implementation branch**

Require `LightToken Native CI` plus existing repository Restoration/Security/Browser/Native Core workflows if triggered. Do not merge with any failing required repo workflow.

- [ ] **Step 4: Merge preserving meaningful history after all PR gates are green**

Use expected-head guard. Record merge SHA.

- [ ] **Step 5: Require fresh `main` LightToken CI success**

The final Windows run must execute real named-branch UPDATE, not a PR merge-ref skip.

- [ ] **Step 6: Download retained artifacts and independently verify checksums**

Compare `checksums.txt` against downloaded Windows/Java/native files before writing evidence.

- [ ] **Step 7: Write `LIGHTTOKEN_NATIVE_WORKSTATION_FINAL_REPORT.md` on a documentation-only follow-up branch**

Include the definitive implementation `main` SHA and exact run/artifact evidence. Preserve material failed runs and corrections.

- [ ] **Step 8: PR/merge the report and require one final report-inclusive `main` verification**

Closure requires all final triggered workflows green on the report-inclusive `main` SHA.

- [ ] **Step 9: Stop this project and move to the next repo project only after closure**

Next planned subsystem: multimodal encoding/adapters. Do not begin it on the LightToken branch.
