# LightToken Native Workstation — Cross-Language Design

**Date:** 2026-09-17  
**Repository:** `NavisWORLD/cosmic-synapse-A-lmi-v.2`  
**Base:** verified `main` at `f9c429c8c35c3a1283fddc715462f8e1d09e0c60`  
**Design branch:** `design/lighttoken-native-workstation-001`  
**Status:** design specification for user review; no implementation is authorized by this document alone

## 1. Purpose

Build the next repository subsystem after the closed A-LMI Native Core: a production-oriented, Windows-capable **LightToken Native Workstation** that preserves the existing Python LightToken implementation as the compatibility/reference oracle while adding a real Rust core, a narrow C++ acceleration layer, and a functional Java desktop application.

This is not a language-port exercise for its own sake. Each language has one deliberate role:

- **Python** remains the historical/reference implementation and compatibility oracle.
- **Rust** owns the new canonical native LightToken contract, validation, serialization, indexing orchestration, persistence adapters, and safe FFI boundaries.
- **C++** is optional and limited to performance-sensitive numerical kernels where benchmarks justify it.
- **Java** provides a durable desktop/workstation application and exercises a real native integration path through JNI.
- **Windows** is a first-class end-user packaging and lifecycle target rather than an afterthought.

The subsystem must remain compatible with the architectural invariants already established by A-LMI:

- `MODEL != SYSTEM`
- `MODEL != MEMORY`
- `MODEL != AUTHORITY`
- `UI != SYSTEM`
- `LANGUAGE != ARCHITECTURE`

## 2. Existing contract to preserve

The active Python implementation is `a_lmi/core/light_token.py`.

The native implementation must preserve the current software contract unless a deliberate, separately reviewed schema version is introduced.

The active LightToken contract includes:

- a unique `token_id`;
- UTC timestamp;
- `source_uri`;
- `modality`;
- `raw_data_ref`;
- optional `content_text`;
- arbitrary JSON-compatible metadata;
- optional perceptual hash;
- an optional semantic/model embedding with exactly **1536 `float32` values**;
- an optional one-sided real FFT spectral representation with exactly **769 complex bins** for the active representation;
- metadata labels identifying the transform as `embedding_rfft`;
- deterministic JSON field semantics compatible with the Python serializer;
- backward-compatible parsing of the historical magnitude/phase spectral representation;
- spectral power derived as magnitude of complex spectral bins;
- dominant-bin reporting;
- spectral similarity methods:
  - power correlation,
  - cosine,
  - normalized euclidean similarity;
- resonance matching by threshold and descending similarity.

The spectral vector is a software transform over embedding coordinates. It must not be described as physical frequency, a graph Fourier transform, biological resonance, or experimentally established physics unless an external mapping and evidence are separately supplied.

## 3. Scope

### 3.1 Included

This project includes:

1. A versioned Rust LightToken crate and API.
2. Python ↔ Rust LightToken parity and round-trip tests.
3. A narrow C ABI suitable for JNI and other native consumers.
4. An optional C++ numerical acceleration library.
5. Rust fallback implementations for every C++-accelerated operation.
6. C++ ↔ Rust numerical parity tests.
7. A Java desktop workstation using JNI to the Rust library.
8. A functional token library/index that can ingest, inspect, compare, search, and persist LightToken records.
9. A real A-LMI workspace and `.cosmos` read path for discovering LightToken-compatible records/artifact references where present.
10. Explicit export/import of LightToken JSON collections independent of `.cosmos`.
11. Windows build/install/run/test/verify/update/uninstall flows.
12. CI for Linux, Windows, and macOS where the component is portable.
13. Golden fixtures, fuzz/property tests, SBOM/checksum artifacts, and descriptive benchmarks.
14. A final evidence report for this subsystem after merge and post-merge verification.

### 3.2 Excluded

This project does not include:

- downloading or executing CLIP/WavLM model weights;
- live microphone capture;
- God Music browser audio functionality;
- HRCS radio/acoustic transport;
- Unity runtime integration;
- production Milvus/Neo4j deployment;
- universal semantic alignment claims across modalities;
- physical-frequency interpretation of embedding FFT bins;
- claims of consciousness, biological state, emotion detection, AGI, quantum behavior, or new physics;
- mandatory GPU acceleration;
- mandatory C++ execution when the Rust fallback is available.

Those are independent projects/gates.

## 4. Architecture

```text
                    ┌──────────────────────────────┐
                    │  Java LightToken Workstation│
                    │  UI + application services  │
                    └──────────────┬───────────────┘
                                   │ JNI
                                   v
                    ┌──────────────────────────────┐
                    │ Rust LightToken Native Core  │
                    │ schema / IO / search / index │
                    │ validation / persistence     │
                    └──────────┬───────────┬───────┘
                               │           │
                    C ABI/FFI  │           │ Rust fallback
                               v           v
                    ┌────────────────┐  ┌────────────────┐
                    │ C++ batch      │  │ Rust numerical │
                    │ similarity     │  │ implementation │
                    └────────────────┘  └────────────────┘
                               ^
                               │ parity fixtures
                               │
                    ┌──────────────────────────────┐
                    │ Python reference/oracle      │
                    │ a_lmi/core/light_token.py    │
                    └──────────────────────────────┘
```

The Java application never owns the canonical token format. C++ never owns persistence or schema semantics. Rust never deletes or replaces the Python reference implementation.

## 5. Rust subsystem

Create a new native workspace area under:

`native/lighttoken-rs/`

Crates:

- `lighttoken-core`
  - types;
  - schema/version constants;
  - validation;
  - JSON serialization/deserialization;
  - historical spectral-format compatibility;
  - exact dimension checks;
  - similarity method definitions.

- `lighttoken-spectrum`
  - active `rfft` computation;
  - spectral power;
  - dominant-bin calculation;
  - scalar Rust similarity implementations;
  - optional C++ backend dispatch.

- `lighttoken-index`
  - in-memory token collection;
  - stable ordering;
  - top-K search;
  - threshold search;
  - modality/source filters;
  - persistence of derived index metadata;
  - no hidden network access.

- `lighttoken-io`
  - JSON and JSONL token collections;
  - A-LMI workspace discovery adapters;
  - direct reuse of the existing verified Rust A-LMI continuity/`.cosmos` crates through path dependencies rather than a second archive parser;
  - raw artifact-reference resolution without silently claiming the referenced bytes exist.

- `lighttoken-ffi`
  - stable versioned C ABI;
  - opaque handles;
  - explicit allocation/free APIs;
  - panic containment;
  - deterministic error codes/messages;
  - JNI-facing entry points through a separate adapter layer.

- `lighttoken-cli`
  - dependency-light command-line diagnostics and automation paths;
  - useful independently of Java.

The Rust implementation is the canonical native behavior for this subsystem, but parity is judged against the preserved Python active contract.

## 6. Numerical behavior and floating-point policy

### 6.1 FFT

For a 1536-value real embedding, the active one-sided spectrum contains 769 complex values.

The implementation must use an established FFT library rather than handwritten DFT code for production paths. Library selection must be compatible with GPL-3.0 repository licensing and supported target platforms.

### 6.2 Cross-language parity

The following are exact where practical:

- dimensions;
- JSON keys;
- metadata labels;
- token IDs supplied in fixtures;
- timestamps supplied in fixtures;
- source/modality/raw-data fields;
- ordering rules;
- validation errors by semantic category;
- empty/degenerate similarity behavior.

Independent FFT implementations are not required to be bit-identical. Golden tests use documented absolute/relative tolerances for complex FFT components, spectral magnitudes, and similarity results.

Tolerance values must be selected from measured Python/Rust/C++ fixture differences, committed explicitly, and must not be widened merely to force a pass.

### 6.3 Degenerate cases

Behavior must match Python semantics:

- correlation of two identical zero-variance power arrays: `1.0`;
- correlation of unequal zero-variance power arrays: `0.0`;
- cosine with zero denominator: `1.0` only for equal arrays, otherwise `0.0`;
- normalized euclidean with zero maximum distance: `1.0`.

Non-finite input values are rejected by the native contract unless Python compatibility requires a documented historical read path. New writes must not persist NaN or infinity.

## 7. C++ acceleration boundary

C++ lives under:

`native/lighttoken-cpp/`

Build system: **CMake** with portable scalar code required and platform-specific SIMD used only behind compile/runtime capability checks.

Its API is deliberately small. Initial accelerated operations:

- batch L2 norm;
- batch spectral power magnitude;
- cosine similarity against many candidate vectors;
- power-correlation similarity against many candidates;
- normalized euclidean similarity against many candidates;
- top-K selection over score arrays.

C++ must not:

- parse workspace archives;
- own token JSON serialization;
- read secrets;
- perform networking;
- mutate A-LMI authority state;
- silently change scoring semantics;
- become a required dependency for correctness.

Rust performs a runtime backend self-test before using C++ acceleration. If the library is missing, incompatible, or fails self-test, Rust falls back to the verified Rust implementation and exposes the active backend in diagnostics.

No C++ performance claim is made until descriptive benchmarks demonstrate one on the tested hardware.

## 8. Java workstation

Create:

`apps/lighttoken-workstation-java/`

Target Java: **JDK 21 LTS bytecode/runtime baseline**.

Desktop toolkit: **JavaFX**. The project uses a pinned OpenJFX version compatible with JDK 21, Gradle dependency locking, and standard `jlink`/`jpackage` packaging. Windows release artifacts include an application-specific Java runtime image, so an end user does not need to install/configure a system JDK.

### 8.1 Functional capabilities

The workstation must perform real operations:

- open a directory containing LightToken JSON/JSONL data;
- open an A-LMI workspace and enumerate supported LightToken records/references;
- open a verified `.cosmos` bundle through the existing safe A-LMI verification/import surfaces;
- inspect token identity, timestamp, source, modality, metadata, raw-data reference, perceptual hash, embedding status, and spectral status;
- visualize the 1536 embedding values;
- visualize 769 spectral magnitudes;
- display dominant spectral bin and magnitude;
- compare two selected tokens using all supported methods;
- run top-K and threshold search across the loaded collection;
- filter search by modality/source metadata;
- show which numerical backend was used (`rust` or `cpp`);
- export selected tokens to canonical JSON;
- export query/search results with score, method, timestamp, and source token IDs;
- persist user-created collections/index metadata to a workstation-owned data directory without modifying the original source unless the user explicitly chooses an export destination.

### 8.2 Workstation library model

The application uses **SQLite** for its application-owned, rebuildable local library/index.

Default Windows database location:

`%LOCALAPPDATA%\A-LMI\LightToken\data\library.db`

SQLite stores derived/searchable metadata such as token ID, source location, modality, selected metadata fields, index revision, and cache status. The original LightToken JSON/JSONL or A-LMI source remains authoritative. Embedding/spectrum payload caching must be rebuildable and version-tagged; the application must be able to discard/rebuild incompatible derived cache state without modifying source tokens.

The Java application accesses SQLite through a pinned JDBC driver. Database migrations are explicit, versioned, transactional, and tested.

The application must display when a raw artifact reference cannot be resolved instead of fabricating availability.

### 8.3 Resonance Explorer capability

The visually distinctive part of the workstation is a real comparison surface, not a scripted animation:

- choose a query token;
- search the currently loaded collection;
- rank results by the selected real similarity method;
- render query and candidate spectral-power overlays;
- display score, dominant bins, metadata, source provenance, modality, backend, and raw reference;
- allow the user to switch methods and see rankings recompute;
- allow saving the result set as a deterministic query-results artifact.

The UI must label these as embedding-spectrum similarity results, not physical resonance measurements.

## 9. JNI/native integration

Java calls Rust through JNI. Rust is compiled as the native JNI library; C++ remains behind Rust and is never loaded directly by arbitrary Java application paths.

JNI functions expose high-level operations, not raw Rust structs:

- native library/ABI version;
- create/free engine context;
- load token/collection;
- validate token;
- compute spectrum;
- compare token pair;
- execute top-K query;
- return compact typed result payloads/primitive arrays;
- obtain backend diagnostics;
- release native buffers.

The JNI layer must:

- validate array lengths before native use;
- copy or pin memory only for bounded durations;
- never retain arbitrary JVM object pointers;
- translate native failures into deterministic Java exceptions/error objects;
- contain Rust panics before crossing the boundary;
- avoid leaking native allocations.

Stress tests repeatedly create/free contexts and execute comparisons to detect obvious lifecycle leaks/crashes.

## 10. Persistence and A-LMI integration

The workstation is a consumer of the A-LMI continuity model, not a replacement for it.

Rules:

1. `.cosmos` verification/import uses the already-verified A-LMI Rust crates/surfaces; this project does not create a second archive verifier.
2. Opening a workspace is read-only by default.
3. Search/index caches live under the workstation's own data directory.
4. Explicit exports may write to user-selected locations.
5. No operation grants tool/network/filesystem/cloud/deployment/actuator authority to a model.
6. No model inference is required to use LightToken search.
7. Raw artifact references are references; existence and digest must be verified before presenting bytes as available.

A later multimodal project may create LightTokens from CLIP/WavLM output. This project only consumes already-supplied embeddings/tokens and synthetic fixtures.

## 11. CLI

The new native CLI supports automation independent of the desktop application.

Commands:

```text
lighttoken doctor
lighttoken version
lighttoken inspect <token.json>
lighttoken validate <token.json>
lighttoken spectrum <token.json>
lighttoken compare <a.json> <b.json> --method <correlation|cosine|euclidean>
lighttoken index build <input> <index-dir>
lighttoken search <index-dir> <query.json> --top-k N --method <correlation|cosine|euclidean>
lighttoken backend
```

JSON output is required for automation paths.

## 12. Windows lifecycle

Windows is a mandatory closure surface.

Root wrappers:

- `INSTALL_LIGHTTOKEN_WINDOWS.bat`
- `BUILD_LIGHTTOKEN_WINDOWS.bat`
- `RUN_LIGHTTOKEN_WINDOWS.bat`
- `TEST_LIGHTTOKEN_WINDOWS.bat`
- `VERIFY_LIGHTTOKEN_WINDOWS.bat`
- `UPDATE_LIGHTTOKEN_WINDOWS.bat`
- `UNINSTALL_LIGHTTOKEN_WINDOWS.bat`

PowerShell implementation lives under:

`scripts/windows/lighttoken/`

Default user-local installation root:

`%LOCALAPPDATA%\A-LMI\LightToken`

Requirements:

- no administrator requirement for normal install;
- package the JavaFX application with a `jlink` runtime image and `jpackage` application image/installer artifact;
- package Rust native DLLs and optional C++ DLLs beside the application using a deterministic application-owned lookup strategy;
- fail with actionable diagnostics if architecture mismatch occurs;
- support Windows x86_64 first;
- preserve user workstation libraries, indexes, exports, A-LMI workspaces, `.cosmos` bundles, and backups during update/uninstall;
- uninstall removes only application-owned binaries/manifests by default;
- explicit destructive data removal requires a separate confirmation path;
- update is fail-closed on dirty source builds when updating from a Git checkout and uses fast-forward-only source update semantics, consistent with the established native-core approach.

## 13. Security and robustness

Required controls:

- bounded JSON/token size limits;
- bounded token-count/index limits configurable with safe defaults;
- no arbitrary native library path loading from token metadata;
- no execution of content from `source_uri`, `raw_data_ref`, metadata, or content text;
- no implicit network requests when opening tokens;
- no embedded credentials in persisted provenance paths;
- safe path handling for exports/imports;
- JNI buffer ownership documented and tested;
- native error messages sanitized of secrets;
- fuzz token JSON/deserialization and collection parsing;
- property tests for serialization round trips and similarity invariants;
- malformed historical payloads fail closed where compatibility is not explicitly supported;
- SQLite queries use prepared statements for user/source-derived values;
- database migrations run transactionally and back up application-owned metadata before destructive schema transitions.

This is software security engineering evidence, not a production security certification.

## 14. Testing strategy

### 14.1 Python oracle fixtures

Commit synthetic fixtures containing:

- fixed UUIDs/timestamps;
- fixed metadata;
- embeddings covering:
  - all-zero,
  - constant,
  - impulse-like,
  - sinusoidal coordinate pattern,
  - deterministic pseudo-random finite vector,
  - pairs designed for high/medium/low similarity;
- Python-generated active complex spectrum values;
- Python similarity outputs;
- historical magnitude/phase serialization example.

No private/user data belongs in fixtures.

### 14.2 Rust

Tests include:

- exact schema/dimension validation;
- Python fixture parse/write parity;
- FFT parity within fixed tolerances;
- similarity parity;
- historical reader compatibility;
- deterministic top-K ordering including score ties;
- malformed/non-finite rejection;
- property tests;
- fuzz parsing;
- C ABI smoke caller.

### 14.3 C++

Tests include:

- scalar known vectors;
- Rust-vs-C++ batch score parity;
- top-K parity;
- empty and degenerate cases;
- alignment/length failures;
- backend self-test failure and Rust fallback behavior.

### 14.4 Java

Tests include:

- JNI library version/ABI check;
- load known fixture;
- compare known token pair;
- top-K known collection;
- repeated context lifecycle;
- invalid input → deterministic Java error;
- SQLite schema/migration and library/index persistence;
- headless application-service tests independent of JavaFX rendering;
- JavaFX controller/view-model tests for selection, method switching, search-result display, and export actions without fake computed data.

### 14.5 End-to-end Windows

CI must execute the actual user-facing lifecycle:

```text
BUILD
→ TEST
→ INSTALL
→ VERIFY
→ RUN workstation/CLI smoke
→ load golden collection
→ Java→JNI→Rust comparison
→ Java→JNI→Rust→C++ comparison if backend available
→ compare backend outputs
→ UPDATE
→ PACKAGE
→ UPLOAD ARTIFACTS
→ UNINSTALL
```

A successful native-core Windows run is not reused as proof for this new subsystem.

## 15. CI and release evidence

Add a dedicated workflow:

`.github/workflows/lighttoken-native-ci.yml`

Jobs:

- Rust format/Clippy/audit;
- Rust contract/property tests;
- Python↔Rust oracle parity;
- C++ build/parity on Linux/Windows/macOS where supported;
- C ABI/JNI native smoke;
- Java unit/application-service tests;
- JavaFX packaged desktop build;
- Windows lifecycle smoke;
- fuzz smoke;
- descriptive benchmarks;
- SBOM generation;
- artifact checksums.

Artifacts include as applicable:

- Windows workstation package;
- native CLI;
- Rust native/JNI library;
- C++ acceleration DLL/library;
- packaged JavaFX application/runtime image;
- SBOM;
- checksums;
- benchmark JSON;
- golden fixture manifest.

## 16. Benchmarks

Benchmarks are descriptive only and must record environment metadata.

Measure:

- Python spectral computation;
- Rust spectral computation;
- Rust scalar batch search;
- Rust + C++ batch search;
- JNI overhead for pair compare and batch query;
- SQLite index load/query overhead separately from numerical scoring;
- index load/search at multiple synthetic collection sizes.

No threshold or language-superiority statement is required for acceptance. If C++ is not measurably useful on tested workloads, it remains optional and the result is reported honestly.

## 17. Compatibility/versioning

Define a LightToken native schema/API version separate from the global A-LMI workspace version.

Initial native version reads the existing active Python format and the explicitly supported historical magnitude/phase spectral form.

Writers emit only the active canonical form:

- 1536-value embedding;
- 769 real/imag spectral bins when spectrum is present;
- active spectral metadata labels.

Future breaking changes require a version bump plus migration/compatibility tests. The workstation shows unsupported-version errors rather than silently coercing unknown data.

## 18. Repository layout

Target additive layout:

```text
native/
├── almi-core-rs/                  # existing verified A-LMI native core
├── lighttoken-rs/                 # new Rust LightToken subsystem
└── lighttoken-cpp/                # optional acceleration kernels

apps/
└── lighttoken-workstation-java/   # functional JavaFX desktop application

scripts/windows/lighttoken/        # Windows lifecycle implementation

tests/fixtures/lighttoken/         # synthetic cross-language golden vectors

docs/lighttoken/                   # user/developer/security/interop docs
```

The original Python file remains at `a_lmi/core/light_token.py`.

## 19. Delivery sequence

Implementation is decomposed into independently reviewable phases:

1. RED golden parity tests and fixtures.
2. Rust schema/serialization implementation.
3. Rust FFT/similarity parity.
4. Rust index/search implementation.
5. Stable C ABI/JNI contract.
6. C++ optional acceleration plus fallback/self-test.
7. Java headless application services and SQLite library.
8. Functional JavaFX workstation UI.
9. A-LMI workspace/verified `.cosmos` read integration.
10. Windows lifecycle/package.
11. Fuzz/property/security hardening.
12. Benchmarks/SBOM/checksums/docs.
13. Full PR verification.
14. Merge and post-merge `main` verification.
15. Final LightToken evidence report.

No later repository project is mixed into this branch.

## 20. Acceptance criteria

The subsystem is complete only when all of the following are evidenced on the final merged commit or an explicitly identified definitive pre-report commit followed by report-inclusive final verification:

1. Existing Python LightToken tests remain green.
2. Rust reads/writes active Python fixtures correctly.
3. Rust FFT and similarity match Python within committed measured tolerances.
4. C++ acceleration matches Rust within committed tolerances and can be disabled/fail over without correctness loss.
5. Java loads real fixture/token files through JNI, performs real comparison/search, and presents actual results.
6. The workstation can load a collection, select a token, run top-K/threshold search, visualize embedding/spectral data, switch similarity methods, and export results.
7. SQLite library/index state is rebuildable and does not replace authoritative source tokens.
8. A-LMI workspace/`.cosmos` integration uses safe existing verification surfaces and is read-only by default.
9. Windows user-facing BAT flows build/test/install/verify/run/update/package/uninstall successfully in CI.
10. The packaged Windows application runs with its application-specific Java runtime and does not require a preconfigured system JDK.
11. Uninstall/update do not delete user data by default.
12. Rust/C++/Java artifacts are retained with checksums.
13. Fuzz/property/ABI/JNI lifecycle tests pass.
14. SBOM and descriptive benchmark evidence are retained.
15. No unsupported physical/consciousness/security/performance claims are introduced.
16. A final evidence report records exact commits, CI runs, artifact IDs/hashes, toolchains, failures encountered, limitations, and unexecuted external gates.

## 21. Follow-on repository sequence

After this LightToken subsystem closes, the recommended next projects are handled separately in this order unless later repository evidence changes the dependency graph:

1. Multimodal encoding/adapters.
2. Memory/vector/graph integration and reasoning utilities.
3. HRCS native expansion.
4. God Music native/WASM integration.
5. Unity/CST client/runtime integration.
6. Whole-product packaging and final cross-subsystem release closure.

Each receives its own design, implementation plan, branch, CI evidence, and final report. Language roles are reused where valuable rather than mechanically rewriting every line in C++, Java, Rust, and Python.

## 22. Design decision summary

The chosen design is a real multi-language product surface:

`JavaFX desktop → JNI → Rust canonical LightToken native engine → optional C++ acceleration`

with Python retained as the active compatibility oracle, SQLite used only for rebuildable workstation index/cache state, and Windows treated as a first-class user platform.

The distinctive visual “Resonance Explorer” is therefore not a mock demo. It is the human-facing surface of the actual LightToken query engine and must display only results computed from loaded token data through the same native APIs used by the CLI/tests.
