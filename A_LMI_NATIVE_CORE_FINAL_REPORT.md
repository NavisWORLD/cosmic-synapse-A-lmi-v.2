# A-LMI Native Core / Rust — Final Evidence Report

**Repository:** `NavisWORLD/cosmic-synapse-A-lmi-v.2`  
**Subsystem:** A-LMI Native Core / Rust  
**Scope:** native foundation only; no C++ LightToken kernel, native HRCS expansion, God Music WASM expansion, or Unity expansion is included in this closure.

## 1. Closure status

This report records the evidence for the first native A-LMI subsystem. The existing Python implementation remains present and remains the compatibility/reference implementation. The Rust native core is additive and is not a rewrite or deletion of the Python runtime.

Architectural invariants preserved:

- `MODEL != SYSTEM`
- `MODEL != MEMORY`
- `MODEL != AUTHORITY`
- `UI != SYSTEM`
- `LANGUAGE != ARCHITECTURE`

This closure demonstrates software continuity and cross-language interoperability. It does not demonstrate or claim consciousness, sentience, biological identity, soul continuity, resurrection, AGI, new physics, physical extra dimensions, quantum advantage, or production security certification.

## 2. Source-control provenance

| Item | Value |
| --- | --- |
| Starting live `main` | `fc233619367cd9d2f5069198d4b0d7bf3683bb24` |
| Implementation branch | `native/almi-core-rust-001` |
| Final implementation PR head | `d643b5433d686b48239026f072b4b1d93b3599dc` |
| Implementation PR | #4 — `A-LMI Native Core / Rust foundation closure` |
| Implementation merge SHA | `2a212b1376c9cb705f66c7cecf77656835e7571f` |
| Closure hardening branch | `native/almi-core-rust-001-closure-hardening` |
| Closure hardening PR head | `50d9007496db8a4e380cb7e15771f0b3ced8d826` |
| Closure hardening PR | #5 — `Native closure evidence hardening` |
| Evidence-hardening merge SHA | `1f4b289d22e4937cd94e9a33352089b3ff353911` |
| Safe-update fix PR | #6 — `Fix safe Windows update after native builds` |
| Safe-update fix PR head | `b6554db5f5f22ce7956ad17cf28af6bce2ad01b7` |
| Definitive native closure `main` SHA | `74d30f42c6ab563b8a780e13de46ab3f2432f661` |

The implementation branch was merged only after its branch-head and PR-triggered native/repository workflows were green. The closure-hardening PR retained the Linux PyO3 wheel as an artifact and added named-branch execution of `UPDATE_WINDOWS.bat`. That new gate exposed a real update-path failure on main run #88: normal Cargo output made the checkout appear dirty. PR #6 fixed the root cause by ignoring Rust `target/` output and added a pre-update clean-tree assertion without weakening the updater's fail-closed dirty-tree guard.

## 3. Native architecture delivered

The maintained Rust workspace is under `native/almi-core-rs/` and contains narrow crates for:

- `almi-core` — versioned domain contracts and shared errors/canonical serialization
- `almi-continuity` — continuity workspace initialization/inspection
- `almi-memory` — append/scan/verify memory ledger
- `almi-state` — active canonical CST state load/validate/persist/replay
- `almi-provider` — provider abstraction and Ollama-compatible adapter
- `almi-cosmos` — deterministic `.cosmos` export/verify/inspect/import
- `almi-runtime` — minimum persistent runtime loop and provider-swap continuity
- `almi-cli` — native `almi` executable and benchmark harness
- `almi-ffi` — deliberately small stable C ABI v1
- `almi-python` — PyO3 Python bindings

The model remains replaceable. Memory, CST state, routing, policy/authority, continuity, and provenance remain outside the model-provider boundary.

## 4. Active continuity contract

Native workspace initialization and interoperability preserve the active continuity layout used by the Python reference:

- `system.json`
- `memory/ledger.jsonl`
- `state/cst.json`
- `knowledge/graph.json`
- `artifacts/manifest.json`
- `provenance/provider.json`
- `routing/state.json`
- `policy/authority.json`

Default authority remains deny-by-default for:

- tool authority
- network authority
- filesystem authority
- cloud authority
- deployment authority
- actuator authority

Configuring or swapping a model/provider does not grant those authorities.

## 5. `.cosmos` interoperability and security

Rust implements:

- workspace creation
- deterministic `.cosmos` export
- `.cosmos` verification
- metadata/workspace inspection
- `.cosmos` import

Validation rejects unsafe/malformed input including traversal and `..`, absolute paths, backslash/drive-path forms, symlinks, duplicate members, undeclared members, missing declared members, SHA-256 mismatches, size mismatches, unsupported schema/format versions, secret-bearing file names, excessive member counts, oversized members/archives, unsafe extraction destinations, and prohibited non-empty import destinations.

Cross-language tests execute both directions:

1. Python workspace → Python export → Rust verify/import.
2. Rust workspace → Rust export → Python verify/import.

Committed synthetic golden vectors additionally force both implementations to export the same fixed workspace and compare canonical manifest and payload bytes.

## 6. Golden-vector SHA-256 values

Canonical bundle manifest:

`8cec9f101e1014aa4529fc9a78135dc179ea984d5dcaeda5fbfcf312ec95829f`

| Payload | SHA-256 |
| --- | --- |
| `artifacts/manifest.json` | `b571dbecd090b4ea3c9e790e8d81ade86f71b58565cb3ad7208aa5f271bd4d5c` |
| `knowledge/graph.json` | `1412537fdaadcc335a927141faf99be219ce39d25fc6a4e9a500e424e9204bb2` |
| `memory/ledger.jsonl` | `f6a12f8144ce6a52f366176b8b4d695d039ee076cae0a367594fb05d6a7e1167` |
| `policy/authority.json` | `5b37ce887860dcfb251396946d990ed5ff413d6cfb409afeb611b2cd42a031cc` |
| `provenance/provider.json` | `8648f2895e642899fda0a71e26cfd938066d12014617a659c86fd185360cdc12` |
| `routing/state.json` | `268d3aa84cc852f1d90bdad075ed4de23cc2bfed588f0a0892865611fa8cc55a` |
| `state/cst.json` | `ddb1bec05ed7b39fcc9f4abeba62d4e7b069ff629dfb5ede706966bb72897e49` |
| `system.json` | `40a89b94717b3a3d1bec5ccf0c5126edd823be54b35703c1d48bb89595dec1a8` |

These fixtures contain synthetic/non-sensitive data only.

## 7. Memory and CST parity

The native memory ledger preserves explicit record versions, append ordering, canonical serialization, provenance fields, malformed-record rejection, and integrity validation. The implementation does not describe the ledger as cryptographically immutable beyond the integrity mechanisms actually present.

The native CST layer implements the active persistent computational-state contract rather than reimplementing every historical experiment. It supports load, validation, serialization, persistence, deterministic supported transitions, replay, and snapshot comparison. Persisted canonical state is authoritative for cross-language replay. Project terms such as `12D` remain computational/historical terminology and are not physical-dimension claims.

## 8. Provider and persistent runtime

The native provider boundary models provider identity, model ID/revision, endpoint, health, capabilities, model request/response, timeout/retry behavior, and provenance. The Ollama-compatible default endpoint is loopback/local. Credential-bearing endpoints are rejected where the active reference contract rejects them. Secret values are not intentionally written to provenance or bundle metadata.

The persistent runtime loop is:

`workspace → load continuity → construct request → provider → response → provenance → append interaction memory → persist state`

The deterministic provider-swap test exercises A → B → A software continuity while verifying that authority is unchanged. This proves software-state continuity only.

**Known provider limitation:** the repository's Restoration CI runs real Ollama CPU inference through the existing Python product provider. Native Core CI verifies the Rust adapter contract, defaults, request handling, and security behavior, but this closure did not execute a live Rust→Ollama network inference. No such native-live claim is made.

## 9. CLI surface

The native executable is `almi` and includes:

- `almi doctor`
- `almi version`
- `almi init`
- `almi inspect`
- `almi export`
- `almi verify`
- `almi import`
- `almi memory verify`
- `almi state replay`
- `almi provider list`
- `almi provider health`
- `almi provider run`
- `almi runtime run`

Failures return non-zero status. JSON output is available for automation-oriented paths.

## 10. Python bindings

PyO3 bindings expose native continuity functionality through `almi_native` while preserving the Python reference implementation. The binding uses PyO3 0.29.2 after a real compatibility/lint issue was encountered with the earlier 0.22.6 macro expansion under the current Rust toolchain.

Definitive closure Linux wheel:

- file: `almi_native-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl`
- SHA-256: `08515ff66c5c95f447fe2132104cb3f386fd9a4d023159b19a005d0f5ebfa94e`

- Windows file: `almi_native-0.1.0-cp311-abi3-win_amd64.whl`
- Windows SHA-256: `95b147639613a3622fa341b156a4b25caa08aef2a668a195272c47969c6b7b35`

No wheel was published to PyPI.

## 11. Stable C ABI

Public header:

`native/almi-core-rs/include/almi.h`

ABI version:

`ALMI_ABI_VERSION = 1`

The ABI uses opaque context ownership, explicit free functions, stable integer error codes, null/error handling, and Rust panic containment. An external C caller is compiled, linked, and executed in CI. Rust-native struct layout is not exposed as the public ABI.

## 12. Windows user experience

The repository includes and CI executes the user-facing Windows entry points:

- `INSTALL_WINDOWS.bat`
- `BUILD_WINDOWS.bat`
- `RUN_WINDOWS.bat`
- `TEST_WINDOWS.bat`
- `VERIFY_WINDOWS.bat`
- `UPDATE_WINDOWS.bat`
- `UNINSTALL_WINDOWS.bat`

The implementation uses user-local installation (`%LOCALAPPDATA%\A-LMI` by default), avoids administrator requirements unless actually needed, supports optional Python bindings, and protects user workspaces, `.cosmos` bundles, memories, and backups from ordinary uninstall/update operations.

The definitive named-`main` push runs the update BAT in CI; the pull-request merge-ref run deliberately skips it because the update script requires a named Git branch and performs a fail-closed `fetch` + `merge --ff-only` update.

On Native Core CI #90 (`35260994732`) at `74d30f42c6ab563b8a780e13de46ab3f2432f661`, the pre-update source-tree assertion passed, `UPDATE_WINDOWS.bat -WithPython` fetched `origin/main`, executed `git merge --ff-only origin/main`, reported `Already up to date.`, rebuilt/reinstalled the native CLI and Python binding, passed its installed-binary verification, and ended with `UPDATE PASS. Existing user workspaces and .cosmos bundles were not deleted or modified.`

## 13. Toolchain and OS matrix

| Surface | Verified environment |
| --- | --- |
| Rust | `rustc 1.98.1 (48a229cea 2026-09-01)`, Cargo 1.98.1 |
| Linux | Ubuntu 24.04.5 LTS, x86_64, CPython 3.11.16 |
| macOS | macOS 14.7.8, arm64/aarch64 GitHub-hosted runner |
| Windows | Windows Server 2025 Datacenter 10.0.26100 x64, GitHub `windows-2025-vs2026`, CPython 3.11.9 |

These results describe hosted CI environments, not certification across arbitrary end-user hardware.

## 14. Executed test and quality gates

The native closure executes:

- `cargo fmt --all -- --check`
- strict `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --locked`
- release portable builds
- `cargo audit`
- Python/Rust cross-language interoperability
- PyO3 wheel build/install/import
- external C ABI compile/link/run
- bounded cargo-fuzz hostile `.cosmos` smoke (`256` runs, max input length `65536`)
- property tests for critical serialization/validation contracts
- descriptive Python-vs-Rust benchmark harness
- SPDX JSON SBOM generation
- Linux native build/test
- macOS ARM64 native build/test
- Windows BAT build/test/install/verify/run/update/package/upload/uninstall on named `main`

The Rust workspace test run executes 28 Rust tests across unit/integration/property suites. The cross-language Python suite executes 5 tests. Passing fuzz smoke is a bounded robustness signal, not exhaustive proof of parser security.

## 15. CI evidence chronology

Material RED/GREEN and correction history was preserved instead of rewritten:

- domain-contract RED: run `35052285227`
- domain-contract GREEN: run `35052362927`
- broader native RED: run `35052436355`
- missing direct `serde` dependency surfaced: run `35052697649`; corrected
- Rust workspace GREEN: run `35052746039`
- CLI/C ABI RED: run `35052852255`
- CLI/C ABI GREEN: run `35053055539`
- deliberate missing PyO3 module initializer RED: run `35053219181`
- bundle/PyO3 interop GREEN after implementation: run `35053519209`
- branch-head closure run #84: `35147904235`, SUCCESS
- implementation PR #4 Native Core run #85: `35148689581`, SUCCESS
- PR #4 repository gates: Restoration `35148689632`, Security VM `35148689573`, Browser VM `35148689600`, all SUCCESS
- implementation merge: `2a212b1376c9cb705f66c7cecf77656835e7571f`
- implementation post-merge Native Core run #86: `35151381794`, SUCCESS
- closure-hardening PR #5 Native Core run #87: `35173524254`, SUCCESS
- PR #5 repository gates: Restoration `35173524287`, Security VM `35173524258`, Browser VM `35173524433`, all SUCCESS
- evidence-hardening main SHA: `1f4b289d22e4937cd94e9a33352089b3ff353911`
- Native Core run #88: `35173905318`, FAILED at the newly exercised `UPDATE_WINDOWS.bat` gate after build/test/install/verify/run had passed; root cause was unignored Cargo `target/` output making the checkout appear dirty
- safe-update PR #6 head: `b6554db5f5f22ce7956ad17cf28af6bce2ad01b7`; Native Core PR run #89 `35174396329`, SUCCESS, including the new clean-tree assertion
- PR #6 repository gates: Restoration `35174396306`, Security VM `35174396332`, Browser VM `35174396304`, all SUCCESS
- safe-update merge / definitive native closure SHA: `74d30f42c6ab563b8a780e13de46ab3f2432f661`
- definitive native closure Native Core run #90: `35260994732`, SUCCESS
- definitive native closure repository gates on the same SHA: Browser VM #40 `35260994716`, Security VM #35 `35260994757`, Restoration CI #251 `35260994820`, all SUCCESS

The history also includes the Windows PowerShell `$LASTEXITCODE:` parser fix, the Windows imported-workspace schema assertion correction, FFI Clippy/safety corrections, and the PyO3 0.22.6 → 0.29.2 migration with machine-generated lockfile refresh.

## 16. Definitive closure artifacts

All values in this section are from exact definitive native closure SHA `74d30f42c6ab563b8a780e13de46ab3f2432f661` / Native Core CI #90 unless stated otherwise.

| Artifact | Evidence |
| --- | --- |
| Linux PyO3 artifact | ID `10513918724`; Actions artifact digest `sha256:dd5003c85c89711b277c6ec7585d703f5a6762ccc98caf91893b2a73b6ac1be7` |
| Linux PyO3 wheel | `almi_native-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl`; SHA-256 `08515ff66c5c95f447fe2132104cb3f386fd9a4d023159b19a005d0f5ebfa94e` |
| Benchmark artifact | ID `10514797178`; Actions digest `sha256:868031502d7f8cece839e3a2f9ab906d62a6f543d9ec10322c78c49b35700867`; `ci.json` SHA-256 `d582ab35b70c55614ed20dbd4e8a6e252cb809c07204a321787760bd03fbfaef` |
| SBOM artifact | ID `10514647332`; Actions digest `sha256:084c4d5b8e3056db37a9850b1fa6060a596b1f7af6f80e6e78314d93c150ea8c` |
| Windows artifact | ID `10513909321`; Actions digest `sha256:0f38b71d504dd36f842013b345ad78d65dfec338c8341e279bd5be150a00fc3c` |
| Portable Windows ZIP | `A-LMI-native-0.1.0-windows-x86_64.zip`; SHA-256 `e2e77df59a19524b17796268ba6af7df29dff7a6f414c4abb8299a0e9d3737e3` |
| Windows `almi.exe` | `almi.exe`; SHA-256 `9afcfbd4e5d4b25640cbebfaece4246e07a508485eb60b494ad79c85510d30df` |
| Windows PyO3 wheel | `almi_native-0.1.0-cp311-abi3-win_amd64.whl`; SHA-256 `95b147639613a3622fa341b156a4b25caa08aef2a668a195272c47969c6b7b35` |

The Windows artifact's generated `checksums.txt` records the same ZIP, executable, and wheel digests shown above; those values were independently recomputed from the downloaded artifact and matched exactly.

No crate was published to crates.io and no wheel was published to PyPI as part of this closure.

## 17. Descriptive benchmark evidence

Definitive native closure run #90, 15 iterations, Linux x86_64, Python 3.11.16. Values are microseconds. They are descriptive for this CI machine/run and are not generalized performance claims.

| Operation | Python median | Python p95 | Rust median | Rust p95 |
| --- | ---: | ---: | ---: | ---: |
| workspace init | 903.227 | 8576.861 | 474.632 | 521.700 |
| bundle verify | 485.201 | 967.718 | 136.746 | 178.906 |
| bundle export | 1336.771 | 1932.711 | 684.687 | 1209.482 |
| bundle import | 2001.530 | 2081.139 | 834.558 | 996.752 |
| memory append | 163.578 | 234.601 | 31.499 | 58.520 |
| memory scan (100 records) | 181.181 | 208.732 | 85.681 | 101.370 |
| state serialize | 16.440 | 34.515 | 4.568 | 7.173 |

## 18. Definition-of-Done audit

The final audit is evidence-based. A check is marked complete only when the implementation exists and the corresponding applicable test/build/user path was executed successfully.

- [x] maintainable multi-crate Rust workspace
- [x] native core release build
- [x] versioned continuity structures
- [x] memory ledger
- [x] deny-by-default authority/policy
- [x] routing/provenance contracts
- [x] active CST state interoperability/replay
- [x] `.cosmos` verify/export/import/inspect
- [x] Python → Rust bundle verify/import
- [x] Rust → Python bundle verify/import
- [x] deterministic committed golden vectors
- [x] native provider/runtime path implemented and tested
- [x] provider swap preserves continuity
- [x] provider swap does not inherit authority
- [x] native CLI
- [x] Python bindings build/install/import and interop tests
- [x] minimal stable C ABI v1 and external C smoke caller
- [x] Rust unit/integration/property tests
- [x] bounded fuzz smoke
- [x] rustfmt
- [x] strict Clippy
- [x] `cargo audit`
- [x] Windows release executable build
- [x] `INSTALL_WINDOWS.bat`
- [x] `BUILD_WINDOWS.bat`
- [x] `RUN_WINDOWS.bat`
- [x] `TEST_WINDOWS.bat`
- [x] `VERIFY_WINDOWS.bat`
- [x] `UPDATE_WINDOWS.bat` — executed on named `main` in Native Core CI #90 and SUCCESS
- [x] `UNINSTALL_WINDOWS.bat` protects user data by default
- [x] Windows install/verify/package/uninstall smoke
- [x] Linux build/test
- [x] macOS Apple Silicon build/test
- [x] versioned Windows release candidate artifact
- [x] SHA-256 checksums
- [x] SPDX JSON SBOM
- [x] Linux and Windows Python wheels retained/hashes recorded
- [x] native documentation and root README/claim boundaries updated
- [x] exact implementation branch-head CI green
- [x] implementation PR created and merged only after PR checks
- [x] exact implementation post-merge `main` CI green
- [x] closure-hardening PR checked and merged
- [x] exact definitive closure `main` CI — Native Core CI #90 (`35260994732`) SUCCESS; Browser VM #40 (`35260994716`) SUCCESS; Security VM #35 (`35260994757`) SUCCESS; Restoration CI #251 (`35260994820`) SUCCESS on SHA `74d30f42c6ab563b8a780e13de46ab3f2432f661`
- [x] final evidence report records exact evidence rather than future claims

## 19. Known limitations and remaining external/hardware gates

This native subsystem is complete for the scoped software foundation, but the following are outside the evidence established here:

- live Rust→Ollama inference against a real Ollama server was not executed by Native Core CI; only the Rust adapter contract and the existing Python product's real Ollama path are evidenced
- no production security certification or formal audit
- no arbitrary real-user Windows hardware matrix beyond the hosted Windows x64 runner
- no arbitrary Linux distributions beyond the hosted Ubuntu runner
- no arbitrary macOS hardware beyond hosted Apple Silicon/macOS 14
- no production signing/notarization/installer code-signing claim
- fuzz smoke is bounded, not exhaustive
- no crates.io/PyPI publication
- no C++ LightToken kernel work in this subsystem
- no native HRCS expansion in this subsystem
- no God Music WASM expansion in this subsystem
- no Unity expansion in this subsystem
- no untested physical sensor/actuator behavior claim

Repository-wide Restoration CI may execute Kafka, MinIO, Milvus, Neo4j, Ollama, CLIP, WavLM, and Vosk gates for the broader existing product. Those are separate from the evidence scope of the Rust native core and are not re-labeled as native-Rust hardware/provider proof.

## 20. Final claim boundary

Evidence supports the following scoped statements:

- the Rust implementation interoperates with the Python reference for the tested continuity and active CST contracts
- Rust verifies/imports Python-generated `.cosmos` bundles in the tested compatibility cases
- Python verifies/imports Rust-generated `.cosmos` bundles in the tested compatibility cases
- deterministic golden payload/manifest serialization is shared for the committed synthetic fixture
- provider replacement does not automatically replace memory/state/routing/policy/authority in the tested persistent runtime path
- the tested Windows user flow builds, installs, verifies, runs, updates, packages, and uninstalls while preserving the default user-data boundary
- the tested Linux and macOS native builds pass on the recorded hosted runners
- the minimal C ABI v1 can be compiled/linked/called by an external C program in the recorded CI environment

This is evidence of software architecture and continuity. It is not evidence of consciousness, sentience, biological identity, a soul, resurrection, AGI, new physics, extra physical dimensions, or quantum advantage.
