# Final Product Closure Evidence

This document records what the final product-closure work establishes through deterministic software execution and what still requires a real external environment. It is not a production-readiness certificate.

## Evidence classifications

- `VERIFIED_SOFTWARE` — exercised by deterministic tests/builds in CI.
- `BLOCKED_EXTERNAL_ENVIRONMENT` — software/configuration exists, but the required external service/model was not executed in the closure environment.
- `REQUIRES_DEVICE_ENVIRONMENT` — requires browser/audio/device permissions and physical I/O.
- `REQUIRES_HARDWARE` — requires specific physical hardware and measurement.
- `REQUIRES_UNITY_ENVIRONMENT` — requires a real Unity editor/player build/runtime.
- `REQUIRES_GPU_ENVIRONMENT` — requires a GPU/CUDA-capable environment.
- `NOT_PRODUCTION_CERTIFIED` — research/development behavior exists, but production deployment/security/performance has not been independently established.

## Closure matrix

| Gate | Status | What was verified | What still requires a real environment |
|---|---|---|---|
| Portable continuity workspace | `VERIFIED_SOFTWARE` | Versioned workspace, separated memory/state/provider/routing/policy surfaces, zero default tool/network/filesystem authority | Real user acceptance across multiple machines/filesystems |
| `.cosmos` export / verify / import | `VERIFIED_SOFTWARE` | Deterministic frozen-workspace export, SHA-256 + byte-size manifest, safe import, corruption/traversal/symlink/duplicate/undeclared-member/secret-file rejection | Long-term migration testing across future format versions |
| Provider swap continuity | `VERIFIED_SOFTWARE` | Deterministic providers prove memory survives provider replacement and policy authority remains unchanged | Additional real hosted/local provider integrations |
| Ollama provider client | `VERIFIED_SOFTWARE` for client contract; `BLOCKED_EXTERNAL_ENVIRONMENT` for a real model run | Loopback default, explicit identity, endpoint validation, embedded-credential rejection, timeout/retry/error behavior, response provenance, no network on construction | Running Ollama plus an installed selected model and recording a live response |
| CLI product flow | `VERIFIED_SOFTWARE` | `init`, `inspect`, `export`, `verify`, `import`, `providers`, `run`; installed-wheel portable smoke | Human UX acceptance on target operating systems |
| Local benchmark/soak harness | `VERIFIED_SOFTWARE` | Measured CST + continuity round-trip harness with environment metadata and no fake performance threshold | Representative long-duration/product-load benchmarking on target hardware |
| Active security regression gate | `VERIFIED_SOFTWARE` | Active-path checks for tracked root `.env`, placeholder secrets, secret fallbacks, loopback Compose ports, selected direct dynamic-execution shortcuts, bundle import hardening, authority separation, provider endpoint credential rejection | Deployment-specific threat model, dependency/image vulnerability scanning, authn/authz, rate limits, external review |
| Python 3.11 / 3.12 deterministic contracts | `VERIFIED_SOFTWARE` | Active core, continuity, provider/runtime, CLI, HRCS deterministic surfaces, IPC schema/source contracts | Other Python/OS matrices if desired |
| Wheel / sdist / clean install | `VERIFIED_SOFTWARE` | Build, clean virtualenv install, doctor/import smoke, installed product round trip | Distribution-channel-specific installation testing |
| God Music Node/Vite | `VERIFIED_SOFTWARE` for deterministic tests/build | Node tests and Vite production build | Live microphone/browser/device behavior |
| Kafka | `BLOCKED_EXTERNAL_ENVIRONMENT` | Compose/configuration contract and dependency boundary | Live broker startup, event flow, restart persistence, failure/recovery observation |
| MinIO | `BLOCKED_EXTERNAL_ENVIRONMENT` | Client/object provenance contracts and Compose/configuration contract | Live object write/read/restart persistence against MinIO |
| Milvus | `BLOCKED_EXTERNAL_ENVIRONMENT` | Vector schema/client contracts and Compose/configuration contract | Live collection/write/search/restart persistence against Milvus |
| Neo4j | `BLOCKED_EXTERNAL_ENVIRONMENT` | Graph client/query/visualization contracts and Compose/configuration contract | Live graph write/read/restart persistence against Neo4j |
| CLIP `openai/clip-vit-large-patch14` | `BLOCKED_EXTERNAL_ENVIRONMENT` | Lazy adapter/model-space contract; intended Hub revision pinned to `32bd64288804d66eefd0ccbe215aa642df71cc41` | Download that snapshot, record cache/file hashes as needed, execute real text/image inference |
| WavLM `microsoft/wavlm-base-plus` | `BLOCKED_EXTERNAL_ENVIRONMENT` | Lazy adapter/separate audio-space contract; intended Hub revision pinned to `4c66d4806a428f2e922ccfa1a962776e232d487b` | Download that snapshot and execute real audio inference |
| Vosk speech path | `BLOCKED_EXTERNAL_ENVIRONMENT` | Optional dependency/failure boundary and documented local model path | Install an exact Vosk model and execute speech recognition on real audio |
| Microphone/browser/audio devices | `REQUIRES_DEVICE_ENVIRONMENT` | Optional-import/no-device failure contracts and God Music build | Permission lifecycle, capture, start/stop cleanup, browser-specific behavior |
| HRCS SDR transmit/receive/range | `REQUIRES_HARDWARE` | Packet/crypto/acoustic/simulated mesh/radio-plan software contracts; experimental TX retuning boundary | Compatible SDR, synchronized RX hopping if implemented, measured RF range/error/reliability |
| Unity editor/player | `REQUIRES_UNITY_ENVIRONMENT` | Versioned Python/Unity IPC schema and C# source contract including fragmented WebSocket handling | Real editor compile, player build, bidirectional runtime IPC on target platform |
| GPU/CUDA | `REQUIRES_GPU_ENVIRONMENT` | Heavy ML remains optional and core package does not require it | GPU/CUDA-specific inference/performance validation |
| Production deployment/security | `NOT_PRODUCTION_CERTIFIED` | Loopback Compose bindings, required secret-bearing variables, portable-import hardening, authority separation, deterministic security regression tests | Threat modeling for a specific deployment, authn/authz, service hardening, vulnerability review, monitoring, rate limits, incident/recovery testing, external review |

## Security closure observations

The active product path fails closed for portable-bundle integrity/security problems and starts new user workspaces without tool/network/filesystem authority. Model-provider replacement cannot implicitly modify that authority configuration.

Provider endpoints are validated before persistence; embedded URL credentials are rejected so they cannot be written into provider provenance or echoed through provider request errors. Active configuration uses environment-supplied secret values, the root `.env` must remain untracked, and `.env.example` contains placeholders rather than live credentials.

The static security gate is intentionally scoped to the active product surface. It does not certify every preserved historical artifact and does not replace a dependency vulnerability scanner or external security review.

## Multimodal revision discipline

The intended CLIP and WavLM Hub snapshots are pinned by exact revision in the active code and embedding-space provenance. Pinning a repository revision improves reproducibility but is not evidence that the weights were downloaded, hashed locally, loaded, or executed in this closure environment. Those remain external integration gates.

## Performance discipline

The benchmark harness reports measured values from the machine on which it runs. CI uses it as an integrity/structure check, not as a speed threshold. No CI timing number should be promoted as representative hardware or production capacity without a controlled benchmark report.

## Final exact-SHA rule

A completion statement must cite a fresh GitHub Actions run against the exact final `main` SHA. This file intentionally avoids embedding a self-referential final SHA; repository history and Actions provide the authoritative immutable linkage.
