# COSMIC SYNAPSE // A-LMI Complete System Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the repository's research lineage while producing an installable, testable, accurately documented active software layer.

**Architecture:** Repair existing modular boundaries instead of replacing the repository with a toy or monolith. Deterministic software-only contracts form the baseline; external services, heavyweight ML, Unity, SDR, and microphone paths are explicit higher-level gates.

**Tech Stack:** Python 3.11+, NumPy/SciPy, cryptography, Kafka/MinIO/Milvus/Neo4j optional integrations, PyTorch/Transformers optional ML, pytest, Docker Compose, Unity/C# optional, HRCS Python package.

**Spec:** `docs/superpowers/specs/2026-09-14-complete-system-restoration-design.md`

## Global Constraints

- Preserve all meaningful historical work and Git history; no destructive cleanup.
- Pre-restoration head: `1770061052f7ac92292f7a27e84df874a4a31098`.
- Never manufacture execution results, dates, scientific evidence, hardware results, or security properties.
- Keep models, memory/state, sensors, and authority as distinct system boundaries.
- Default network services to localhost and keep optional hardware/services fail-closed or gracefully unavailable.
- Every behavior repair begins with a failing automated test when the environment can exercise it.

---

### Task 1: Deterministic restoration CI and contract tests

**Files:**
- Create: `.github/workflows/restoration-ci.yml`
- Create: `tests/test_config_contract.py`
- Create: `tests/test_light_token_contract.py`
- Create: `tests/test_password_encryption.py`
- Create: `coms/hrcs/tests/test_packet_protocol_v2.py`
- Create: `coms/hrcs/tests/test_acoustic_roundtrip.py`
- Create: `coms/hrcs/tests/test_mesh_replay.py`
- Create: `coms/hrcs/tests/test_simulated_e2e.py`
- Create: `cosmic_synapse/tests/test_ipc_schema.py`

- [ ] Add tests describing the repaired contracts.
- [ ] Push them before production fixes and confirm CI fails for the expected missing/broken behavior.
- [ ] Keep the baseline independent of Kafka, MinIO, Milvus, Neo4j, microphones, SDR, Unity, and downloaded model weights.

### Task 2: Configuration, startup, encryption, and LightToken core

**Files:**
- Create: `a_lmi/config.py`
- Modify: `main.py`
- Modify: `a_lmi/core/agent.py`
- Modify: `a_lmi/core/light_token.py`
- Modify: `a_lmi/memory/vector_db_client.py`
- Modify: `a_lmi/security/encryption.py`
- Modify: `infrastructure/config.yaml`
- Create: `.env.example`

- [ ] Accept either a config mapping or config path through one canonical loader.
- [ ] Make the LightToken transform an explicit one-sided real embedding spectrum and align storage schema dimensions.
- [ ] Add a versioned password-encryption envelope carrying KDF salt, iterations, nonce, and ciphertext.
- [ ] Move active credentials to environment overrides/safe local defaults.
- [ ] Run deterministic CI to green for these contracts.

### Task 3: Raw artifacts, multimodal semantics, TKG, and visualization

**Files:**
- Modify: `a_lmi/memory/object_storage_client.py`
- Modify: `a_lmi/services/web_crawler.py`
- Modify: `a_lmi/services/audio_processor.py`
- Modify: `a_lmi/services/multimodal_encoder.py`
- Modify: `a_lmi/services/processing_core.py`
- Modify: `a_lmi/memory/tkg_client.py`
- Modify: `interfaces/visualization/graph_3d.py`
- Add focused tests under `tests/`.

- [ ] Persist actual bytes with URI + SHA-256 provenance.
- [ ] Remove random production embeddings and placeholder hashes.
- [ ] Preserve CLIP text/image alignment without pretending WavLM audio shares that semantic space.
- [ ] Execute real Neo4j graph retrieval and fix session/safe-label handling.
- [ ] Make optional audio dependencies and providers explicit and graceful.

### Task 4: HRCS protocol, crypto, modem, mesh, and simulated E2E

**Files:**
- Modify: `coms/hrcs/src/hrcs/core/packet.py`
- Modify: `coms/hrcs/src/hrcs/core/crypto.py`
- Modify: `coms/hrcs/src/hrcs/core/math.py`
- Modify: `coms/hrcs/src/hrcs/physical/acoustic.py`
- Modify: `coms/hrcs/src/hrcs/physical/radio.py`
- Modify: `coms/hrcs/src/hrcs/network/mesh.py`
- Modify: `coms/hrcs/src/hrcs/node.py`
- Create: `coms/hrcs/src/hrcs/physical/simulated.py`

- [ ] Introduce an explicit packet-v2 preamble and CRC-32 with strict lengths while retaining a documented legacy reader.
- [ ] Correct ChaCha20-Poly1305 claims and Lorenz seed range math.
- [ ] Use signed/non-orthogonal matched demodulation and framed payload lengths for synthetic acoustic round trips.
- [ ] Correct duplicate/forwarding behavior.
- [ ] Inject an in-memory transport and prove two-node encrypted message delivery without hardware.
- [ ] Use stable digest seeds; mark RF hopping experimental until receiver-synchronized retuning is implemented.

### Task 5: CST canonical active layer and deterministic replay

**Files:**
- Preserve: `cosmic420/finished/12d_cosmic_synapse_engine.py`
- Create or adapt a conventional active CST module under `cosmic_synapse/` without deleting historical generations.
- Add: `tests/test_cst_state.py`, `tests/test_cst_replay.py`.

- [ ] Test x12 bounds, m12 evolution, phase range, seed control, serialization, and replay determinism.
- [ ] Document x12/m12/12D as internal computational state and retain historical theory separately.

### Task 6: Unity IPC, hypothesis provenance, federated terminology, God Music

- [ ] Add a versioned IPC JSON schema and serialization tests.
- [ ] Repair invalid Unity coroutine/async constructs and structured status messages.
- [ ] Replace fake hypothesis search URLs with provider interfaces plus evidence/provenance/uncertainty fields.
- [ ] Rename federated privacy/aggregation methods and claims to what the code actually implements.
- [ ] Package/test deterministic God Music signal-analysis utilities while keeping live audio optional.

### Task 7: Packaging, security, docs, archive map, and reproducibility

- [ ] Add root `pyproject.toml` with minimal and optional extras.
- [ ] Validate/update Docker Compose and localhost-only active configuration.
- [ ] Add `ARCHITECTURE`, `SYSTEM_MAP`, `PROVENANCE`, `HISTORICAL_TIMELINE`, `CLAIMS_AND_LIMITATIONS`, `REPRODUCIBILITY`, `CST`, `A_LMI`, `HRCS`, `GOD_MUSIC`, `MULTIMODAL_PIPELINE`, `MEMORY_MODEL`, `SECURITY_MODEL`, `DEVELOPMENT`, and migration/path docs.
- [ ] Hash/classify important duplicates and record original paths instead of deleting them.
- [ ] Add `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`, and `SECURITY.md` where absent.
- [ ] Rewrite `README.md` as the evidence-backed cosmic control-deck front door.

### Task 8: Verification and ship

- [ ] Run deterministic CI, import/package tests, JavaScript tests where present, native build gates where real native targets exist, Docker config validation, and command/path audit.
- [ ] Run/label external-service and hardware gates separately.
- [ ] Grep active code/docs for stale secrets and overclaims.
- [ ] Record verified results with revision/timestamp and hashes where appropriate.
- [ ] Open a restoration pull request summarizing preserved, repaired, connected, experimental, blocked, and verified areas.
- [ ] Prepare a modernization release only if all required release gates actually pass.
