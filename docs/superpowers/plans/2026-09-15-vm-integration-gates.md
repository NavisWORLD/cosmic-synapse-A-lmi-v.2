# VM Integration Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the remaining VM-verifiable service/model/device-path gates as real isolated CI integrations and produce exact-SHA evidence without upgrading physical-only claims.

**Architecture:** GitHub Actions Linux runners act as disposable VMs. Infrastructure, model, browser, and security gates are split into isolated jobs; repository clients perform meaningful operations and persistence jobs restart the service before final verification.

**Tech Stack:** GitHub Actions, Docker, Kafka, MinIO, Milvus, Neo4j, Ollama, Python 3.11, PyTorch CPU, Transformers, Vosk, Node/Vite, Chromium/Playwright.

**Spec:** `docs/superpowers/specs/2026-09-15-vm-integration-gates-design.md`

## Global Constraints

- No service is promoted to verified merely because a container starts.
- Persistence gates require write/read, service restart, and post-restart verification.
- Model gates require actual inference; repository/model identifiers alone are not execution evidence.
- Browser fake-media is simulation evidence, not physical microphone evidence.
- No VM result may be described as GPU, Unity-editor, SDR/RF hardware, or production-security certification.
- Preserve all historical artifacts and existing deterministic CI.

---

### Task 1: Live infrastructure VM harness

**Files:**
- Create: `integration/vm/live_services.py`
- Create: `.github/workflows/vm-integration.yml`

**Interfaces:**
- Consumes: active repository clients `ObjectStorageClient`, `VectorDBClient`, `TKGClient` and Kafka Python client.
- Produces: independent `kafka-live`, `minio-live`, `milvus-live`, `neo4j-live` workflow jobs.

- [ ] Add service-specific commands to `integration/vm/live_services.py` that fail nonzero unless the real service operation and required restart-persistence check succeed.
- [ ] Add isolated Docker-backed Actions jobs with explicit health waits and deterministic test payloads.
- [ ] Run the workflow and record every initial failure rather than weakening assertions.
- [ ] Repair repository/runtime integration defects exposed by the live services using red/green tests.
- [ ] Re-run until each service job is independently green.

### Task 2: Real Ollama inference gate

**Files:**
- Extend: `integration/vm/live_services.py`
- Extend: `.github/workflows/vm-integration.yml`

**Interfaces:**
- Consumes: `a_lmi.providers.OllamaProvider`.
- Produces: `ollama-live` job that proves real inference and explicit provider identity.

- [ ] Install/start Ollama in the VM and pull a small public model.
- [ ] Execute one bounded inference through `OllamaProvider`.
- [ ] Assert non-empty response text and expected provider/model identity.
- [ ] Preserve failures as evidence and only mark green after a real model response.

### Task 3: Real CPU multimodal execution

**Files:**
- Create: `integration/vm/live_models.py`
- Extend: `.github/workflows/vm-integration.yml`

**Interfaces:**
- Consumes: pinned CLIP/WavLM constants and `MultimodalEncoder`.
- Produces: `multimodal-live-cpu` evidence for actual text/image/audio inference plus a real Vosk engine-path execution.

- [ ] Install CPU Torch/Transformers/Pillow/soundfile/Vosk dependencies.
- [ ] Execute CLIP text and generated-image inference using the exact pinned revision.
- [ ] Execute WavLM on generated deterministic PCM audio using the exact pinned revision.
- [ ] Download a small public Vosk English model and run recognition on deterministic generated/fixture audio; label the exact model used.
- [ ] Record shape/model/revision metadata without claiming semantic quality.

### Task 4: Browser media lifecycle VM gate

**Files:**
- Create or extend: `god music` browser integration test files.
- Extend: `.github/workflows/vm-integration.yml`

**Interfaces:**
- Consumes: built God Music Vite application.
- Produces: `browser-fake-media` gate using Chromium fake media devices.

- [ ] Start the built app on loopback.
- [ ] Launch Chromium with fake media stream and automatic permission flags.
- [ ] Exercise the application's media start/stop lifecycle observable from the page.
- [ ] Assert no unhandled page error and clean stop behavior.
- [ ] Classify this as simulated device-path evidence only.

### Task 5: Security VM hardening evidence

**Files:**
- Extend: `.github/workflows/vm-integration.yml`
- Update: `docs/FINAL_CLOSURE_EVIDENCE.md`

**Interfaces:**
- Produces: `security-vm` job with Python dependency/static checks and Node dependency audit evidence.

- [ ] Run `pip-audit` against the installed root package dependencies.
- [ ] Run `bandit` against active Python product paths with explicit exclusions only where documented.
- [ ] Run `npm audit` for God Music at a defined severity threshold.
- [ ] Do not relabel a green automated scan as production certification.

### Task 6: Evidence promotion and exact-SHA closure

**Files:**
- Update: `docs/FINAL_CLOSURE_EVIDENCE.md`
- Update: `TESTING.md`
- Update: `README.md` only if gate status wording changes.

**Interfaces:**
- Consumes: completed VM workflow run IDs and exact commit SHA.
- Produces: auditable status matrix separating verified VM integrations from remaining physical gates.

- [ ] Promote only jobs that actually completed successfully.
- [ ] Leave GPU/CUDA, real Unity editor/player, SDR/RF, physical microphone, and production certification explicit where still blocked.
- [ ] Run deterministic CI and VM integration CI on the exact PR head.
- [ ] Merge only after both suites are green.
- [ ] Require fresh push-triggered deterministic and applicable VM verification on the exact resulting `main` SHA before final completion claims.
