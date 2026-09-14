# Restoration Status

> Historical filename retained for compatibility. The presence of `SYSTEM_READY.md` does **not** mean the repository is production-ready.

## Current classification

**Research/alpha software with a reproducible deterministic core and multiple optional/integration-dependent subsystems.**

The restoration branch has converted several previously aspirational or misleading status claims into explicit contracts and evidence categories.

## Verified in deterministic CI

The active restoration workflow verifies:

- Python 3.11 and 3.12 deterministic contracts;
- configuration/startup behavior;
- LightToken dimensions/serialization;
- AES/password-envelope behavior;
- raw-artifact provenance;
- multimodal embedding-space labeling;
- vector/memory/graph helper contracts;
- optional dependency boundaries;
- deterministic CST state/replay;
- hypothesis provenance;
- federated averaging/noise claim boundaries;
- Docker Compose safety/configuration contracts;
- HRCS packet/crypto/acoustic/mesh/simulated/radio-planning software behavior;
- Python IPC and Unity IPC source contracts.

CI also runs independent jobs for:

- building the root wheel/source distribution and installing the wheel into a fresh virtual environment;
- God Music deterministic Node tests and a Vite production build.

## Implemented but not implied by core CI

The repository contains integrations for Kafka, MinIO, Milvus, Neo4j, ML models, audio capture, visualization, WebSockets, Unity, browser microphone input, and SDR hardware.

Those require environment-specific integration evidence. A green deterministic run is not a substitute for those tests.

## What the restoration corrected

Examples include:

- parsed configuration can be passed through the active runtime without being reopened as a path;
- password-based encryption retains the salt/derivation data required for decryption;
- LightToken uses an explicit 1536-value semantic vector and 769-bin one-sided FFT representation;
- the active docs no longer call that transform a Graph Fourier Transform;
- raw object bytes can be persisted with URI/hash/size provenance instead of manufacturing a storage key without an upload;
- multimodal code no longer treats random vectors/untrained random projections as successful semantic alignment;
- Neo4j visualization helpers execute/fetch graph records through testable boundaries;
- optional audio/ML/database/UI packages are no longer forced by core imports;
- CST has a canonical deterministic software-state/replay adapter;
- hypothesis outputs retain evidence/provenance/uncertainty rather than placeholder URLs;
- federated helpers no longer present simple averaging/noise as formal DP or secure aggregation;
- HRCS packet/integrity, acoustic demodulation, mesh forwarding, simulated E2E, and deterministic radio-hop behavior have explicit contracts;
- static/pre-shared-key crypto is not described as forward-secret;
- Python/Unity IPC uses a shared versioned envelope and valid async source structure;
- the root project builds as an installable wheel with lightweight defaults and optional extras;
- local Compose ports bind to loopback and secret-bearing service credentials are required from environment variables.

## What is not proven

The repository does not currently establish:

- consciousness, sentience, AGI, biological life, or identity persistence;
- new physical laws or extra physical dimensions;
- quantum advantage/consciousness;
- golden-ratio performance superiority;
- formal differential privacy or cryptographic secure aggregation;
- production security of all optional historical modules;
- RF anti-jamming superiority, range, or field reliability;
- live microphone/browser behavior across devices;
- Unity editor/player build success on every target;
- theory validation simply because an experiment file exists.

## Historical experiments

Historical spectral, recall, golden-ratio, communications, CST, resonance, music, and other experiments are preserved. Their results must be evaluated from actual data/controls before being described as validation.

## Deployment status

The current Docker Compose stack is a localhost research/development convenience. It is not a production deployment manifest.

Before production or shared-network deployment, additional work is required around service authentication/authorization, TLS, secret management, backups, monitoring, vulnerability management, resource limits, and integration testing.

## Where to look

- `README.md` — active front door
- `QUICK_START.md` — current installation/setup
- `TESTING.md` — verification and integration gates
- `docs/ARCHITECTURE.md` — active architecture
- `docs/PROVENANCE.md` — preserved history/restoration method
- `docs/CLAIMS_AND_LIMITATIONS.md` — claim boundary
- `docs/SECURITY.md` — security posture
- `docs/REPRODUCIBILITY.md` — reproducibility model

## Source of truth

For current readiness, use the status of the restoration CI on the **current branch head**, not old screenshots, old status documents, historical experiment names, or this filename.
