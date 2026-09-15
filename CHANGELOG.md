# Changelog

This changelog tracks the active restoration layer. Historical project generations remain available through Git history and preserved artifacts.

## Unreleased — 2026-09-14 restoration branch

### Preservation

- Preserved pre-restoration head `1770061052f7ac92292f7a27e84df874a4a31098` on `preservation/pre-restoration-2026-09-14`.
- Isolated restoration work on `restoration/complete-system-2026-09-14`.
- Kept historical theory/publication/demo/archive material rather than deleting or rewriting lineage.

### Added

- Deterministic restoration CI for Python 3.11 and 3.12.
- Clean wheel/source-distribution build and fresh-environment install gate.
- Dependency-light `cosmic-synapse doctor` CLI.
- Root `pyproject.toml` with optional `audio`, `ml`, `infra`, `viz`, `ipc`, and `dev` extras.
- Deterministic CST software-state/replay adapter.
- Raw artifact URI/SHA-256/size provenance helpers.
- Explicit multimodal embedding-space metadata/adaptation contracts.
- God Music deterministic Node tests and CI production build.
- Compose security/configuration contracts.
- Active architecture, provenance, security, reproducibility, claims/limitations, and migration documentation.
- Root `.gitignore` protecting local `.env`, keys, caches, build output, and Node artifacts.

### Fixed

- Parsed mapping vs path handling in active configuration flow.
- Password-based encryption salt/derivation round-trip behavior.
- LightToken spectral/vector dimension mismatch and misleading Graph Fourier terminology in active docs.
- Raw-object storage paths that previously manufactured references without persisting bytes.
- Random production embeddings and untrained random projection being treated as semantic alignment.
- Eager optional imports for audio, ML, infrastructure, visualization, and WebSocket-adjacent surfaces.
- Temporal graph/visualization query execution boundaries.
- HRCS packet/integrity, acoustic BPSK, replay/mesh forwarding, simulated end-to-end, deterministic radio hop planning, and transmit retuning boundaries.
- Unity IPC invalid coroutine/`await` structure and Python transport/schema coupling.
- Local Docker Compose Kafka hostname, host port exposure, credential handling, and volume separation.
- Stale Quick Start/Testing/System Ready instructions and hard-coded historical credentials in active docs.
- God Music active status/license metadata mismatch.

### Changed

- Active `12D` documentation now means a twelve-channel computational state representation rather than an asserted physical dimensionality.
- Active LightToken spectral wording now describes a one-dimensional `rfft` over the embedding.
- Federated helpers explicitly distinguish weighted averaging/experimental noise from formal differential privacy and cryptographic secure aggregation.
- HRCS active documentation now separates deterministic software tests from acoustic hardware, SDR/RF, anti-jamming, range, and emergency-readiness claims.
- God Music active documentation now describes deterministic/reactive/predictive rules rather than unverified “world first,” biological, or trained-AI claims.
- Docker Compose now requires explicit secret-bearing environment variables instead of fallback passwords.

### Evidence policy

- Failed/red CI runs are preserved as part of the red/green restoration record.
- Hardware/external-service/model/browser/Unity gates are not silently counted as deterministic passes.
- Strong claims are constrained by `docs/CLAIMS_AND_LIMITATIONS.md`.

## Historical versions

Earlier status/version labels inside preserved subprojects and publications are part of project history and may use different terminology or readiness claims. Use Git history and `docs/PROVENANCE.md` when reconstructing those states.
