# Changelog

This changelog tracks the active engineering layer. Historical project generations remain available through Git history and preserved artifacts.

## Unreleased — 2026-09-15 product closure

### Added

- First-class user-owned continuity workspace separating memory, CST state, provider provenance, routing, knowledge/artifact surfaces, and policy/authority from model parameters.
- Deterministic `.cosmos` export, SHA-256/size verification, safe import, format versioning, and corruption detection.
- Portable-bundle defenses for traversal, symlinks, duplicate/undeclared members, secret-bearing filenames, size/file-count limits, and non-empty destinations.
- Explicit model-provider contract with provider/model/revision/capability provenance.
- Dependency-light Ollama provider with loopback default, timeout/retry/error handling, and no network on construction.
- Persistent runtime proving provider swaps can continue against the same ledger without transferring authority.
- Product CLI commands: `init`, `inspect`, `export`, `verify`, `import`, `providers`, and `run`.
- Installed-wheel product round-trip smoke in CI.
- Bounded local CST/continuity benchmark/soak harness that reports measurements without universal performance thresholds.
- Active-surface security regression test and dedicated CI job.
- Product-facing workflow and final closure evidence documentation.

### Security

- Provider endpoints now reject embedded URL credentials before they can enter persisted provenance or provider request-error strings.
- New workspaces start with no tool/network/filesystem authority.
- Active security regression checks root `.env` tracking, placeholder secret examples, selected direct dynamic-execution shortcuts, active config secret fallbacks, and loopback Compose bindings.
- Portable continuity import verifies all declared data before materialization and fails closed on integrity/archive violations.

### Reproducibility

- Pinned intended CLIP Hub revision: `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`.
- Pinned intended WavLM Hub revision: `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b`.
- Embedding-space provenance includes the pinned revision.
- Pinning does not imply model weights were downloaded/executed; real inference remains an external integration gate.

### Documentation

- Re-centered the repository front door on the persistent AI runtime and user-owned continuity model.
- Added installation, first launch, provider switching, export/import, backup/recovery, integrity verification, integration boundaries, and troubleshooting guidance.
- Updated architecture, provenance, claims, security, reproducibility, migration, testing, status, contributor, and citation documentation to the same evidence boundary.

### Evidence policy

- Final closure requires fresh CI on the exact final PR head and, after merge, a fresh push-triggered CI run on the exact resulting `main` SHA.
- External services/models/devices/hardware remain explicitly blocked or environment-dependent unless actually executed and recorded.
- Historical artifacts remain preserved.

## 2026-09-14 restoration

### Preservation

- Preserved pre-restoration head `1770061052f7ac92292f7a27e84df874a4a31098` on `preservation/pre-restoration-2026-09-14`.
- Isolated restoration work on `restoration/complete-system-2026-09-14` before merge.
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

- Active `12D` documentation describes computational software state rather than asserted physical dimensionality.
- LightToken spectral wording describes a one-dimensional `rfft` over the embedding.
- Federated helpers distinguish weighted averaging/experimental noise from formal differential privacy and cryptographic secure aggregation.
- HRCS documentation separates deterministic software tests from acoustic hardware, SDR/RF, anti-jamming, range, and emergency-readiness claims.
- God Music documentation describes deterministic/reactive/predictive rule logic rather than unverified biological, world-first, or trained-AI claims.
- Docker Compose requires explicit secret-bearing environment variables instead of fallback passwords.

## Historical versions

Earlier status/version labels inside preserved subprojects and publications remain part of project history and may use different terminology or readiness claims. Use Git history and `docs/PROVENANCE.md` to reconstruct those states.
