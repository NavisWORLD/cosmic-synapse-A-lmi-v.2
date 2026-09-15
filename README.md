# COSMIC SYNAPSE / A-LMI

Preserved and modernized research software by Cory Shane Davis.

This repository contains the A-LMI Python package, COSMIC SYNAPSE computational-state/simulation code, HRCS communications research, the God Music browser experiment, Unity IPC source, historical theory/publication artifacts, and optional local data infrastructure.

The active restoration layer is intentionally evidence-bounded: historical terminology and experiments are preserved, while the current README/docs distinguish verified software behavior from theory, hypothesis, simulation, optional integration, hardware work, and unverified claims.

## Current status

**Research/alpha software, not a production-ready platform.**

The restoration branch currently verifies:

- deterministic Python contracts on Python 3.11 and 3.12;
- root wheel/sdist build plus installation into a fresh virtual environment;
- dependency-light CLI diagnostics;
- God Music deterministic Node tests plus a Vite production build;
- local Docker Compose safety/configuration contracts;
- selected HRCS packet, crypto, acoustic, mesh, simulated E2E, and radio-planning behavior;
- Python/Unity IPC schema/source contracts.

The fully green code/config checkpoint after Compose hardening is `cdbd5b9c0cb5f2071437e33cbf5a89241881e8d2` (Restoration CI run #102). Later commits on the restoration branch are documentation/metadata cleanup unless otherwise noted.

## Quick start: dependency-light core

Requires Python 3.11+.

```bash
git clone https://github.com/NavisWORLD/cosmic-synapse-A-lmi-v.2.git
cd cosmic-synapse-A-lmi-v.2
python -m pip install .
cosmic-synapse doctor --json
```

The default install intentionally does **not** force microphone libraries, Torch/Transformers, Kafka/MinIO/Milvus/Neo4j clients, Dash/Plotly, or WebSockets.

Install only the optional surfaces you need:

```bash
python -m pip install '.[audio]'
python -m pip install '.[ml]'
python -m pip install '.[infra]'
python -m pip install '.[viz]'
python -m pip install '.[ipc]'
python -m pip install '.[dev]'
```

For a broad research environment:

```bash
python -m pip install '.[audio,ml,infra,viz,ipc,dev]'
```

See [QUICK_START.md](QUICK_START.md) for local infrastructure and optional integration setup.

## Architecture

The active system is split into independently testable surfaces:

```text
A-LMI Python core
  ├─ LightToken / provenance
  ├─ multimodal adapters
  ├─ object/vector/graph memory clients
  ├─ reasoning + hypothesis utilities
  └─ security helpers

COSMIC SYNAPSE
  ├─ deterministic CST software state/replay
  ├─ simulation code
  └─ versioned Python ↔ Unity IPC

HRCS (nested project)
  ├─ packet + authenticated crypto
  ├─ acoustic + simulated modems
  ├─ mesh routing
  └─ experimental SDR radio path

God Music (web app)
  ├─ audio analysis
  ├─ deterministic timing/harmonic rules
  ├─ prediction helpers
  └─ synthesis + visualization

Optional local infrastructure
  └─ Kafka / MinIO / Milvus / Neo4j
```

Full architecture details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## LightToken and spectral representation

The active LightToken contract uses a 1536-value semantic embedding and a 769-bin `numpy.fft.rfft` representation.

That 769-bin vector is a one-dimensional Fourier transform of software embedding values. It is **not** described as a Graph Fourier Transform or as evidence of a new physical frequency domain.

## Multimodal boundary

The restoration does not pretend all model outputs occupy one universal semantic space.

- CLIP-family text/image vectors may share their model's learned space.
- WavLM-family audio vectors are labeled as a separate embedding space unless a trained alignment is explicitly supplied.
- Random production embeddings and untrained random “alignment” projections are not accepted as successful inference.

## CST / `12D` terminology

Historical CST and `12D` terminology is preserved. In the canonical active adapter, `12D` refers to a twelve-channel computational state representation (`x12`/`m12`, phase, seed, snapshots, replay). It is not a claim that physical spacetime has twelve experimentally established dimensions.

## Memory and provenance

The active path distinguishes:

- raw object bytes + storage URI + SHA-256 + byte size;
- semantic/spectral vector records with explicit embedding-space metadata;
- temporal graph records for query/visualization.

MinIO, Milvus, and Neo4j are optional integrations rather than import-time requirements.

## Security boundary

Verified dependency-light security includes AES-GCM helpers and password envelopes that preserve the salt/derivation data required for decryption.

The active documentation does **not** claim:

- forward secrecy from static/pre-shared keys;
- formal differential privacy from simple Gaussian-noise injection;
- cryptographic secure aggregation from ordinary federated averaging;
- a third-party audit of every preserved security experiment.

See [docs/SECURITY.md](docs/SECURITY.md).

## HRCS

`coms/hrcs/` is a separate nested Python project. The restoration verifies software packet/crypto/acoustic/mesh/simulation behavior and deterministic radio hop planning/transmit retuning boundaries.

It does not establish synchronized receive hopping, anti-jamming superiority, RF range/reliability, or emergency/production readiness. See [coms/hrcs/README.md](coms/hrcs/README.md).

## God Music

`god music/` is an algorithmic browser music experiment with audio analysis, deterministic timing/harmonic rules, predictive helpers, synthesis, and visualization.

CI runs its Node tests and a Vite production build. Live microphone/browser behavior remains an integration surface. The active docs do not claim biological inference, trained musical intelligence, “world first” status, or golden-ratio superiority.

## Local infrastructure

Set local-development credential variables using `.env.example` as a name/reference guide, then start the stack from the repository root:

```bash
docker compose -f infrastructure/docker-compose.yml up -d
```

Published development ports bind to `127.0.0.1`. The Compose stack is a local research convenience, not a production deployment template.

## Verification

See [TESTING.md](TESTING.md) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

CI separates deterministic verification into:

1. Python 3.11 contracts
2. Python 3.12 contracts
3. clean package build/install smoke test
4. God Music tests/build

External databases, downloaded model weights, microphones, SDR hardware, browser behavior, and Unity editor/player execution require separate integration/hardware evidence.

## Historical material and provenance

The project deliberately keeps earlier papers, PDFs, ZIPs, HTML demos, Unity material, terminology, and experimental generations. Their presence documents the project lineage; it does not automatically validate every statement inside them.

See:

- [docs/PROVENANCE.md](docs/PROVENANCE.md)
- [docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

Pre-restoration state is preserved at:

`preservation/pre-restoration-2026-09-14`

Restoration work is isolated at:

`restoration/complete-system-2026-09-14`

## Claims this repository does not currently establish

The current evidence does not establish consciousness, sentience, AGI, biological life, a soul, identity persistence, new physics, extra physical dimensions, quantum advantage/consciousness, golden-ratio performance superiority, formal DP, secure aggregation, RF anti-jamming superiority, or unmeasured hardware performance.

For the exact boundary and criteria for promoting stronger claims, read [docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md).

## Project structure

```text
.
├── a_lmi/                  # installable Python core
├── cosmic_synapse/         # state/simulation/IPC + Unity source
├── coms/hrcs/              # nested HRCS communications project
├── god music/              # Vite/Web Audio music experiment
├── interfaces/             # optional UI/visualization surfaces
├── infrastructure/         # local config/Compose/init helpers
├── experiments/            # historical/current experiment code
├── tests/                  # restoration contracts
├── docs/                   # active architecture/evidence docs
└── pyproject.toml          # root package + optional extras
```

## License

The repository root is licensed under GPL-3.0; see [LICENSE](LICENSE).

Some preserved/nested historical components contain their own license metadata/files. In particular HRCS has a nested license file. The restoration does not silently rewrite those legal terms; redistribution of combined/derived work should follow the applicable licenses and, when needed, appropriate legal review.

## Author

Cory Shane Davis
