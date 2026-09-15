# COSMIC SYNAPSE / A-LMI

**A research-oriented persistent AI runtime that separates user-owned memory, state, provenance, routing, and authority from replaceable model providers.**

COSMIC SYNAPSE / A-LMI is preserved and modernized research software by Cory Shane Davis. The current engineering path treats the model as one replaceable component of a larger system:

```text
USER-OWNED CONTINUITY
  memory + CST software state + provenance + routing + artifact references + policy
                                  |
                                  v
                        REPLACEABLE MODEL PROVIDER
                                  |
                                  v
                       EXPLICIT AUTHORITY BOUNDARY
```

Core invariants:

- **MODEL != SYSTEM**
- **MODEL != MEMORY**
- **MODEL != AUTHORITY**

The repository also preserves the wider COSMIC SYNAPSE research lineage: HRCS communications work, God Music, Unity IPC, CST experiments, historical papers/demos/ZIPs, and optional local data infrastructure. Preservation of an artifact records provenance; it does not automatically validate every historical claim inside it.

## Status

**Research/alpha software, not a production-ready platform.**

Current deterministic CI verifies:

- Python 3.11 and 3.12 active contracts;
- portable continuity workspace creation and inspection;
- deterministic `.cosmos` export, SHA-256 verification, safe import, and corruption/security rejection;
- memory continuity across replaceable model-provider swaps while authority remains unchanged;
- dependency-light CLI product workflow;
- wheel/sdist build, clean virtualenv install, and installed-wheel portable round trip;
- bounded local CST/continuity benchmark harness without performance thresholds;
- active-surface security regression checks;
- selected HRCS packet/crypto/acoustic/mesh/simulated-E2E/radio-planning behavior;
- Python/Unity IPC schema and source contracts;
- God Music deterministic Node tests plus Vite build;
- local Compose configuration/safety contracts.

Live databases, model-weight execution, Ollama model execution, microphones/browser devices, SDR hardware, Unity editor/player execution, GPU/CUDA, and production deployment remain separate external gates. See [docs/FINAL_CLOSURE_EVIDENCE.md](docs/FINAL_CLOSURE_EVIDENCE.md).

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/NavisWORLD/cosmic-synapse-A-lmi-v.2.git
cd cosmic-synapse-A-lmi-v.2
python -m pip install .
cosmic-synapse doctor --json
```

The default install intentionally does **not** force microphone libraries, Torch/Transformers, Kafka/MinIO/Milvus/Neo4j clients, Dash/Plotly, or WebSockets.

Optional extras:

```bash
python -m pip install '.[audio]'
python -m pip install '.[ml]'
python -m pip install '.[infra]'
python -m pip install '.[viz]'
python -m pip install '.[ipc]'
python -m pip install '.[dev]'
```

## Own the continuity bundle

Create a local user-owned workspace:

```bash
cosmic-synapse init ./my-cosmos --name "My Cosmos" --seed 2026 --json
cosmic-synapse inspect ./my-cosmos --json
```

The workspace keeps continuity surfaces outside model parameters:

```text
my-cosmos/
├── system.json
├── memory/ledger.jsonl
├── state/cst.json
├── knowledge/graph.json
├── artifacts/manifest.json
├── provenance/provider.json
├── routing/state.json
└── policy/authority.json
```

A new workspace starts with no tool, network, or filesystem authority.

Export, verify, and restore it:

```bash
cosmic-synapse export ./my-cosmos ./my-cosmos.cosmos --json
cosmic-synapse verify ./my-cosmos.cosmos --json
cosmic-synapse import ./my-cosmos.cosmos ./restored-cosmos --json
```

For the complete workflow and security boundary, see [docs/PRODUCT_WORKFLOW.md](docs/PRODUCT_WORKFLOW.md).

## Replace the model, keep the user-owned history

The first concrete dependency-light provider is Ollama over HTTP. Listing providers is intentionally non-networking:

```bash
cosmic-synapse providers --json
```

With a real Ollama service/model configured:

```bash
cosmic-synapse run ./my-cosmos \
  --provider ollama \
  --model qwen2:latest \
  --prompt "Continue from my saved history." \
  --json
```

The default endpoint is `http://127.0.0.1:11434`. Custom provider endpoints reject embedded URL credentials before they can be persisted in provenance/error context.

The provider is inference, not authority. Selecting or swapping a provider does not grant shell, filesystem, network, cloud, deployment, or actuator permissions.

A successful `run` proves a structurally valid provider response was received and persisted; it does not certify the factual correctness of model output.

## Architecture

```text
A-LMI product core
  ├─ portable continuity workspace / .cosmos bundles
  ├─ replaceable provider contract + persistent runtime
  ├─ LightToken / provenance
  ├─ multimodal adapters
  ├─ object/vector/graph memory clients
  ├─ reasoning + hypothesis utilities
  └─ security helpers

COSMIC SYNAPSE
  ├─ deterministic CST software state/replay
  ├─ simulation code
  └─ versioned Python <-> Unity IPC

HRCS (nested project)
  ├─ packet + authenticated crypto
  ├─ acoustic + simulated modems
  ├─ mesh routing
  └─ experimental SDR radio path

God Music
  ├─ browser audio analysis
  ├─ deterministic timing/harmonic rules
  ├─ prediction helpers
  └─ synthesis + visualization

Optional local infrastructure
  └─ Kafka / MinIO / Milvus / Neo4j
```

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Portable-bundle security

The `.cosmos` format is a deterministic ZIP container for a frozen workspace. The manifest records SHA-256 and byte size for every declared payload.

Import/export fail closed on relevant conditions including secret-bearing filenames, path traversal/absolute/backslash archive paths, symlinks, duplicate/undeclared members, missing declared payloads, size/hash mismatch, unsupported versions, bounded file-count/uncompressed-size limits, and importing over a non-empty destination.

The separate CI security-static job also checks selected active product paths for direct dynamic-execution shortcuts, root `.env` tracking, secret placeholders/fallbacks, and loopback Compose port bindings. It is a regression gate, not a complete security audit.

See [docs/SECURITY.md](docs/SECURITY.md).

## CST and `12D` terminology

Historical CST/`12D` terminology remains preserved. In the canonical active adapter, the state is a bounded computational/software representation with deterministic seed, replay, phase, diagnostic energy/entropy, and serialization. It is not evidence of experimentally established extra physical dimensions.

## LightToken and multimodal boundary

The active LightToken contract uses a 1536-value semantic embedding and a 769-bin `numpy.fft.rfft` representation. The spectral vector is a software transform, not a physical-frequency or Graph Fourier Transform claim.

The intended optional external model snapshots are pinned in active code/provenance:

- CLIP text/image: `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`
- WavLM audio/speech: `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b`

CLIP text/image and WavLM audio are not treated as one universal space. Random production embeddings and untrained random projections are not treated as successful semantic alignment.

Revision pinning identifies the intended snapshot; dependency-light CI does **not** claim those weights were downloaded or executed.

## Local infrastructure

Use `.env.example` as the variable-name guide, supply your own local-development credentials, then:

```bash
docker compose -f infrastructure/docker-compose.yml up -d
```

Published Compose ports bind to `127.0.0.1`, and secret-bearing MinIO/Milvus/Neo4j values have no active fallback password. This Compose file is a local research/development configuration, not a production deployment template.

## Measured local benchmark

`a_lmi.benchmarking.benchmark_core()` measures bounded CST replay plus portable bundle export/verify/import on the machine where it is executed. The result includes environment metadata, elapsed time, operation counts, integrity outcome, and error count.

There is intentionally no CI performance threshold and no cross-machine/production-capacity claim.

## HRCS

`coms/hrcs/` is a separate nested research project. Deterministic tests cover packet/integrity, authenticated symmetric crypto contracts, acoustic software round trips, replay/mesh behavior, simulated node-to-node communication, and deterministic radio planning/transmit retuning boundaries.

They do not establish synchronized receive hopping, anti-jamming superiority, RF range/reliability, or emergency/production readiness. Actual RF claims require compatible SDR hardware and measurement.

## God Music

`god music/` is an algorithmic browser music experiment. CI runs deterministic Node tests and a Vite production build. Live microphone/browser behavior remains a device integration gate. Active docs do not claim biological inference, trained musical intelligence, “world first” status, or golden-ratio superiority.

## Verification and evidence

- [TESTING.md](TESTING.md)
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- [docs/FINAL_CLOSURE_EVIDENCE.md](docs/FINAL_CLOSURE_EVIDENCE.md)
- [docs/CLAIMS_AND_LIMITATIONS.md](docs/CLAIMS_AND_LIMITATIONS.md)
- [docs/PROVENANCE.md](docs/PROVENANCE.md)

Pre-restoration state remains preserved at `preservation/pre-restoration-2026-09-14`. The completed restoration lineage remains visible through merged history. Product-closure work is developed through review/CI rather than history rewrite.

## Claims this repository does not establish

Current evidence does not establish consciousness, sentience, AGI, biological life, a soul, identity resurrection/persistence, new physics, extra physical dimensions, quantum advantage/consciousness, golden-ratio superiority, formal differential privacy, cryptographic secure aggregation, RF anti-jamming superiority, or unmeasured hardware performance.

## Project structure

```text
.
├── a_lmi/                  # installable persistent-runtime Python core
├── cosmic_synapse/         # CST state/simulation/IPC + Unity source
├── coms/hrcs/              # nested communications research project
├── god music/              # Vite/Web Audio experiment
├── interfaces/             # optional UI/visualization surfaces
├── infrastructure/         # local config/Compose/init helpers
├── experiments/            # historical/current experiments
├── tests/                  # deterministic software contracts
├── docs/                   # active architecture/evidence/product docs
└── pyproject.toml          # root package + optional extras
```

## License

The repository root is GPL-3.0; see [LICENSE](LICENSE). Some preserved/nested historical components contain separate license metadata/files, including HRCS. This project does not silently rewrite those terms.

## Author

Cory Shane Davis
