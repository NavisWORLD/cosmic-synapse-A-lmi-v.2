# Active Architecture

COSMIC SYNAPSE / A-LMI is a research-oriented persistent AI runtime built around one central separation:

```text
MODEL != SYSTEM
MODEL != MEMORY
MODEL != AUTHORITY
```

The model is a replaceable inference component. User-owned continuity, software state, provenance, routing, storage, policy, and authority belong to the surrounding runtime.

Historical papers, demos, ZIPs, earlier CST engines, and terminology remain preserved and are not silently promoted into active-runtime claims.

## Product spine

```text
USER / EVENT
    |
    v
PORTABLE CONTINUITY WORKSPACE
    |-- system identity + format version
    |-- memory ledger
    |-- CST computational state
    |-- knowledge / artifact references
    |-- provider provenance
    |-- routing state
    `-- policy / authority
             |
             v
EXPLICIT MODEL PROVIDER CONTRACT
             |
             v
MODEL RESPONSE + PROVENANCE
             |
             v
POLICY / AUTHORITY BOUNDARY
             |
             +--> optional tools/services/devices only when separately authorized
```

A provider swap changes the inference component, not ownership of memory/state and not authority. Deterministic tests verify that provider replacement preserves prior memory and leaves the authority file unchanged.

## Portable continuity workspace

`a_lmi.continuity` defines the versioned user-owned workspace and deterministic `.cosmos` bundle format. The canonical workspace contains:

- `system.json`
- `memory/ledger.jsonl`
- `state/cst.json`
- `knowledge/graph.json`
- `artifacts/manifest.json`
- `provenance/provider.json`
- `routing/state.json`
- `policy/authority.json`

New workspaces start with no tool, network, or filesystem authority. Export records SHA-256 and byte sizes for all declared payloads. Import verifies the archive before materialization and rejects unsafe paths, symlinks, duplicate/undeclared members, corruption, secret-bearing filenames, unsupported versions, oversize/file-count limits, and non-empty destinations.

## Model provider boundary

`a_lmi.providers` defines explicit provider identity, request, response, health, capabilities, model ID/revision, endpoint, timeout/retry behavior, and provenance. `a_lmi.runtime.PersistentRuntime` persists interactions into the surrounding user workspace.

The first concrete dependency-light provider is Ollama. Its default endpoint is loopback and construction performs no network access. Embedded endpoint credentials are rejected before the endpoint can enter persisted provider provenance or request error reporting.

Provider output is not authority. A provider cannot automatically gain shell, filesystem, network, cloud, deployment, actuator, or service permissions.

## A-LMI data path

The wider A-LMI research path remains independently usable:

```text
input/event
   |
   v
modality/service adapter
   |
   +--> raw artifact persistence (optional MinIO)
   |
   +--> semantic/model embedding (optional ML extra)
   |
   +--> LightToken
          |-- semantic embedding: 1536 values
          |-- spectral representation: 769-bin real FFT
          |-- provenance / modality / storage metadata
          |
          +--> vector memory (optional Milvus)
          +--> temporal graph (optional Neo4j)
          +--> reasoning / hypothesis utilities
```

The 769-bin representation is `numpy.fft.rfft` over a 1536-value software embedding. It is not a Graph Fourier Transform and is not evidence of a new physical frequency domain.

## Multimodal boundary

The active multimodal adapter preserves embedding-space identity and pins the intended Hub snapshots:

- CLIP text/image: `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`
- WavLM audio/speech: `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b`

CLIP text/image may share their pretrained space. WavLM audio is a separate space unless a real trained alignment is introduced. The 1536-value carrier adaptation is deterministic; random production vectors/untrained random projections are not treated as successful semantic inference.

Pinning the revision specifies the intended external snapshot. It does not imply the weights were downloaded or executed by dependency-light CI.

## Canonical CST software state

`cosmic_synapse.cst_state` is the stable deterministic adapter for the historical CST/12D engineering lineage. It exposes explicit seeded state, phase, memory terms, snapshots, serialization, and replay.

Historical `12D` naming is retained as software/project terminology. The active adapter does not establish extra physical dimensions or new physics.

## Memory and provenance

The architecture separates:

- raw object bytes + URI + SHA-256 + byte size;
- semantic/spectral vectors + explicit embedding-space identity;
- temporal graph records;
- portable continuity memory/state;
- provider/model provenance;
- authority configuration.

MinIO, Milvus, and Neo4j remain optional lazy integrations rather than requirements for the base package.

## Security / authority boundary

Active software protections include authenticated encryption helpers, password envelopes with persistent derivation metadata, portable archive hardening, secret-file exclusion, localhost Compose bindings, env-driven service secrets, provider endpoint credential rejection, and a scoped static security regression gate.

These protections do not amount to production authentication/authorization, complete dependency vulnerability analysis, or a third-party security audit.

## HRCS boundary

HRCS preserves versioned packets, authenticated symmetric encryption, software acoustic round trips, replay handling, simulated multi-hop communication, deterministic radio planning, and experimental transmit retuning when compatible hardware exists.

The current evidence does not establish forward secrecy from static/pre-shared keys, synchronized RX hopping, anti-jamming superiority, field range, or emergency reliability.

## IPC / Unity boundary

Python and Unity share a versioned JSON envelope. Python transport handling is optional and dependency-light; Unity source uses Task-based async receive logic and handles fragmented WebSocket messages. CI verifies source/schema contracts. A real Unity editor/player compile/build/runtime remains an environment-specific gate.

## God Music boundary

God Music remains an algorithmic Vite/Web Audio research app. CI verifies deterministic Node tests and the production build. Live browser microphone behavior is a separate device/browser result.

## Optional infrastructure

`infrastructure/docker-compose.yml` supplies a localhost research stack for Kafka, MinIO, Milvus, Neo4j and dependencies. Published host ports bind to `127.0.0.1`; secret-bearing values come from environment variables. Configuration contracts do not substitute for live service startup/write/read/restart evidence.

## Packaging and product interface

The root package keeps lightweight defaults with explicit extras: `audio`, `ml`, `infra`, `viz`, `ipc`, `dev`.

The installed `cosmic-synapse` CLI exposes diagnostics, CST demo, continuity initialization/inspection/export/import/verification, provider listing, and explicit provider execution. CI builds the wheel/sdist, installs the wheel into a clean virtual environment, and runs the portable continuity round trip through the installed executable.

## Verification layers

The active workflow separates:

- deterministic Python contracts on Python 3.11 and 3.12;
- a scoped security-static regression job;
- wheel/sdist + clean-install + installed-product smoke;
- God Music Node tests/build;
- explicit external/model/device/hardware gates documented separately.

See `FINAL_CLOSURE_EVIDENCE.md` for the exact verified-versus-external boundary.
