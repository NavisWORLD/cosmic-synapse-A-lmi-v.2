# History and Migration Map

The repository intentionally contains multiple generations of ideas, demos, publications, packages, and runtime code. This map helps contributors choose the active path without deleting historical material.

## Preservation rule

Do not delete or rewrite a historical artifact merely because its terminology, dependency choices, status claims, or implementation differ from the active restoration layer.

Instead:

- preserve the artifact;
- make the active replacement/path explicit;
- add compatibility only where it is useful and testable;
- avoid routing new users through stale setup/status documents.

The pre-restoration state is preserved on `preservation/pre-restoration-2026-09-14`.

## Active paths

| Concern | Active path | Historical / alternate material |
| --- | --- | --- |
| Root Python packaging | `pyproject.toml` | `requirements.txt` and older setup instructions |
| Core diagnostics | `a_lmi/cli.py` / `cosmic-synapse doctor` | direct one-off import/start scripts |
| Configuration | `a_lmi/config.py`, `infrastructure/config.yaml`, `.env.example` | hard-coded/default credentials in older revisions/docs |
| LightToken | `a_lmi/core/light_token.py` | older full-FFT/terminology variants |
| Raw artifact storage | `a_lmi/memory/object_storage_client.py` | storage-key-only flows without raw upload |
| Vector memory | `a_lmi/memory/vector_db_client.py` | legacy collection/dimension assumptions |
| Temporal graph | `a_lmi/memory/tkg_client.py` | older direct Neo4j-only assumptions |
| CST state/replay | `cosmic_synapse/cst_state.py` | prior CST/12D generations and theory artifacts |
| Python/Unity IPC schema | `cosmic_synapse/ipc/schema.py` | ad-hoc/unversioned message shapes |
| IPC bridge | `cosmic_synapse/ipc/bridge.py` | eager WebSocket dependency / legacy send assumptions |
| Unity IPC client | `cosmic_synapse/Unity/Assets/Scripts/IPCBridgeClient.cs` | prior coroutine/`await` source |
| HRCS active implementation | `coms/hrcs/src/hrcs/` | theory/docs and earlier assumptions inside nested project history |
| God Music active web app | `god music/src/` | older standalone HTML/publication generations |
| Active project documentation | root README + root quick/testing/status files + `docs/` | historical papers, PDFs, ZIPs, demos, old status narratives |

## Dependency migration

### Old pattern

```text
pip install -r requirements.txt
```

This pulled the project toward an all-at-once dependency model and made optional stacks appear mandatory.

### Active pattern

```bash
python -m pip install .
```

Optional groups are explicit:

```text
audio
ml
infra
viz
ipc
dev
```

This keeps the core importable/installable without PyAudio, Torch/Transformers, database clients, Dash/Plotly, or WebSockets.

## Configuration migration

Active Python configuration should flow through `a_lmi.config.load_config`, which accepts either a mapping or a YAML path and expands `${NAME}` / `${NAME:-default}` placeholders.

For local Docker infrastructure:

- use `.env.example` as a variable-name template;
- create a local untracked `.env`;
- supply required secret-bearing values explicitly;
- do not reuse credentials found in Git history.

## Spectral terminology migration

### Historical wording

Some earlier documentation described the embedding-domain transform as a Graph Fourier Transform or used stronger frequency/physical interpretations.

### Active wording

The current LightToken spectral vector is `numpy.fft.rfft` over a 1536-value embedding, producing 769 complex-frequency bins before serialization/representation handling.

Call it a one-dimensional FFT/rFFT representation of the software embedding unless a genuine graph Laplacian/eigenbasis transform is introduced and tested.

## Multimodal migration

### Historical behavior

Some paths could return random vectors or use an untrained random projection while describing the result as aligned/shared semantics.

### Active behavior

- production success paths do not manufacture random embeddings;
- embedding-space identity is explicit;
- CLIP text/image and WavLM audio are not assumed to be one universal space;
- any future cross-space alignment must identify the trained/algebraic mapping and its evidence.

## CST / 12D migration

Historical CST engines/files remain preserved. The active `cosmic_synapse.cst_state` adapter is the stable engineering interface for deterministic state, snapshots, events, and replay.

`12D` in active engineering docs means a twelve-channel computational state representation, not an experimentally established physical dimensionality claim.

## Federated/security terminology migration

Compatibility methods may retain names used by earlier code, but active docs distinguish mechanism from guarantee:

- weighted averaging is federated aggregation logic;
- Gaussian-noise injection alone is not a complete formal-DP guarantee;
- ordinary averaging is not cryptographic secure aggregation;
- static/pre-shared authenticated encryption is not forward secrecy.

## HRCS migration

The active HRCS test surface now centers on packet/integrity behavior, authenticated encryption, acoustic software round trips, mesh forwarding/replay handling, simulated end-to-end delivery, and deterministic radio hop planning/transmit retuning.

Do not migrate historical anti-jamming, zero-infrastructure, optimality, range, or emergency-readiness language into active status unless separately demonstrated.

## God Music migration

The active Vite app remains under `god music/` and retains historical phi/bio/psi class names where they are part of code lineage.

Describe the present engine as algorithmic/reactive/predictive rule logic unless a trained model is actually integrated and evaluated.

## Status-document migration

Files with names such as `SYSTEM_READY.md`, old implementation-complete notes, publications, and historical READMEs may contain stronger status language from earlier development phases.

For current project status, use this priority order:

1. CI result on the current restoration head
2. root `README.md`
3. `TESTING.md` / `QUICK_START.md`
4. `docs/ARCHITECTURE.md`
5. `docs/CLAIMS_AND_LIMITATIONS.md`
6. `docs/SECURITY.md` / `docs/REPRODUCIBILITY.md`
7. preserved historical artifacts for lineage/context

## Adding a new canonical path

When replacing another historical path:

1. identify the exact old behavior;
2. add a failing contract if behavior is executable/testable;
3. implement the smallest replacement;
4. preserve compatibility only where it does not hide errors;
5. document old → new path here;
6. run the relevant deterministic/integration gate;
7. preserve failure evidence and limitations.
