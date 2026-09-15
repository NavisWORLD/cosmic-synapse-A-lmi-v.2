# History and Migration Map

The repository intentionally preserves multiple generations of ideas, demos, publications, packages, runtime code, and terminology. This map identifies the current canonical engineering path without deleting historical material.

## Preservation rule

Do not remove or rewrite a historical artifact merely because its terminology, dependencies, status claims, or implementation differ from the current product layer. Preserve lineage, make the active replacement explicit, and avoid routing new users through stale setup/security/status claims.

The pre-restoration state remains preserved at `preservation/pre-restoration-2026-09-14`.

## Current canonical paths

| Concern | Current path | Historical / alternate material |
| --- | --- | --- |
| Root Python packaging | `pyproject.toml` | older requirements/setup assumptions |
| CLI / product front door | `a_lmi/cli.py` / `cosmic-synapse` | one-off direct scripts |
| User-owned continuity | `a_lmi/continuity.py` | ad-hoc state/memory files from earlier generations |
| Provider contract | `a_lmi/providers.py` | model-specific direct calls |
| Persistent provider runtime | `a_lmi/runtime.py` | tightly coupled model/session flows |
| Bounded local benchmark | `a_lmi/benchmarking.py` | historical experiments with different goals |
| Configuration | `a_lmi/config.py`, `infrastructure/config.yaml`, `.env.example` | old hard-coded/default credentials |
| LightToken | `a_lmi/core/light_token.py` | older full-FFT/terminology variants |
| Multimodal | `a_lmi/services/multimodal_encoder.py` | random/untrained alignment fallbacks |
| Raw artifact storage | `a_lmi/memory/object_storage_client.py` | storage-key-only flows without raw upload |
| Vector memory | `a_lmi/memory/vector_db_client.py` | legacy collection/dimension assumptions |
| Temporal graph | `a_lmi/memory/tkg_client.py` | direct/older Neo4j assumptions |
| CST state/replay | `cosmic_synapse/cst_state.py` | prior CST/12D engines and theory artifacts |
| Python/Unity IPC schema | `cosmic_synapse/ipc/schema.py` | ad-hoc/unversioned shapes |
| IPC bridge | `cosmic_synapse/ipc/bridge.py` | eager transport assumptions |
| Unity IPC client | `cosmic_synapse/Unity/Assets/Scripts/IPCBridgeClient.cs` | prior coroutine/await source |
| HRCS active code | `coms/hrcs/src/hrcs/` | earlier project generations and theory docs |
| God Music web app | `god music/src/` | standalone HTML/publication generations |
| Active evidence/docs | root product docs + `docs/` | preserved papers/PDFs/ZIPs/demos/status narratives |

## Product-flow migration

The current front door is intentionally simple:

```text
install
-> cosmic-synapse init
-> inspect
-> run with an explicit provider when available
-> export .cosmos
-> verify
-> import
-> continue
```

Memory/state/authority live in the surrounding workspace, not inside a provider. Swapping providers is therefore an explicit runtime operation rather than a migration of the user's system into model parameters.

## Dependency migration

The base install is:

```bash
python -m pip install .
```

Optional groups remain explicit: `audio`, `ml`, `infra`, `viz`, `ipc`, `dev`. This replaces the earlier all-at-once dependency posture for the current package while leaving historical setup material preserved.

## Configuration / secret migration

Active Python configuration flows through `a_lmi.config.load_config`. Local infrastructure uses untracked environment values based on `.env.example`; secret-bearing active values are not intended to have committed fallback passwords.

Do not reuse credentials found in Git history or preserved artifacts.

## Spectral terminology migration

The current LightToken spectral vector is `numpy.fft.rfft` over a 1536-value embedding, yielding 769 bins. Call it an embedding rFFT representation unless an actual graph Laplacian/eigenbasis transform is introduced. Historical Graph-Fourier/physical-frequency wording remains context, not current engineering status.

## Multimodal migration

Current model-space provenance is explicit and pins intended external Hub snapshots:

- `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`
- `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b`

CLIP text/image and WavLM audio are not treated as one universal embedding space. Production success paths do not manufacture random vectors or call an untrained random projection semantic alignment.

## CST / 12D migration

Historical CST engines/files remain preserved. `cosmic_synapse.cst_state` is the current stable deterministic interface for software state, snapshots, events, serialization, and replay. Active `12D` language is computational/project terminology rather than an established physical dimensionality claim.

## Provider / authority migration

The canonical provider boundary now records provider/model identity and keeps authority in `policy/authority.json`. Provider replacement does not inherit shell/filesystem/network/cloud/deployment/actuator authority.

Custom Ollama endpoints may not embed credentials. This avoids turning endpoint configuration into persisted secret provenance.

## Federated/security terminology migration

Active docs distinguish mechanism from guarantee:

- weighted averaging is not cryptographic secure aggregation;
- Gaussian-noise injection alone is not a formal-DP guarantee;
- static/pre-shared authenticated encryption is not forward secrecy;
- a scoped static security regression is not a complete security audit.

## HRCS migration

The current verified software surface centers on packet/integrity behavior, authenticated encryption, software acoustic round trips, mesh/replay behavior, simulated E2E delivery, deterministic radio planning, and experimental TX retuning. Historical anti-jamming/range/emergency-readiness language is not promoted without hardware evidence.

## God Music migration

The active Vite app retains historical naming where useful for lineage. Describe its present behavior as algorithmic/reactive/predictive rule logic unless a real trained model is introduced and evaluated.

## Status/source migration

For current status, use this priority:

1. CI on the exact current `main` SHA;
2. `README.md`, `QUICK_START.md`, `TESTING.md`, `SYSTEM_READY.md`;
3. `docs/PRODUCT_WORKFLOW.md` and `docs/FINAL_CLOSURE_EVIDENCE.md`;
4. active architecture/claims/security/reproducibility docs;
5. historical artifacts for lineage/context.

## Adding or replacing a canonical path

1. identify the exact old behavior;
2. preserve historical evidence;
3. add a failing contract when practical;
4. implement the smallest coherent replacement;
5. preserve compatibility only where it does not hide errors;
6. update this mapping and active docs;
7. run the relevant deterministic/integration gate on the exact commit;
8. preserve failures and limitations.
