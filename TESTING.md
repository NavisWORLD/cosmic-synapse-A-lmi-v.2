# Testing and Validation Guide

COSMIC SYNAPSE / A-LMI separates deterministic software verification from external-service, browser, model, Unity, audio-device, GPU, and SDR integration work. A green CI run means the listed software contracts passed on that exact commit. It does not imply every optional service or hardware path was executed.

## Deterministic Python contracts

CI runs the active deterministic suite on Python 3.11 and 3.12. The workflow is the executable source of truth; the current suite includes:

- configuration and environment expansion;
- LightToken serialization/dimensions;
- password encryption envelopes;
- object/artifact SHA-256 provenance;
- multimodal model-space and exact revision provenance;
- vector/memory/graph contracts;
- optional audio boundaries;
- deterministic CST state/replay;
- hypothesis provenance;
- federated claim boundaries;
- packaging and Compose contracts;
- portable continuity export/import/integrity/security;
- provider/runtime swapping and authority separation;
- end-user CLI product flow;
- bounded benchmark/soak integrity;
- active-surface security regressions;
- HRCS packet/acoustic/mesh/simulated/radio contracts;
- Python/Unity IPC schema/source contracts.

Local equivalent:

```bash
python -m pip install --upgrade pip
python -m pip install pytest pytest-cov numpy scipy pyyaml cryptography
PYTHONPATH=.:coms/hrcs/src python -m pytest -q \
  tests/test_config_contract.py \
  tests/test_light_token_contract.py \
  tests/test_password_encryption.py \
  tests/test_object_storage_contract.py \
  tests/test_artifact_persistence.py \
  tests/test_multimodal_contract.py \
  tests/test_embedding_spaces.py \
  tests/test_vector_schema_contract.py \
  tests/test_memory_schema.py \
  tests/test_tkg_contract.py \
  tests/test_tkg_visualization.py \
  tests/test_audio_optional_contract.py \
  tests/test_optional_audio.py \
  tests/test_cst_state.py \
  tests/test_cst_replay.py \
  tests/test_hypothesis_provenance.py \
  tests/test_federated_claims.py \
  tests/test_packaging_contract.py \
  tests/test_compose_contract.py \
  tests/test_continuity_bundle.py \
  tests/test_provider_runtime.py \
  tests/test_cli_product_flow.py \
  tests/test_benchmarking.py \
  tests/test_security_active_surface.py \
  coms/hrcs/tests/test_packet_protocol_v2.py \
  coms/hrcs/tests/test_acoustic_roundtrip.py \
  coms/hrcs/tests/test_mesh_replay.py \
  coms/hrcs/tests/test_simulated_e2e.py \
  coms/hrcs/tests/test_radio_contract.py \
  cosmic_synapse/tests/test_ipc_schema.py \
  cosmic_synapse/tests/test_ipc_bridge_contract.py \
  cosmic_synapse/tests/test_unity_ipc_source_contract.py
```

## Security-static gate

CI also runs `tests/test_security_active_surface.py` as a separate job. It checks selected active product paths for direct dynamic-execution shortcuts, verifies the root `.env` is not tracked, confirms secret-bearing example values remain placeholders, checks active config secret fallbacks, and keeps published Compose ports loopback-bound.

This is a regression gate, not a complete penetration test, dependency vulnerability scan, or certification of historical artifacts.

## Package-build and installed-product gate

CI separately builds the wheel and source distribution, installs the wheel into a fresh virtual environment, runs `cosmic-synapse doctor --json`, imports representative core classes, and executes the portable product flow through the installed CLI:

```text
init -> inspect -> export -> verify -> import -> inspect -> providers
```

The smoke verifies that the restored workspace name and empty tool-authority state survive the round trip.

Local packaging equivalent:

```bash
python -m pip install --upgrade pip build
python -m build
python -m venv .clean-venv
. .clean-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/*.whl
cosmic-synapse doctor --json
```

## God Music gate

From `god music/`:

```bash
npm install --no-audit --no-fund
npm test
npm run build
```

This proves deterministic JavaScript behavior and Vite buildability, not live microphone behavior on a device/browser.

## Portable continuity checks

The `.cosmos` tests cover deterministic export, integrity verification, safe import, SHA-256/size corruption detection, path traversal rejection, symlink rejection, undeclared/duplicate-member rejection, secret-file exclusion, version checks, and destination safety.

Provider/runtime tests prove that memory survives a provider swap while the authority file remains unchanged. Provider failures cannot create a fabricated assistant record, and response identity mismatches fail closed.

## Multimodal/model integration

The active code pins intended model revisions:

- CLIP: `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`
- WavLM: `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b`

The deterministic suite verifies the identifiers/revisions and embedding-space boundary without downloading weights. A real integration result must separately record download/cache provenance, runtime/device versions, input fixture, output dimensions, and failures.

## Local infrastructure integration

For Kafka/MinIO/Milvus/Neo4j, create local untracked credentials from `.env.example`, start Compose, and record real service evidence:

```bash
cp .env.example .env
docker compose -f infrastructure/docker-compose.yml up -d
docker compose -f infrastructure/docker-compose.yml ps
```

When applicable, verify live startup, health, writes, reads/searches, event flow, persistence across restart, useful failure behavior, and clean shutdown. Configuration tests alone are not a live-integration pass.

## Audio, browser, Unity, GPU, and HRCS hardware

Keep these evidence classes separate:

- software/simulation result;
- browser/device result;
- Unity editor/player result;
- GPU/CUDA result;
- SDR/RF hardware result.

Static C# tests do not equal a Unity build. Synthetic acoustic tests do not equal speaker/microphone testing. Radio planning/TX code does not establish synchronized RX hopping, range, anti-jamming superiority, or field reliability.

## Benchmark discipline

`a_lmi.benchmarking.benchmark_core()` measures bounded CST and continuity operations and records environment metadata. It has no universal speed threshold. Report measured results only for the machine/environment that produced them.

## Evidence rules

For any stronger claim, retain the exact commit SHA, environment/configuration, raw outputs, controls/baselines when relevant, negative/null results, and limitations. Preserve blocked external gates as blocked; never relabel them as deterministic passes.

See `docs/REPRODUCIBILITY.md`, `docs/CLAIMS_AND_LIMITATIONS.md`, and `docs/FINAL_CLOSURE_EVIDENCE.md`.
