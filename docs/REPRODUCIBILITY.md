# Reproducibility Guide

This restoration separates deterministic software verification from integration/hardware validation so a passing CI run has a precise meaning.

## Deterministic CI

The restoration workflow runs on Python 3.11 and 3.12 and covers the active contracts for configuration, crypto envelopes, LightToken/vector dimensions, raw-artifact provenance, multimodal-space labeling, memory/graph helpers, optional audio boundaries, CST state/replay, hypothesis provenance, federated terminology, Docker Compose safety, HRCS software behavior, Python IPC, and Unity IPC source contracts.

Run the same dependency-light suite from a checkout with:

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
  coms/hrcs/tests/test_packet_protocol_v2.py \
  coms/hrcs/tests/test_acoustic_roundtrip.py \
  coms/hrcs/tests/test_mesh_replay.py \
  coms/hrcs/tests/test_simulated_e2e.py \
  coms/hrcs/tests/test_radio_contract.py \
  cosmic_synapse/tests/test_ipc_schema.py \
  cosmic_synapse/tests/test_ipc_bridge_contract.py \
  cosmic_synapse/tests/test_unity_ipc_source_contract.py
```

## Package build gate

CI separately builds the Python source distribution and wheel, installs the built wheel into a fresh virtual environment, runs:

```bash
cosmic-synapse doctor --json
```

and imports representative core classes. This prevents a checkout-only import path from being mistaken for a valid installed package.

Local equivalent:

```bash
python -m pip install --upgrade pip build
python -m build
python -m venv .clean-venv
. .clean-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install dist/*.whl
cosmic-synapse doctor --json
```

On Windows PowerShell, activate the environment with the platform-appropriate script instead of `. .clean-venv/bin/activate`.

## God Music gate

From `god music/`:

```bash
npm install --no-audit --no-fund
npm test
npm run build
```

CI runs both the deterministic Node tests and a Vite production build. That verifies utility behavior/buildability, not microphone hardware or every browser/device.

## Core installation

```bash
python -m pip install .
cosmic-synapse doctor
```

Optional extras can be installed individually or together:

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

Some extras may require platform system libraries or large model downloads.

## Local infrastructure

Copy the variable names from `.env.example` into your shell or local secret-loading mechanism and replace all placeholder values before starting shared/non-disposable environments.

Start the local stack from the repository root with:

```bash
docker compose -f infrastructure/docker-compose.yml up -d
```

The active Compose file binds published service ports to `127.0.0.1`.

Check status with:

```bash
docker compose -f infrastructure/docker-compose.yml ps
```

Stop it with:

```bash
docker compose -f infrastructure/docker-compose.yml down
```

Persisted volumes are not deleted by the normal `down` command.

## Integration gates that CI does not claim

The deterministic workflow intentionally does not require or claim success for:

- a live Kafka broker pipeline;
- real MinIO/Milvus/Neo4j persistence across all production paths;
- downloaded CLIP/WavLM/Vosk weights;
- live microphone capture;
- browser microphone routing on every supported browser;
- SDR transmit/receive hardware;
- synchronized RF frequency hopping;
- a Unity editor/player build;
- internet/network crawler behavior;
- GPU/CUDA execution.

When validating one of these, record the exact hardware/service versions, operating system, configuration, model revision/checksum, commands, raw outputs, and failure cases.

## Determinism and seeds

CST replay contracts expose explicit seeds/state snapshots. HRCS radio hop planning derives its deterministic seed from SHA-256 data rather than Python's process-randomized `hash()`.

Whenever randomness is part of an experiment, record the seed and distinguish deterministic replay from statistical evidence.

## Evidence categories

Use these labels consistently in reports:

- **verified software result** — deterministic test/build passed;
- **integration result** — external service/model path executed in a specified environment;
- **hardware result** — physical device path executed with recorded hardware/config;
- **simulation result** — produced by a modeled/simulated environment;
- **hypothesis** — proposed relationship not yet established;
- **historical claim** — preserved statement from earlier project material;
- **blocked/null result** — attempted test could not establish the target result.

Do not promote one category into another without new evidence.

## Provenance

Use `docs/PROVENANCE.md` for the preservation branch/base commit and the restoration method. Keep raw logs/artifacts and commit SHAs alongside any future benchmark or scientific report so results can be traced back to the implementation that produced them.
