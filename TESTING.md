# Testing and Validation Guide

The restoration branch separates deterministic software verification from external-service, browser, model, Unity, audio-device, and SDR integration work.

A passing deterministic CI run means the tested software contracts passed. It does **not** mean every optional service/hardware path was exercised.

## Deterministic Python contracts

CI runs the active deterministic suite on Python 3.11 and 3.12.

From the repository root:

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

## What those tests establish

They verify software contracts including:

- configuration loading/environment expansion;
- password encryption/decryption envelopes;
- LightToken dimensional and serialization behavior;
- raw-artifact persistence metadata/hashes;
- explicit multimodal embedding-space labeling;
- vector/memory schema alignment;
- temporal graph loading/visualization helpers with test backends;
- optional audio import boundaries;
- deterministic CST state snapshots/replay;
- hypothesis provenance/uncertainty fields;
- federated averaging/noise terminology and reproducibility;
- local Compose credential/port/hostname safety rules;
- HRCS packet/crypto/acoustic/mesh/simulated communication/radio-planning contracts;
- Python IPC schema/bridge behavior;
- Unity IPC source-level schema/async contract.

## Package-build gate

CI separately builds the root Python distribution and installs the resulting wheel into a fresh environment.

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

This catches packaging errors that can be hidden when importing directly from a checkout.

## God Music gate

From `god music/`:

```bash
npm install --no-audit --no-fund
npm test
npm run build
```

CI verifies deterministic JavaScript utilities plus the Vite production build.

## Local infrastructure integration

Before starting Docker Compose, create local untracked credentials:

```bash
cp .env.example .env
```

Replace all placeholders, then:

```bash
docker compose -f infrastructure/docker-compose.yml up -d
docker compose -f infrastructure/docker-compose.yml ps
```

Optional initialization helpers:

```bash
python infrastructure/setup_kafka.py
python infrastructure/init_milvus.py
python infrastructure/init_neo4j.py
```

These are **integration** tests, not deterministic CI results. Record exact image versions, environment, commands, and outputs when reporting them.

## Multimodal/model integration

Install the relevant extras:

```bash
python -m pip install '.[ml]'
```

Then separately record:

- model name/revision/checksum;
- preprocessing configuration;
- input fixture;
- output vector dimension/space;
- device/runtime versions.

Do not replace unavailable model output with random vectors and call the path successful.

## Audio integration

Install:

```bash
python -m pip install '.[audio]'
```

Live tests should record microphone/audio-interface model, operating system, sample rate, permissions, and whether a downloaded speech model is present.

The deterministic CI audio contracts only verify dependency boundaries and software behavior that does not require a physical microphone.

## Neo4j / visualization integration

With `infra` and `viz` extras installed and Neo4j running, verify that graph queries actually return records before judging visualization behavior.

The restoration fixed the previous class of bug where visualization code could define a query without executing it; test-backed graph behavior is covered, while a real Neo4j deployment remains an integration gate.

## HRCS testing

Selected HRCS software tests run from the root restoration workflow. A full nested-package run can be performed with:

```bash
cd coms/hrcs
python -m pip install -e '.[dev]'
pytest
```

Keep these categories separate:

- software-generated acoustic round trip;
- simulated modem/network result;
- speaker/microphone hardware result;
- SDR hardware result;
- RF field result.

Do not infer RF range, anti-jamming performance, or emergency reliability from the software tests.

## Unity integration

The restoration CI checks the Unity IPC source contract, including use of Task-based async receive code and the shared versioned envelope.

A full Unity editor/player build is not performed by the current workflow. Record Unity version, target platform, build logs, and runtime IPC evidence for any Unity integration claim.

## Validation experiments

Historical experiments under `experiments/` remain useful as hypotheses/prototypes, but their names do not make their conclusions established.

For any experiment promoted as current evidence, document:

1. hypothesis;
2. control/baseline;
3. metric and threshold;
4. seed/environment;
5. raw inputs/outputs;
6. statistical method when relevant;
7. negative/null results;
8. exact commit SHA.

Examples such as spectral clustering, frequency-dependent recall, golden-ratio stability, or communication matching should be reported as experiment outcomes, not theory validation, unless the evidence actually supports the stronger statement.

## Security testing

Current deterministic tests cover specific crypto/config contracts; they are not a full security audit.

Before external deployment, separately review:

- secret storage/rotation;
- TLS/network policy;
- authentication/authorization;
- dependency/image vulnerabilities;
- service exposure;
- key lifecycle;
- logging and sensitive-data handling;
- adversarial protocol behavior.

## Troubleshooting

Start with:

```bash
cosmic-synapse doctor --json
```

Then check only the optional layer you are actually using.

Docker:

```bash
docker compose -f infrastructure/docker-compose.yml ps
```

God Music:

```bash
cd "god music"
npm test
npm run build
```

HRCS:

```bash
cd coms/hrcs
pytest
```

## Evidence rules

Use the categories defined in `docs/REPRODUCIBILITY.md` and the claim boundary in `docs/CLAIMS_AND_LIMITATIONS.md`.

Preserve failures and blocked/null results. Do not convert a skipped hardware/service test into a pass.
