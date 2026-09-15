# Quick Start

Use this guide for the current product-facing research runtime. Historical setup guides remain preserved for lineage but may describe older dependency models or credentials.

## 1. Install the dependency-light core

Requirements: Python 3.11+ and Git.

```bash
git clone https://github.com/NavisWORLD/cosmic-synapse-A-lmi-v.2.git
cd cosmic-synapse-A-lmi-v.2
python -m pip install .
cosmic-synapse doctor --json
```

Optional extras remain opt-in:

```bash
python -m pip install '.[audio]'
python -m pip install '.[ml]'
python -m pip install '.[infra]'
python -m pip install '.[viz]'
python -m pip install '.[ipc]'
python -m pip install '.[dev]'
```

## 2. Create your continuity workspace

```bash
cosmic-synapse init ./my-cosmos --name "My Cosmos" --seed 2026 --json
cosmic-synapse inspect ./my-cosmos --json
```

A new workspace separates memory, CST software state, provider provenance, routing, artifacts/knowledge surfaces, and policy. It starts with no tool/network/filesystem authority.

## 3. Export, verify, and restore

```bash
cosmic-synapse export ./my-cosmos ./my-cosmos.cosmos --json
cosmic-synapse verify ./my-cosmos.cosmos --json
cosmic-synapse import ./my-cosmos.cosmos ./restored-cosmos --json
cosmic-synapse inspect ./restored-cosmos --json
```

The `.cosmos` bundle is integrity-addressed with SHA-256 and byte sizes. Import verifies the whole bundle before writing and rejects unsafe archive paths, symlinks, duplicate/undeclared files, corruption, unsupported versions, excessive size/count, secret-bearing filenames, and non-empty destinations.

## 4. Use a replaceable model provider

List supported provider clients without networking:

```bash
cosmic-synapse providers --json
```

The first concrete provider is Ollama. If a real local Ollama service and model are installed:

```bash
cosmic-synapse run ./my-cosmos \
  --provider ollama \
  --model qwen2:latest \
  --prompt "Continue from my saved history." \
  --json
```

The default endpoint is `http://127.0.0.1:11434`. Provider selection does not grant tool authority. A successful response means the provider call returned a structurally valid response that was persisted; it does not establish factual correctness of the response.

See `docs/PRODUCT_WORKFLOW.md` for backup/recovery and provider-swap details.

## 5. Optional local infrastructure

Docker/Compose are needed only for the Kafka/MinIO/Milvus/Neo4j research stack.

```bash
cp .env.example .env
```

Replace every placeholder locally. Required secret-bearing variables include:

```text
A_LMI_MINIO_ACCESS_KEY
A_LMI_MINIO_SECRET_KEY
A_LMI_NEO4J_PASSWORD
MILVUS_MINIO_ACCESS_KEY
MILVUS_MINIO_SECRET_KEY
```

`A_LMI_NEO4J_USERNAME` may default to `neo4j`.

Start/inspect/stop:

```bash
docker compose -f infrastructure/docker-compose.yml up -d
docker compose -f infrastructure/docker-compose.yml ps
docker compose -f infrastructure/docker-compose.yml down
```

Published development ports bind to `127.0.0.1`. The Compose file is not a production deployment template. Live service success is a separate integration gate.

## 6. Optional infrastructure initialization

With the `infra` extra and disposable/local services running:

```bash
python infrastructure/setup_kafka.py
python infrastructure/init_milvus.py
python infrastructure/init_neo4j.py
```

Review these scripts and your service configuration before use. Deterministic CI does not claim a live service deployment from configuration tests alone.

## 7. God Music

```bash
cd "god music"
npm install
npm test
npm run dev
```

Production build:

```bash
npm run build
```

CI verifies deterministic Node tests and Vite buildability. Live microphone/browser behavior requires target-device testing.

## 8. HRCS

HRCS is a nested package, not part of the root wheel:

```bash
cd coms/hrcs
python -m pip install -e '.[dev]'
pytest
```

Software tests do not establish RF range, anti-jamming superiority, synchronized receive hopping, or emergency/production readiness.

## 9. Verification

The root CI separately verifies:

- deterministic Python contracts on 3.11 and 3.12;
- portable bundle/provider/runtime/CLI security contracts;
- bounded local benchmark integrity;
- wheel/sdist build and clean-wheel install;
- an installed-wheel init/export/verify/import round trip;
- God Music tests and build.

See `TESTING.md`, `docs/REPRODUCIBILITY.md`, and `docs/FINAL_CLOSURE_EVIDENCE.md`.

## 10. External gates

A successful core install does not automatically establish:

- a live Ollama model response;
- live Kafka/MinIO/Milvus/Neo4j operation;
- downloaded CLIP/WavLM/Vosk snapshots;
- microphone/browser/device behavior;
- SDR hardware/RF performance;
- Unity editor/player runtime;
- GPU/CUDA behavior;
- production deployment/security readiness.

Those require separate recorded evidence in the actual target environment.
