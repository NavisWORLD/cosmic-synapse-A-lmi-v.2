# Quick Start

This guide covers the active restoration branch. Historical setup guides may reference older dependency files, default credentials, or all-in-one startup assumptions; use this file for the current install path.

## 1. Core install

Requirements:

- Python 3.11+
- Git

```bash
git clone https://github.com/NavisWORLD/cosmic-synapse-A-lmi-v.2.git
cd cosmic-synapse-A-lmi-v.2
python -m pip install .
cosmic-synapse doctor --json
```

The default package is intentionally dependency-light.

## 2. Optional extras

Install only what you need:

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

Some extras require platform libraries, model downloads, device permissions, or substantial disk/RAM.

## 3. Local infrastructure

Docker and Docker Compose are required only if you want the local Kafka/MinIO/Milvus/Neo4j stack.

Create a local environment file from the non-secret template:

```bash
cp .env.example .env
```

Replace **every** `replace-with-*` value in `.env` before starting the stack. The root `.gitignore` excludes `.env`.

Required secret-bearing variables are:

```text
A_LMI_MINIO_ACCESS_KEY
A_LMI_MINIO_SECRET_KEY
A_LMI_NEO4J_PASSWORD
MILVUS_MINIO_ACCESS_KEY
MILVUS_MINIO_SECRET_KEY
```

`A_LMI_NEO4J_USERNAME` defaults to `neo4j` in Compose but can be overridden.

Start the local stack from the repository root:

```bash
docker compose -f infrastructure/docker-compose.yml up -d
```

Check service state:

```bash
docker compose -f infrastructure/docker-compose.yml ps
```

Stop services:

```bash
docker compose -f infrastructure/docker-compose.yml down
```

Published development ports bind to `127.0.0.1`. This Compose file is a local research/development stack, not a production deployment template.

## 4. Configuration

Active configuration lives in:

```text
infrastructure/config.yaml
```

It supports environment expansion for active service credentials. Do not put real secrets directly into tracked YAML/Markdown files.

## 5. Optional infrastructure initialization

If you installed the `infra` extra and started the local stack, the historical initialization helpers are available under `infrastructure/`:

```bash
python infrastructure/setup_kafka.py
python infrastructure/init_milvus.py
python infrastructure/init_neo4j.py
```

These commands are integration paths and are not part of the dependency-light deterministic CI gate. Run them only against a disposable/local environment until you have reviewed the scripts and your service configuration.

## 6. Optional UI / IPC surfaces

Visualization dependencies:

```bash
python -m pip install '.[viz]'
```

WebSocket IPC dependencies:

```bash
python -m pip install '.[ipc]'
```

The restoration keeps optional UI/network imports lazy so installing the core package does not require Dash, Plotly, or WebSockets.

## 7. God Music

God Music is a separate Vite/Web Audio app:

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

Live microphone analysis requires browser microphone permission. The deterministic CI result does not substitute for browser/device integration testing.

## 8. HRCS

HRCS is a separate nested Python package and is not included in the root wheel:

```bash
cd coms/hrcs
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

The restored software tests do not establish RF range, anti-jamming superiority, synchronized receive hopping, or emergency/production readiness.

## 9. Run verification

For the exact deterministic commands used by CI, see:

- `TESTING.md`
- `docs/REPRODUCIBILITY.md`

The package-build gate separately builds a wheel/sdist, installs the wheel into a fresh virtual environment, and runs `cosmic-synapse doctor --json`.

## 10. What is not automatic

A successful core install does not automatically mean the following are available:

- downloaded CLIP/WavLM/Vosk weights;
- live microphone capture;
- Kafka/MinIO/Milvus/Neo4j services;
- browser microphone behavior;
- SDR hardware;
- Unity editor/player execution;
- GPU/CUDA execution.

Use `cosmic-synapse doctor --json` to see which optional Python capabilities are present, then validate external services/hardware separately.

## Next reading

- `README.md` — project front door
- `docs/ARCHITECTURE.md` — active architecture
- `docs/PROVENANCE.md` — preservation/restoration record
- `docs/CLAIMS_AND_LIMITATIONS.md` — evidence boundary
- `docs/SECURITY.md` — security posture
- `docs/REPRODUCIBILITY.md` — exact verification model
