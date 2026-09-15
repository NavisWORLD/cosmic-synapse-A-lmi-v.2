# Product Workflow

COSMIC SYNAPSE / A-LMI is a research-oriented persistent AI runtime. The user-owned continuity workspace stores memory, computational state, provenance, routing state, artifact/knowledge references, and policy/authority separately from the model provider.

The model is replaceable. The workspace is the continuity boundary.

## Install

Python 3.11+:

```bash
python -m pip install .
cosmic-synapse doctor --json
```

The default package stays dependency-light. Optional ML, audio, infrastructure, visualization, and IPC extras remain opt-in.

## Create a user-owned workspace

```bash
cosmic-synapse init ./my-cosmos --name "My Cosmos" --seed 2026 --json
```

A new workspace starts with:

- `system.json` — workspace identity/version;
- `memory/ledger.jsonl` — append-only local interaction records;
- `state/cst.json` — deterministic CST software state;
- `knowledge/graph.json` — portable graph export surface;
- `artifacts/manifest.json` — artifact references/provenance surface;
- `provenance/provider.json` — selected/last provider identity/provenance;
- `routing/state.json` — routing state;
- `policy/authority.json` — authority configuration.

New workspaces grant no tool, network, or filesystem authority by default.

## Inspect without networking

```bash
cosmic-synapse inspect ./my-cosmos --json
cosmic-synapse providers --json
```

`inspect` reads only local continuity state. `providers` lists supported provider clients without contacting a service. A listed provider is not the same as an available provider.

## Use an explicit model provider

The first dependency-light concrete provider is Ollama over its HTTP API. Construction performs no network access; `run` performs the actual request.

```bash
cosmic-synapse run ./my-cosmos \
  --provider ollama \
  --model qwen2:latest \
  --prompt "Continue from my saved history." \
  --json
```

The default endpoint is `http://127.0.0.1:11434`. Custom endpoints must use HTTP(S), contain a hostname, and must not embed `user:password@host` credentials because provider identity/provenance is user-visible persistent state.

A successful `run` verifies that the selected provider returned a structurally valid response that the runtime persisted. It does **not** certify the factual correctness of model output.

Provider selection never grants tools, shell, filesystem, cloud, deployment, network, or actuator authority. Those remain separate policy concerns.

## Swap the model without swapping the story

Run the same workspace with another explicitly selected model/provider. Memory and workspace state remain outside provider parameters.

```bash
cosmic-synapse run ./my-cosmos \
  --provider ollama \
  --model another-model:latest \
  --prompt "Continue." \
  --json
```

Deterministic tests verify that provider swaps preserve the existing memory ledger and leave the authority file unchanged. The system does not claim that two providers have the same internal identity or behavior; it claims that the surrounding software history remains available across the swap.

## Export a portable continuity bundle

```bash
cosmic-synapse export ./my-cosmos ./my-cosmos.cosmos --json
cosmic-synapse verify ./my-cosmos.cosmos --json
```

The `.cosmos` file is a deterministic ZIP container for a frozen workspace. Its manifest records SHA-256 and byte size for every declared payload.

Default export rejects secret-bearing filenames such as `.env*`, private-key formats, and common credential files. Symlinks are not portable bundle payloads.

## Import and continue

```bash
cosmic-synapse import ./my-cosmos.cosmos ./restored-cosmos --json
cosmic-synapse inspect ./restored-cosmos --json
```

Import verifies the complete archive before materializing files and rejects absolute/traversal/backslash paths, symlinks/directories, duplicate/undeclared members, missing payloads, size/SHA mismatches, unsupported versions, oversized bundles/file counts, secret-bearing members, and non-empty destinations.

## Backup and recovery

Treat the verified `.cosmos` bundle as the portable user-owned backup artifact. Keep service credentials outside the bundle. After copying a bundle to another machine, run `cosmic-synapse verify` before import.

Format versioning is explicit. Future migrations should preserve the previous reader or provide a documented migration path rather than silently reinterpreting old data.

## Multimodal provenance

The optional multimodal adapter pins intended external model snapshots:

- `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`
- `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b`

Those revisions are provenance/configuration, not evidence that the models have been downloaded or executed. Real inference requires the optional ML stack and a separately recorded integration result.

## Local performance observation

`a_lmi.benchmarking.benchmark_core()` runs bounded local measurements of CST replay plus bundle export/verify/import. It reports elapsed time, operation counts, environment metadata, errors, and round-trip integrity. It defines no minimum threshold and makes no cross-machine or production-performance claim.

## Optional infrastructure

Kafka, MinIO, Milvus, and Neo4j remain optional integration surfaces. The Compose stack is a local research/development configuration, not a production deployment template. Configuration tests do not prove live service startup/persistence.

See `QUICK_START.md` for setup, `TESTING.md` for executable gates, and `docs/FINAL_CLOSURE_EVIDENCE.md` for the verified-versus-external boundary.
