# COSMIC SYNAPSE / A-LMI Final Product Closure Design

## Status

Approved implementation contract from Cory Davis / NavisWORLD on 2026-09-15.

## Product center

COSMIC SYNAPSE / A-LMI is a research-oriented persistent AI runtime that keeps user-owned memory, computational state, provenance, routing, policy/authority configuration, and artifact references separate from replaceable model providers.

Core invariants:

- MODEL != SYSTEM
- MODEL != MEMORY
- MODEL != AUTHORITY
- user-owned continuity is portable and independently integrity-verifiable
- provider changes never grant tool authority
- theory, simulation, software verification, external-service integration, and hardware evidence remain separately classified

## Closure architecture

The dependency-light core gains a portable continuity subsystem and an explicit model-provider contract. A continuity workspace is a normal directory with canonical JSON/JSONL state. Export produces a deterministic ZIP bundle with a versioned manifest, SHA-256 hashes, file sizes, and an explicit exclusion policy for secret-bearing filenames. Import verifies every payload before writing, rejects path traversal/symlinks/undeclared files, and writes into a new destination unless explicitly empty.

The provider boundary is transport-neutral at the runtime interface. The initial concrete provider is Ollama over its local HTTP API using only the Python standard library. Provider metadata records provider ID, model ID, optional revision, capabilities, endpoint, timeout, health, and response provenance. Tool/authority permissions are stored separately in the continuity workspace and are never inferred from model output or provider selection.

The CLI becomes the product front door:

- `cosmic-synapse init PATH --name NAME`
- `cosmic-synapse inspect PATH [--json]`
- `cosmic-synapse export PATH BUNDLE`
- `cosmic-synapse import BUNDLE PATH`
- `cosmic-synapse verify BUNDLE`
- `cosmic-synapse providers [--json]`
- `cosmic-synapse run PATH --provider ollama --model MODEL --prompt TEXT`
- existing `doctor` and `cst-demo` remain supported

`run` appends user/assistant records to the local memory ledger and persists provider provenance. It does not grant shell/filesystem/network/cloud/tool authority beyond the explicitly selected provider transport itself.

## Security boundary

Bundle import/export must fail closed on path traversal, absolute paths, symlinks, duplicate names, undeclared payloads, oversized payloads, hash mismatches, unsupported format versions, non-empty destinations, and secret-bearing filenames. Default policy contains no tool grants. Provider endpoints default to loopback and custom endpoints require explicit CLI/config selection.

## Evidence boundary

Deterministic CI verifies portable continuity, provider contracts without requiring a live provider, CLI behavior, packaging, security regression cases, existing CST/IPC/HRCS contracts, and God Music. A real Ollama integration is reported only if an Ollama service is actually contacted. Kafka/MinIO/Milvus/Neo4j, model-weight downloads, microphone/browser devices, SDR hardware, Unity editor/player, GPU/CUDA, and production deployment remain external gates unless an execution environment actually supplies them.

## Release condition

Merge only after the exact PR head is green. After merge, require a fresh `main` push CI success on the exact resulting main SHA. Do not delete historical artifacts or overwrite historical tags/releases.