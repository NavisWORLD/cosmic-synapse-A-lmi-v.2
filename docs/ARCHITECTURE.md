# Active Architecture

This document describes the active software architecture on `restoration/complete-system-2026-09-14`. Historical papers, demos, ZIP archives, earlier engines, and terminology are preserved separately and are not silently promoted into active-runtime claims.

## System boundary

The repository is a research-software ecosystem with several related but separable surfaces:

1. **A-LMI Python package** — data structures, multimodal processing adapters, memory clients, reasoning utilities, security helpers, and CLI diagnostics.
2. **COSMIC SYNAPSE Python state/simulation layer** — deterministic CST software state plus simulation and IPC helpers.
3. **HRCS communications package** — packet, crypto, acoustic, mesh, simulation, and experimental SDR radio code.
4. **God Music web application** — browser audio analysis, deterministic timing/harmonic rules, prediction helpers, synthesis, and visualization.
5. **Unity client source** — optional C# visualization/IPC client surface.
6. **Optional infrastructure** — Kafka, MinIO, Milvus, Neo4j, and supporting containers.

These pieces can be inspected and tested independently. The minimal Python installation does not require the full infrastructure, microphone stack, ML stack, visualization stack, or WebSocket stack.

## Active data flow

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
          |-- spectral representation: 769-bin real FFT of that embedding
          |-- provenance / modality / storage metadata
          |
          +--> vector memory (optional Milvus)
          +--> temporal graph (optional Neo4j)
          +--> reasoning / hypothesis utilities
```

The 769-bin spectral representation is `numpy.fft.rfft` over the 1536-value embedding. It is a one-dimensional Fourier representation of software data. It is **not** a Graph Fourier Transform and is not evidence of a new physical frequency domain.

## Canonical CST software state

`cosmic_synapse.cst_state` provides the restoration branch's stable deterministic interface for the historical CST state terminology. The adapter exposes explicit software state such as `x12`, `m12`, phase, seed, event progression, snapshot serialization, and deterministic replay.

The historical `12D` naming is preserved as project lineage. In the active documentation it means a twelve-channel computational state representation; it is not presented as a claim that spacetime has twelve experimentally established physical dimensions.

## Memory and provenance

The active memory path separates three concerns:

- **Object storage:** raw bytes, object URI, SHA-256 digest, byte size, and content metadata.
- **Vector memory:** explicit embedding space, dimensions, semantic vector, and spectral vector.
- **Temporal knowledge graph:** entities/relationships plus queryable graph records for visualization and reasoning.

Clients for MinIO, Milvus, and Neo4j are optional dependencies and are imported lazily so the core package remains installable without external services.

## Multimodal boundary

The code treats embedding spaces honestly:

- CLIP-style text/image representations may share their model's semantic space.
- WavLM-style audio representations are identified as a different embedding space unless an explicit trained alignment is supplied.
- Production code does not use random vectors as successful multimodal outputs.
- An untrained random projection is not described as semantic alignment.

## Hypothesis generation

Hypothesis generation records the observation/pattern, evidence inputs, uncertainty/confidence, provider provenance, and provider results. It does not manufacture placeholder search URLs as evidence.

A generated hypothesis is a candidate for testing, not a verified discovery.

## Security boundary

Verified dependency-light security behavior includes AES-GCM helpers and password-based encryption with stored derivation parameters/salt sufficient for decryption.

The repository also preserves experimental or optional homomorphic-encryption, secure-computation, and federated-learning code. The active federated API distinguishes ordinary weighted averaging and experimental Gaussian-noise injection from formal differential privacy or cryptographic secure aggregation.

See `SECURITY.md` and `CLAIMS_AND_LIMITATIONS.md`.

## HRCS boundary

The HRCS package contains:

- versioned packet serialization and integrity checking;
- authenticated symmetric encryption;
- deterministic acoustic modem round-trip tests;
- replay/duplicate handling and simulated multi-hop communication;
- deterministic radio hop planning and transmit-side SDR retuning when compatible hardware exists.

The current restoration does **not** claim cryptographic forward secrecy from a static/pre-shared key, proven anti-jamming superiority, or verified synchronized receive-side SDR hopping.

## IPC / Unity boundary

Python and Unity share a versioned JSON message envelope. Python serialization/processing is dependency-light; the WebSocket transport is optional. The Unity source uses Task-based asynchronous receive logic rather than mixing `await` into a non-async iterator.

CI performs source-contract tests for the Unity IPC code. A full Unity editor/player build is a separate integration gate and is not implied by those tests.

## God Music boundary

God Music is an algorithmic browser music experiment. CI runs deterministic Node tests for core analysis/prediction utilities and builds the Vite application. Live browser microphone behavior remains a browser/hardware integration surface.

## Optional infrastructure

`infrastructure/docker-compose.yml` publishes development service ports on `127.0.0.1` only and takes active credentials from environment variables. The Compose stack is for local research/development and should not be treated as a production deployment template without additional network, secret-management, TLS, backup, monitoring, and hardening work.

## Packaging

The root `pyproject.toml` provides a dependency-light default package plus named extras:

- `audio`
- `ml`
- `infra`
- `viz`
- `ipc`
- `dev`

The `cosmic-synapse doctor --json` command reports core import health and optional capability availability without requiring the heavyweight stacks to be installed.

## Verification layers

The restoration CI intentionally separates:

- deterministic Python contracts on Python 3.11 and 3.12;
- wheel/sdist build plus fresh-environment installation;
- God Music Node tests and Vite production build.

External databases, downloaded model weights, microphones, SDR hardware, and the Unity editor remain explicit integration/hardware gates rather than silent passes.
