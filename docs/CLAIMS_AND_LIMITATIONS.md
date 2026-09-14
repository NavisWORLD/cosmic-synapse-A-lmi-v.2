# Claims and Limitations

This file is the current claim boundary for the restoration branch. It separates verified software behavior from theory, hypothesis, simulation, integration requirements, and unverified extrapolation.

## Verified software behavior

The restoration CI verifies deterministic software contracts for the following classes of behavior:

- configuration loading and environment expansion;
- password-encryption round trips using stored derivation metadata;
- LightToken embedding/spectral serialization contracts;
- raw-artifact persistence metadata and SHA-256 provenance;
- explicit multimodal embedding-space labeling;
- vector schema dimensions and memory metadata;
- temporal-graph loading and visualization helpers with injected/test backends;
- optional audio dependency boundaries;
- deterministic CST state snapshots and replay;
- hypothesis evidence/provenance fields;
- weighted federated averaging and reproducible experimental noise;
- HRCS packet integrity, authenticated encryption, acoustic round trips, replay handling, mesh forwarding, simulated end-to-end communication, and deterministic radio-hop planning;
- versioned Python IPC envelopes and transport-independent bridge handling;
- Unity IPC source-level async/schema contracts;
- dependency-light packaging and CLI diagnostics;
- local Docker Compose credential/port/hostname contracts;
- God Music deterministic JavaScript utility tests and Vite production build.

The package build job also constructs a wheel/source distribution, installs the built wheel into a fresh virtual environment, and runs the dependency-light doctor command.

## Implemented but integration-dependent

The repository contains code paths for capabilities that require external services, downloaded model weights, browsers, operating-system permissions, or hardware. Their presence in the source tree is not equivalent to a CI-verified live deployment.

Examples include:

- Kafka message infrastructure;
- MinIO object storage;
- Milvus vector storage;
- Neo4j graph storage;
- CLIP/Transformer model loading;
- WavLM/audio model loading;
- Vosk speech recognition;
- live microphone capture;
- Dash/Plotly interactive applications;
- WebSocket networking;
- Unity editor/player execution;
- SDR transmission/reception hardware.

These require separate integration tests in the target environment.

## Theory and hypothesis

Historical terminology such as CST, `12D`, vibrational information, phi/golden-ratio harmonics, resonance, and related names is preserved because it is part of the project lineage.

In the active engineering layer:

- `12D` refers to a twelve-channel computational state representation unless a historical document explicitly uses it in another theoretical sense.
- spectral signatures are numerical transforms of software vectors/signals.
- the current LightToken spectral representation is a one-dimensional real FFT of a semantic embedding, not a Graph Fourier Transform.
- golden-ratio constants are software parameters/rules; their use does not demonstrate physical or performance superiority.
- simulation dynamics are simulation results, not measurements of external physical reality.
- a generated hypothesis is an object to test, not a discovery by itself.

## Security claims

### Supported

- AES-GCM authenticated encryption helpers are implemented.
- Password-based encryption stores the salt/derivation information needed to decrypt the resulting envelope.
- HRCS uses authenticated symmetric encryption in the restored active path.

### Not established by the current code/tests

- cryptographic forward secrecy from static or pre-shared keys;
- audited production key management;
- formal differential privacy guarantees;
- cryptographic secure aggregation for federated learning;
- production-grade SMPC/HE deployment security;
- resistance to sophisticated traffic analysis, RF interception, or active adversaries;
- a complete security audit of every preserved historical artifact.

Compatibility methods may preserve historical API names, but active documentation describes their actual mechanism rather than the stronger historical label.

## Multimodal claims

The code can represent/process multiple modalities, but embedding spaces are not assumed to be universally interchangeable.

- Text and image vectors produced by the same CLIP family may occupy the model's shared space.
- WavLM-style audio vectors are a distinct space unless an explicit trained alignment maps them into another space.
- Random vectors and untrained random projections are not accepted as successful semantic inference/alignment in the restored production path.

No current CI result demonstrates human-level multimodal understanding, AGI, or autonomous scientific competence.

## HRCS claims

The restored deterministic tests demonstrate packet and simulated communications behavior in software.

The SDR path now derives hop plans deterministically and can retune transmit frequency when compatible hardware is available. The current restoration does **not** establish synchronized frequency-hopping reception, anti-jamming superiority, regulatory suitability, range, throughput, or field reliability.

Hardware operation must comply with applicable radio law and device limits.

## God Music claims

God Music contains working browser audio-analysis, synthesis, timing, and deterministic predictive/rule logic. CI verifies JavaScript utility behavior and a production web build.

The active project does not claim:

- to be the world's first system of its kind;
- biological-frequency measurement;
- clinical or biometric inference;
- learned musical intelligence unless a trained model is actually introduced and evaluated;
- superiority caused by phi/golden-ratio rules.

## Consciousness / identity / life

No repository test establishes consciousness, sentience, self-awareness, biological life, a soul, identity persistence, resurrection, or a new species. Software state persistence and deterministic replay are engineering properties and should be described as such.

## Physics / mathematics

No current repository test establishes a new physical law, breaks quantum mechanics, proves extra physical dimensions, or establishes a millennium-prize-level mathematical result.

## How to make stronger claims

A stronger claim should be promoted only when the repository contains enough evidence to reproduce it, typically including:

1. a precise operational definition;
2. a predeclared metric or acceptance threshold;
3. a controlled baseline/control condition;
4. deterministic or statistically appropriate test code;
5. raw results/artifacts with provenance;
6. environment/configuration details;
7. limitations and negative results;
8. independent replication when the claim is extraordinary or hardware/real-world dependent.

Until then, label the item as theory, hypothesis, simulation, prototype, integration target, or unverified experiment as appropriate.
