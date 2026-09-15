# Product Closure Status

> Historical filename retained for compatibility. `SYSTEM_READY.md` is not a production-readiness certificate.

## Current classification

**Research/alpha persistent AI runtime with a reproducible deterministic core and optional environment-dependent integrations.**

The strongest current product center is a user-owned continuity workspace whose memory, computational state, provenance, routing, artifact/knowledge surfaces, and authority are separate from the replaceable model provider.

Core invariants:

- MODEL != SYSTEM
- MODEL != MEMORY
- MODEL != AUTHORITY

## Verified in deterministic CI

The active workflow verifies on Python 3.11 and 3.12:

- configuration/startup behavior;
- LightToken dimensions/serialization;
- authenticated password-encryption envelopes;
- raw-artifact SHA-256 provenance;
- explicit multimodal embedding spaces and pinned intended Hub revisions;
- vector/memory/graph helper contracts;
- optional dependency boundaries;
- deterministic CST state/replay;
- hypothesis provenance;
- federated claim boundaries;
- portable user workspace initialization;
- deterministic `.cosmos` export/verify/import and corruption/security handling;
- provider identity/runtime swapping without authority transfer;
- end-user CLI flow;
- bounded benchmark/soak integrity;
- active-surface security regressions;
- Docker Compose credential/port/hostname contracts;
- HRCS packet/crypto/acoustic/mesh/simulation/radio-planning software behavior;
- Python IPC and Unity IPC source/schema behavior.

Independent CI jobs also verify:

- wheel/sdist construction;
- clean-wheel installation and doctor/import smoke;
- installed-wheel portable continuity round trip;
- God Music deterministic Node tests and Vite production build;
- a separate active-surface security-static gate.

## Usable product flow

The installed CLI exposes:

```text
INSTALL
-> init user workspace
-> inspect continuity/authority
-> interact through an explicit provider
-> preserve memory outside provider parameters
-> switch provider/model without granting authority
-> export deterministic .cosmos bundle
-> verify integrity
-> import to an empty destination
-> continue from the restored user-owned history
```

The first concrete provider client is Ollama. Its software contract is tested; a real Ollama model execution remains an external integration result, not a deterministic CI claim.

## Security posture

The portable import path validates integrity before materialization and rejects traversal, symlinks, duplicate/undeclared members, secret-bearing filenames, version mismatches, oversized bundles, and non-empty destinations. New workspaces start with no tool/network/filesystem authority.

Provider endpoints reject embedded credentials before the endpoint can enter provenance/error reporting. Active Compose ports remain loopback-bound and secret-bearing values come from local environment variables rather than committed fallback passwords.

These are engineering protections, not a complete security audit or production certification.

## Implemented but environment-dependent

The repository contains integration paths for Kafka, MinIO, Milvus, Neo4j, CLIP, WavLM, Vosk, audio capture, browsers, WebSockets, Unity, visualization, and SDR hardware.

The active CLIP and WavLM code pins intended Hub revisions, but those weights were not downloaded/executed by the dependency-light closure CI. Live service/model/device/hardware claims require recorded evidence from the target environment.

## Not established

The repository does not establish consciousness, sentience, AGI, biological life, soul/identity persistence, new physics, extra physical dimensions, quantum consciousness/advantage, golden-ratio superiority, formal differential privacy, cryptographic secure aggregation, RF anti-jamming superiority, unmeasured hardware performance, or production security.

## Deployment status

`infrastructure/docker-compose.yml` is a localhost research/development stack. Production/shared-network deployment would require target-specific authentication/authorization, TLS/network policy, secret management, backup/restore testing, monitoring, rate limiting, vulnerability management, resource policy, incident response, and real service integration evidence.

## Source of truth

For current readiness use, in order:

1. CI status on the exact current `main` SHA;
2. `README.md` and `QUICK_START.md`;
3. `TESTING.md` and `docs/FINAL_CLOSURE_EVIDENCE.md`;
4. the active architecture/security/reproducibility/claims docs;
5. historical artifacts for lineage/context only.

A blocked external or hardware gate is not a failed software restoration, and a green deterministic software gate is not permission to claim that external gate passed.
