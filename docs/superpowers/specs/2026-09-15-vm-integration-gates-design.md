# VM Integration Gate Closure Design

## Goal

Turn the remaining software/environment gates into reproducible live integration evidence on disposable Linux VMs while preserving a strict boundary between VM-verifiable behavior and claims that require physical hardware, licensed environments, or independent production certification.

## Architecture

Use GitHub Actions hosted Linux runners as ephemeral VMs. Do not run the entire stack in one job. Each integration family gets an isolated job so failures have a single owner and resource-heavy services do not contend unnecessarily.

The VM suite will add these independent gates:

1. `kafka-live` — start a real Kafka broker, produce/consume a uniquely named record, restart the broker, then consume/produce again and verify topic data survives the restart.
2. `minio-live` — start real MinIO with a named Docker volume, use the repository `ObjectStorageClient` to write/read bytes, restart MinIO, and verify the same object and SHA-256 remain available.
3. `neo4j-live` — start real Neo4j with a named Docker volume, use `TKGClient` for entity/relationship writes and graph reads, restart Neo4j, then verify the graph persists.
4. `milvus-live` — start Milvus standalone with etcd and MinIO, insert a real `LightToken` through `VectorDBClient`, perform semantic retrieval, restart the Milvus service, reconnect, and verify retrieval still works.
5. `ollama-live` — install/start Ollama, pull a deliberately small public model suitable for CI, perform a real inference through `OllamaProvider`, and verify provider identity plus non-empty returned text. This validates the provider path, not answer correctness.
6. `multimodal-live-cpu` — install CPU Torch/Transformers and execute real CLIP text/image inference and real WavLM audio inference at the exact revisions pinned by the active code. Record the resolved revisions and output dimensions. A small real Vosk model may be used to execute the Vosk engine path, but that does not claim the historical/configured 0.22 model itself was exercised unless that exact model is downloaded.
7. `browser-fake-media` — run the God Music browser app in Chromium with fake media devices/permissions and verify microphone/media lifecycle behavior that is observable without physical hardware. The result is classified as simulated device-path evidence, not real microphone hardware evidence.
8. `security-vm` — run dependency/security checks against the built software and web package. Passing this gate is hardening evidence, not production security certification.

## Alternatives considered

### One giant Compose VM

Rejected as the default. It is simple to describe but creates noisy failures, higher memory pressure, and weak provenance because one broken service can mask all other gates.

### Self-hosted all-capabilities runner

Useful later for GPU, Unity, and SDR, but rejected as the primary closure route because it is not reproducible for ordinary contributors and requires owner-provided hardware/licensing.

### Split hosted VM jobs — selected

Best balance of reproducibility, failure isolation, cost, and evidence quality. Physical-only gates remain separate instead of being simulated into false passes.

## Evidence classifications

- `VERIFIED_VM_INTEGRATION`: real process/service/model executed on a hosted VM with a recorded passing workflow job.
- `VERIFIED_SIMULATED_DEVICE_PATH`: browser/device API exercised with fake media devices; no physical microphone claim.
- `REQUIRES_GPU_HARDWARE`: no GPU exists on standard hosted runners.
- `REQUIRES_UNITY_LICENSED_ENVIRONMENT`: real Unity editor/player execution requires a compatible Unity environment/license workflow.
- `REQUIRES_SDR_HARDWARE`: RF transmit/receive/range cannot be established in a VM.
- `NOT_PRODUCTION_CERTIFIED`: automated scans and local integration cannot certify a production deployment.

## Success criteria

A gate moves out of `BLOCKED_EXTERNAL_ENVIRONMENT` only when its exact workflow job runs to completion successfully on the exact commit being evaluated. A service that merely starts is not sufficient; the repository client must perform a meaningful write/read or inference operation. Persistence gates must include a restart before the final verification.

The final evidence document must name the workflow run and exact commit SHA and must leave physical/hardware/security-certification claims explicit rather than upgrading them from simulation.