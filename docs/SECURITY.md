# Security Notes

This document describes the active engineering security posture of COSMIC SYNAPSE / A-LMI. It is not a third-party security audit, penetration test, dependency-vulnerability certification, or claim that the complete historical repository is production-hardened.

## Default posture

The dependency-light core can be installed and inspected without automatically contacting external services, opening audio devices, loading heavyweight model stacks, or granting model output tool authority.

New continuity workspaces start with empty tool, network, and filesystem authority. Provider selection changes inference only; it does not silently transfer those permissions.

## Portable continuity bundle

The `.cosmos` import/export path is designed to fail closed around common archive/data risks:

- every declared payload has SHA-256 and byte-size metadata;
- verification occurs before import materializes files;
- traversal/absolute/backslash path tricks are rejected;
- symlink and directory archive members are rejected;
- duplicate and undeclared archive entries are rejected;
- missing, size-mismatched, or hash-mismatched payloads are rejected;
- unsupported format versions fail explicitly;
- bundle size/file-count limits are enforced;
- secret-bearing filenames such as `.env*` and private-key formats are excluded;
- import refuses a non-empty destination rather than overwriting unrelated user data.

The bundle stores user data/state/provenance. It does not intentionally embed service credentials or model parameters by default.

## Provider endpoints and provenance

The first concrete provider client is Ollama. Its default endpoint is loopback. Provider construction performs no network access.

Custom Ollama endpoints must use HTTP(S), include a hostname, and must not embed URL username/password credentials. Credential-bearing endpoints are rejected before they can be persisted in provider provenance or reflected in request-error context.

A model response does not carry authority. The runtime records explicit provider identity/provenance separately from policy.

## Local environment variables

`.env.example` is a placeholder/template file; the real root `.env` must remain untracked. Active service credential variables include:

- `A_LMI_MINIO_ACCESS_KEY`
- `A_LMI_MINIO_SECRET_KEY`
- `A_LMI_NEO4J_USERNAME`
- `A_LMI_NEO4J_PASSWORD`
- `MILVUS_MINIO_ACCESS_KEY`
- `MILVUS_MINIO_SECRET_KEY`

Secret-bearing values are supplied from the environment rather than committed fallback passwords. Any credential ever committed in historical Git material should be treated as exposed and rotated before reuse.

## Docker Compose

`infrastructure/docker-compose.yml` is a local research/development stack, not a production orchestrator. Published host ports are required by tests to stay bound to `127.0.0.1`.

A remote/shared deployment would need target-specific TLS, network policy, authentication/authorization, secret management, backups/restore tests, resource limits, monitoring, rate limiting, vulnerability management, incident response, and real service integration testing.

## Active-surface static regression gate

CI runs a scoped static test over selected active product Python modules. It checks for direct uses of selected dangerous shortcuts such as `eval`, `exec`, `pickle.load`, unsafe `yaml.load`, `shell=True`, and `os.system`; it also verifies `.env` tracking, placeholder secret examples, config fallback rules, and loopback port bindings.

This is intentionally narrow. It is not proof that arbitrary code execution is impossible, does not recursively certify every historical artifact, and does not replace CodeQL/SAST/DAST, dependency/image vulnerability scanning, or expert security review.

## Encryption

The active dependency-light encryption path uses AES-GCM authenticated encryption. Password-based envelopes preserve salt/derivation metadata required for decryption.

Authenticated encryption does not by itself provide user identity, authorization, key rotation, secure backups, endpoint security, or forward secrecy.

## HRCS cryptography

The active HRCS software path uses authenticated symmetric cryptography. Static/pre-shared keys are not described as cryptographic forward secrecy. Forward secrecy would require an appropriate ephemeral authenticated key-agreement/ratchet design plus tests and documentation.

## Federated/privacy terminology

Weighted averaging and experimental Gaussian-noise injection are not called formal differential privacy or cryptographic secure aggregation. Stronger privacy guarantees require a defined threat model, correct mechanism/accounting, and evidence.

## Object, vector, and graph storage

Object hashes provide integrity/provenance when compared with trusted expected values; they are not access control. MinIO, Milvus, Neo4j, and Kafka need their own production hardening if deployed beyond an isolated localhost research environment.

## Model and multimodal supply chain

The active CLIP and WavLM adapters pin intended Hugging Face revisions. Revision pinning narrows the intended snapshot but does not by itself verify downloaded file hashes, dependency provenance, model behavior, or model security. A real deployment should record the resolved artifacts/cache and review the model/dependency supply chain appropriate to that environment.

## Microphone/browser/device permissions

Live audio and browser capture require explicit target-device permissions. Optional audio imports do not open hardware merely by importing the base package. Live permission/capture/start-stop behavior still requires real-device testing.

## Unity / IPC

The shared IPC schema validates version/type/payload structure but is not an authorization system. Any remotely reachable WebSocket bridge would need authentication, authorization, transport security, network/origin controls, input/rate limits, and deployment-specific threat analysis.

## SDR / radio

Radio support is experimental software. No active result establishes anti-jamming superiority, synchronized RX hopping reliability, RF range, regulatory suitability, or emergency readiness. Real transmission must follow applicable hardware limits and law.

## Historical material

History is intentionally preserved. Removing a secret-like string from the active branch would not erase it from Git history or archived artifacts. Do not reuse historical credentials. Active security claims apply only to the explicitly documented current product surface.

## Reporting

When reporting a security issue, identify the exact commit/path/component, reproducible steps, impact, whether the issue affects the active product or only preserved historical material, and remove live secrets from public reports.
