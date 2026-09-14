# Security Notes

This document describes the current engineering security posture of the restoration branch. It is not a third-party security audit or a claim that the complete historical repository is production-hardened.

## Default posture

The dependency-light core is designed to be inspectable without automatically connecting to external services or importing heavyweight optional stacks.

Local Docker Compose services publish development ports on `127.0.0.1` rather than all interfaces. Active MinIO and Neo4j credentials are supplied through environment variables. Historical credentials remain visible in Git history/preserved files where they originally appeared and must not be reused.

## Local environment variables

See `.env.example` for local-development variable names. Do not commit real credentials.

The active configuration recognizes variables including:

- `A_LMI_MINIO_ACCESS_KEY`
- `A_LMI_MINIO_SECRET_KEY`
- `A_LMI_NEO4J_USERNAME`
- `A_LMI_NEO4J_PASSWORD`
- `MILVUS_MINIO_ACCESS_KEY`
- `MILVUS_MINIO_SECRET_KEY`

Use unique secrets for any environment beyond an isolated disposable local setup.

## Encryption

The active dependency-light encryption path uses AES-GCM authenticated encryption. Password-based encryption persists the random salt/derivation metadata required for decryption rather than deriving with an unrelated new salt.

Authenticated encryption protects confidentiality/integrity of correctly handled ciphertexts, but it does not by itself provide identity/authentication of users, authorization policy, key rotation, secure backup, forward secrecy, or endpoint security.

## HRCS cryptography

The restored HRCS path uses authenticated symmetric cryptography. Static/pre-shared key designs do **not** provide cryptographic forward secrecy simply because authenticated encryption is used.

Do not describe the current HRCS key exchange/storage design as forward-secret unless a separate ephemeral authenticated key-agreement protocol is implemented, tested, and documented.

## Federated / privacy terminology

The active federated helper distinguishes:

- weighted averaging;
- optional experimental Gaussian-noise injection;
- explicit metadata indicating that formal differential privacy and cryptographic secure aggregation are not provided by those helpers.

A privacy guarantee requires a complete mechanism/accountant/threat-model analysis, not merely adding Gaussian noise to a tensor.

## Optional security research modules

Historical/optional homomorphic-encryption and secure-computation modules remain preserved. Their presence does not establish production deployment safety, protocol correctness, or resistance to malicious participants.

Treat them as research/optional components until separately reviewed and integration-tested.

## Object and graph storage

Raw artifact metadata includes content hashes and storage references so callers can retain provenance. A hash detects changes when compared to a trusted expected digest; it is not an access-control mechanism.

MinIO, Milvus, Neo4j, and Kafka require their own deployment hardening if used outside localhost development, including as applicable:

- network segmentation;
- TLS;
- authentication/authorization;
- secret management;
- backups and restore testing;
- resource limits;
- monitoring/logging;
- image/version pinning and vulnerability management.

## Docker Compose

`infrastructure/docker-compose.yml` is a local research/development convenience stack, not a production orchestrator.

Current protections include loopback host bindings, env-driven active credentials, corrected Kafka internal hostname/advertising, and separated data volumes. Before any remote/shared deployment, replace local defaults and add production controls.

## Microphone and browser permissions

Audio components require explicit operating-system/browser permission when live capture is used. The dependency-light package can be installed without PyAudio or browser microphone access.

God Music intends a separate microphone analysis path and synthesized output path. Browser/audio-graph behavior should still be integration-tested on target browsers/devices.

## SDR / radio

Radio code is experimental. Do not transmit outside a lawful, configured test environment. Frequency, bandwidth, gain, antenna, duty cycle, licensing, and local regulations are outside the guarantees of this repository.

No current restoration result demonstrates anti-jamming security or synchronized hopping reliability.

## Unity / IPC

The shared IPC schema validates message version/type/payload shape, but a schema is not an authorization system. A remotely reachable WebSocket bridge would require authentication, authorization, origin/network policy, rate limits, and transport security appropriate to the deployment.

## Historical secrets

Because history is intentionally preserved, removing a credential string from the current branch does not erase it from Git history. Any credential that was ever committed should be treated as exposed and rotated/revoked before reuse.

## Reporting issues

When reporting a security issue, include the affected path/component, exact version/commit, reproduction steps, impact, and whether the issue affects only historical material or the active restoration path. Avoid posting live secrets in issues or pull requests.
