# Contributing

Contributions are welcome when they preserve the repository's historical lineage while improving the active software/evidence boundary.

## Before changing code

Read:

- `README.md`
- `docs/PRODUCT_WORKFLOW.md`
- `docs/ARCHITECTURE.md`
- `docs/CLAIMS_AND_LIMITATIONS.md`
- `docs/HISTORY_AND_MIGRATION.md`
- `docs/REPRODUCIBILITY.md`

Do not delete historical papers, demos, ZIP archives, theory artifacts, or earlier generations merely to make the repository look cleaner.

## Development setup

```bash
python -m pip install -e '.[dev]'
```

Optional stacks are separate extras (`audio`, `ml`, `infra`, `viz`, `ipc`). Install only what your change requires.

## Test-driven repairs

For an executable bug or behavior change:

1. reproduce the current behavior;
2. add the smallest deterministic failing contract when practical;
3. confirm the contract fails for the intended reason;
4. implement the narrow fix;
5. run the relevant expanded suite;
6. preserve limitations and blocked/null results.

Do not weaken a meaningful assertion merely to turn CI green. If a test encoded the wrong contract, document why and correct the test rather than modifying production behavior to match a mistake.

## Current CI expectations

Use `TESTING.md` and `docs/REPRODUCIBILITY.md` for current commands. Pull requests affecting the active Python product should keep these gates green:

- deterministic core on Python 3.11;
- deterministic core on Python 3.12;
- active-surface security-static regression;
- wheel/sdist build + clean install + installed portable-product smoke;
- God Music tests/build when applicable.

Changes to continuity/provider/runtime/CLI behavior should include focused tests and preserve the installed-wheel `init -> export -> verify -> import` path.

## Persistent-runtime invariants

Do not couple user-owned continuity or authority to a specific model provider.

Preserve:

- MODEL != SYSTEM
- MODEL != MEMORY
- MODEL != AUTHORITY

A provider change must not implicitly grant shell, filesystem, network, cloud, deployment, actuator, or tool authority. If a future tool layer is introduced, authorization must remain an explicit surrounding-system decision.

## Optional integrations

For Kafka, MinIO, Milvus, Neo4j, downloaded ML models, microphones, SDR, browsers, GPU/CUDA, or Unity, include environment-specific evidence rather than treating an unavailable integration as a pass.

A useful integration report includes exact commit SHA, OS/runtime versions, service/model/hardware versions, non-secret configuration, commands, raw outputs, failures, and limitations.

## Claim discipline

Use these labels accurately:

- verified software result
- integration result
- device/hardware result
- simulation result
- hypothesis
- historical claim
- blocked/null result

Avoid upgrading words such as “works,” “secure,” “validated,” “optimal,” “production ready,” “anti-jamming,” “differentially private,” “conscious,” or “AGI” beyond the actual accompanying evidence.

## Security

Never commit live credentials, API keys, private keys, tokens, or personal secrets. Keep `.env` untracked and `.env.example` limited to placeholders.

Provider endpoints that would persist embedded URL credentials are intentionally rejected. Portable bundle changes must continue to fail closed on path/integrity/security violations.

The current security-static CI job is a regression gate, not a full security audit. Security-sensitive changes should add narrowly targeted tests and document any external review/scanner used.

Any credential previously committed anywhere in history should be treated as exposed and rotated before reuse.

## Historical terminology

CST, 12D, vibrational information, phi/golden-ratio, resonance, bio/psi, and related names may matter to lineage. Preserve names where compatibility/history requires them while describing current software mechanisms and evidence precisely.

## Documentation changes

Active user/developer docs may be corrected when they contain stale commands, credentials, status, security guarantees, or scientific claims. Historical papers/publications should generally remain preserved as authored and be contextualized from active docs instead of silently rewritten.

## Pull requests

A useful pull request states the goal, preservation impact, affected subsystems, tests/CI, deterministic vs external evidence, security/compatibility considerations, and known limitations. Keep unrelated cleanup out of a focused repair unless it is required by the same root cause.

Before merge, verify the exact PR head. After a release-significant merge to `main`, verify the exact resulting `main` SHA rather than relying on an earlier branch run.
