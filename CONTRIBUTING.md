# Contributing

Contributions are welcome when they preserve the repository's historical lineage while improving the active software/evidence boundary.

## Before changing code

Read:

- `README.md`
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

Do not weaken a meaningful assertion merely to turn CI green. If the test encoded the wrong contract, document why and correct the contract rather than modifying production behavior to match a mistake.

## Deterministic verification

Use `TESTING.md` and `docs/REPRODUCIBILITY.md` for the current commands.

At minimum, changes to the Python core should keep the restoration Python 3.11/3.12 jobs green. Packaging changes should also pass the clean wheel-install job. God Music changes should pass `npm test` and `npm run build`.

## Optional integrations

For Kafka, MinIO, Milvus, Neo4j, downloaded ML models, microphones, SDR, browsers, or Unity, include environment-specific evidence rather than treating an unavailable integration as a pass.

A useful integration report includes:

- exact commit SHA;
- OS/runtime versions;
- service/model/hardware versions;
- relevant configuration with secrets removed;
- commands;
- raw success/failure output;
- known limitations.

## Claim discipline

Use the following labels accurately:

- verified software result
- integration result
- hardware result
- simulation result
- hypothesis
- historical claim
- blocked/null result

Avoid upgrading language such as “works,” “secure,” “validated,” “optimal,” “production ready,” “anti-jamming,” “differentially private,” “conscious,” or “AGI” beyond the evidence that accompanies the change.

## Security

Never commit live credentials, API keys, private keys, tokens, or personal secrets.

Use `.env.example` only as a variable-name/template file and keep `.env` untracked. Any credential previously committed anywhere in history should be treated as exposed and rotated before reuse.

See `docs/SECURITY.md`.

## Historical terminology

Names such as CST, 12D, vibrational information, phi/golden-ratio, resonance, bio/psi, and similar terms may be important to project lineage. Preserve names where compatibility/history requires them, but document the active software mechanism precisely.

## Documentation changes

Active user/developer docs may be corrected when they contain stale commands, credentials, status, security guarantees, or scientific claims. Historical papers/publications should generally remain preserved as authored and be contextualized from active docs instead of silently rewritten.

## Pull requests

A useful pull request should state:

- problem/goal;
- preservation impact;
- files/subsystems affected;
- tests run;
- deterministic vs integration/hardware evidence;
- security/compatibility considerations;
- known limitations.

Keep unrelated cleanup out of a focused repair unless it is required by the same root cause.
