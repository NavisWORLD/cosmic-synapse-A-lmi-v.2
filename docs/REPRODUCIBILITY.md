# Reproducibility Guide

COSMIC SYNAPSE / A-LMI separates deterministic software verification from external service, model, browser, device, Unity, GPU, and RF validation. A passing CI run has a deliberately narrow meaning: the listed software contracts passed on the exact tested commit.

## Exact-SHA rule

Any closure or release statement must identify the exact commit SHA and the GitHub Actions run that tested it. A successful run on an earlier commit is not evidence that a later documentation/code head is green.

After merging product closure, the final acceptance gate requires a fresh push-triggered CI run on the exact resulting `main` SHA.

## Deterministic CI

The active workflow runs Python 3.11 and 3.12 contracts covering configuration, encryption, LightToken, artifacts, multimodal provenance, vector/memory/graph helpers, audio optionality, CST replay, hypothesis provenance, federated terminology, Compose contracts, continuity bundles, provider/runtime behavior, CLI flow, benchmark integrity, active security regressions, HRCS software behavior, and Python/Unity IPC contracts.

The exact command is maintained in `.github/workflows/restoration-ci.yml` and summarized in `TESTING.md`.

## Security-static job

A separate job runs the active-surface security regression test. It verifies selected direct dynamic-execution patterns, root `.env` tracking, placeholder secrets, config secret fallbacks, and loopback Compose port bindings.

This job is a regression check, not a complete security assessment or dependency vulnerability scan.

## Package / installed-product gate

CI builds both wheel and source distribution, installs the wheel into a clean virtual environment, runs `cosmic-synapse doctor --json`, imports representative classes, and executes the portable CLI round trip:

```text
init -> inspect -> export -> verify -> import -> inspect -> providers
```

This distinguishes a valid installed artifact from code that only works when imported from a checkout.

## Portable continuity determinism

For a frozen workspace, `.cosmos` export uses canonical JSON and fixed ZIP metadata so repeated exports are byte-identical. Every payload is declared with SHA-256 and byte size. Verification detects undeclared/missing members and integrity mismatches before import.

Workspace creation includes timestamps; determinism applies to repeated export of the same frozen workspace, not to separately initialized workspaces at different times.

## Model revision discipline

The intended external model snapshots are pinned in active code:

- CLIP: `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41`
- WavLM: `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b`

The deterministic suite protects these revision identities and embedding-space labels without downloading the weights. A true model integration report must additionally record the runtime/dependency versions, resolved/downloaded artifacts or local hashes where appropriate, input fixtures, outputs, device, and failures.

Vosk remains a documented optional local speech-model path and requires a separately recorded installed model/runtime test.

## Provider reproducibility

Provider identity includes provider ID, model ID, optional revision, capabilities, endpoint, and context information when known. Deterministic test providers exist only for contract testing. A real Ollama result requires an actual local service/model run; the provider contract alone is not model-execution evidence.

Provider swaps are tested against the same persistent workspace so memory continuity and unchanged authority can be reproduced independently of a specific model implementation.

## Benchmark / soak discipline

`a_lmi.benchmarking.benchmark_core()` performs bounded local measurements of CST and continuity operations. It records elapsed values, operation counts, environment metadata, errors, and integrity status.

It intentionally has no universal speed threshold. Timing results are only claims about the environment that produced them and should not be generalized to target hardware or production capacity without a controlled benchmark report.

## God Music

From `god music/`:

```bash
npm install --no-audit --no-fund
npm test
npm run build
```

This verifies deterministic utility behavior and web buildability. Microphone permissions and browser/device behavior remain external evidence.

## Local infrastructure

A real Kafka/MinIO/Milvus/Neo4j integration report should record exact container/service versions, host environment, non-secret configuration, commands, startup/health, write/read/search or event-flow evidence, restart persistence, failure behavior, and shutdown/restart results.

The deterministic Compose contract verifies configuration/security properties only; it does not claim the services were run.

## Unity, audio, GPU, and RF

Record these as distinct evidence categories. Source inspection or simulation cannot substitute for:

- Unity editor compile/player build/runtime IPC;
- microphone/browser/device permission and capture lifecycle;
- GPU/CUDA inference/performance;
- SDR transmit/receive/range/error measurements.

## Seeds and state

CST replay exposes explicit seeds/state snapshots. HRCS deterministic hop planning uses stable SHA-derived data instead of Python process-randomized `hash()`. When randomness is part of an experiment, record the seed and distinguish deterministic replay from statistical evidence.

## Evidence labels

Use these labels consistently:

- **verified software result** — deterministic test/build passed;
- **integration result** — external service/model path actually executed in a specified environment;
- **device/hardware result** — physical/runtime device path actually executed;
- **simulation result** — produced by modeled/simulated environment;
- **hypothesis** — proposed relationship not yet established;
- **historical claim** — preserved statement from earlier material;
- **blocked/null result** — target result was not established.

Do not promote one category into another without new evidence.

## Provenance retention

Keep exact commit SHAs, CI run IDs/links, raw logs/artifacts, negative results, and environment details alongside any future benchmark, scientific claim, model integration, hardware test, or release note. See `PROVENANCE.md` and `FINAL_CLOSURE_EVIDENCE.md`.
