# Final Product Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the dependency-light product path around portable continuity, explicit model-provider swapping, CLI/runtime use, security regression coverage, documentation, and exact-SHA CI verification.

**Architecture:** Add two focused core units: `a_lmi/continuity.py` owns portable user-workspace lifecycle and deterministic bundles; `a_lmi/providers.py` owns model-provider identity/requests/responses and the first concrete Ollama transport. `a_lmi/runtime.py` composes them without giving models tool authority. The existing CLI becomes the product front door and CI remains the evidence gate.

**Tech Stack:** Python 3.11+, stdlib (`argparse`, `hashlib`, `json`, `pathlib`, `urllib`, `zipfile`), NumPy/CST already in core, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-final-product-closure-design.md`

## Global Constraints

- Preserve historical artifacts and Git history.
- MODEL != SYSTEM; MODEL != MEMORY; MODEL != AUTHORITY.
- No secret credentials in exported bundles by default.
- No external-service/hardware success claims without actual execution evidence.
- Heavy ML/audio/infra dependencies remain optional.
- Final completion claim requires fresh CI on the exact final `main` SHA.

---

### Task 1: Portable continuity contract

**Files:**
- Create: `tests/test_continuity_bundle.py`
- Create: `a_lmi/continuity.py`

**Interfaces:**
- Produces: `initialize_workspace(path, name, seed=0) -> dict`
- Produces: `inspect_workspace(path) -> dict`
- Produces: `export_bundle(path, bundle_path) -> dict`
- Produces: `verify_bundle(bundle_path) -> dict`
- Produces: `import_bundle(bundle_path, destination) -> dict`
- Produces: `append_memory_record(path, record) -> dict`

- [ ] **Step 1:** Add tests covering deterministic initialization, manifest contents, export/verify/import round trip, corruption detection, secret exclusion, path traversal/symlink rejection, duplicate/undeclared archive member rejection, and non-empty destination refusal.
- [ ] **Step 2:** Put the new test file into Restoration CI and open a PR so the missing module produces an intentional RED run.
- [ ] **Step 3:** Implement the minimum dependency-light continuity module with canonical JSON, SHA-256 manifests, bounded archive sizes, and safe extraction.
- [ ] **Step 4:** Re-run PR CI and require the continuity tests to pass on both Python 3.11 and 3.12.

### Task 2: Explicit provider contract and persistent runtime

**Files:**
- Create: `tests/test_provider_runtime.py`
- Create: `a_lmi/providers.py`
- Create: `a_lmi/runtime.py`

**Interfaces:**
- Produces: `ProviderIdentity`, `ModelRequest`, `ModelResponse`, `ProviderError`
- Produces: `OllamaProvider.health()`, `OllamaProvider.generate(request)`
- Produces: `PersistentRuntime.interact(prompt) -> ModelResponse`

- [ ] **Step 1:** Add provider/runtime tests first: identity/provenance, loopback default endpoint, deterministic test-provider injection, memory append across provider changes, provider swap not altering policy authority, and provider failures not fabricating responses.
- [ ] **Step 2:** Confirm RED CI due to missing provider/runtime modules.
- [ ] **Step 3:** Implement stdlib Ollama provider plus provider-neutral runtime with injected providers for tests.
- [ ] **Step 4:** Require all provider/runtime and existing deterministic tests green.

### Task 3: CLI product front door

**Files:**
- Create: `tests/test_cli_product_flow.py`
- Modify: `a_lmi/cli.py`

**Interfaces:**
- Adds commands: `init`, `inspect`, `export`, `import`, `verify`, `providers`, `run`

- [ ] **Step 1:** Add CLI tests for init/inspect/export/import/verify and parser contract for providers/run.
- [ ] **Step 2:** Confirm RED CI because commands are absent.
- [ ] **Step 3:** Implement command handlers using continuity/provider/runtime interfaces; preserve `doctor` and `cst-demo`.
- [ ] **Step 4:** Require product CLI tests and package clean-install smoke green.

### Task 4: Security, reproducibility, and evidence records

**Files:**
- Create: `docs/PRODUCT_WORKFLOW.md`
- Create: `docs/FINAL_CLOSURE_EVIDENCE.md`
- Modify: `README.md`
- Modify: `QUICK_START.md`
- Modify: `TESTING.md`
- Modify: `SYSTEM_READY.md`
- Modify: `docs/README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/SECURITY.md`
- Modify: `docs/REPRODUCIBILITY.md`
- Modify: `docs/CLAIMS_AND_LIMITATIONS.md`
- Modify: `docs/HISTORY_AND_MIGRATION.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1:** Document the product-centered workflow and exact portable-bundle security/integrity boundary.
- [ ] **Step 2:** Record external gates as `VERIFIED_SOFTWARE`, `BLOCKED_EXTERNAL_ENVIRONMENT`, or `REQUIRES_HARDWARE`; never silently convert a blocked gate to success.
- [ ] **Step 3:** Update the README first screen to center the persistent runtime rather than historical claims.
- [ ] **Step 4:** Cross-check docs against implemented commands and current CI.

### Task 5: CI and release verification

**Files:**
- Modify: `.github/workflows/restoration-ci.yml`
- Modify: `pyproject.toml` only if packaging metadata needs final alignment.

- [ ] **Step 1:** Ensure all new deterministic tests are explicit CI gates on Python 3.11 and 3.12.
- [ ] **Step 2:** Extend package-build smoke to exercise CLI help and a complete temporary init/export/verify/import flow from the built wheel.
- [ ] **Step 3:** Verify PR head exact SHA: deterministic core 3.11/3.12, package build/clean install/product smoke, God Music.
- [ ] **Step 4:** Review PR diff for accidental historical deletions or secret material.
- [ ] **Step 5:** Merge only the exact verified head.
- [ ] **Step 6:** Require a fresh push-triggered CI run on the exact final `main` SHA before declaring software closure complete.
