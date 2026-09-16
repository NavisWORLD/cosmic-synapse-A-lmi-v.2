# A-LMI Native Core / Rust Implementation Plan

> Execution plan for `native/almi-core-rust-001`. Python remains the compatibility oracle until cross-language parity is demonstrated.

**Starting main SHA:** `fc233619367cd9d2f5069198d4b0d7bf3683bb24`

**Goal:** Add a portable Rust native foundation that preserves active `.cosmos`, continuity, CST state, provider, provenance, routing, policy, and authority semantics without replacing the Python implementation.

**Architecture:** Narrow crates own versioned domain contracts, continuity workspace, ledger, CST state, `.cosmos`, provider/runtime, CLI, Python bridge, and C ABI. The Rust implementation consumes the existing Python wire formats rather than inventing parallel schemas. Security validation is fail-closed; provider inference never grants authority.

## TDD sequence

1. Domain/version/authority contracts: write failing tests, observe red, implement typed validated structures.
2. CST state: parity tests from synthetic canonical envelopes and deterministic event vectors, then implementation.
3. Memory ledger: canonical JSONL, legacy-Python-v1 compatibility, malformed/unknown-version rejection.
4. Continuity workspace: deterministic layout and deny-by-default authority.
5. `.cosmos`: hostile-archive tests first, then deterministic export/verify/import.
6. Cross-language interop: Python export -> Rust verify/import and Rust export -> Python verify/import using synthetic fixtures.
7. Provider/runtime: endpoint safety, provenance, provider swap continuity, unchanged authority.
8. CLI: machine-readable and interactive-safe command surfaces with nonzero failures.
9. Python/PyO3 and minimal opaque-handle C ABI with panic containment.
10. Windows bootstrap, portable package, CI matrix, security/property gates, benchmark smoke, documentation, PR, merge, and fresh main verification.

## Claim boundary

Only executed tests/runners are evidence. No consciousness, biological identity, extra-dimensional physics, AGI, hardware, or security-certification claims are introduced.
