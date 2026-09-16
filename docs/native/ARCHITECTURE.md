# A-LMI Native Core Architecture

## Purpose

The Rust subsystem is an additional implementation of the active A-LMI product contracts. Python remains the compatibility oracle during this phase. The native work does not redesign the system or move memory/authority into a model.

## Core invariant

```text
.cosmos continuity
       |
  +----+----+
  |         |
Python    Rust
  |         |
  +----+----+
       |
 same portable contracts
       |
memory / CST state / routing / provenance / policy / authority
       |
 replaceable provider/model
```

`MODEL != SYSTEM`, `MODEL != MEMORY`, `MODEL != AUTHORITY`, `UI != SYSTEM`, and `LANGUAGE != ARCHITECTURE` are architectural constraints rather than slogans: provider configuration and provider responses do not grant runtime authority.

## Crates

- `almi-core`: versioned wire/domain contracts and canonical JSON.
- `almi-continuity`: portable workspace creation, inspection, and validation.
- `almi-memory`: canonical JSONL memory ledger operations and validation.
- `almi-state`: active CST computational-state envelope and deterministic replay from persisted state.
- `almi-cosmos`: deterministic `.cosmos` export, verification, inspection, and import.
- `almi-provider`: provider identity/health/request/response boundary and Ollama-compatible adapter.
- `almi-runtime`: minimum persistent provider interaction loop.
- `almi-cli`: automation-friendly native executable.
- `almi-ffi`: deliberately small stable C ABI.
- `almi-python`: PyO3 extension exposing selected native operations to Python.

## Versioning

Workspace, bundle, CST-state, component-schema, and ABI versions are explicit. Unsupported versions fail closed rather than being partially deserialized as if compatible.

## Continuity ownership

The continuity workspace owns system identity, memory, active CST state, knowledge references, artifact metadata, provider provenance, routing state, and policy/authority. Provider replacement may update provider provenance and append interactions, but it does not replace memory, routing, CST state, or authority.

## CST scope

The native state crate implements only the active canonical computational-state contract. Historical `12D` terminology is retained as project terminology. These variables are software state and do not constitute a claim of physical extra dimensions or new physics.
