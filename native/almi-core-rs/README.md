# A-LMI Native Core / Rust

A-LMI Native Core is the additive Rust implementation of the active A-LMI continuity contracts. The existing Python implementation remains the reference/oracle until native parity is demonstrated by cross-language tests.

## Invariants

- `MODEL != SYSTEM`
- `MODEL != MEMORY`
- `MODEL != AUTHORITY`
- `UI != SYSTEM`
- `LANGUAGE != ARCHITECTURE`

The model/provider is replaceable. Continuity, memory, CST computational state, routing, provenance, policy, and authority remain owned by the surrounding runtime.

## Workspace

The native core preserves the active portable workspace layout:

- `system.json`
- `memory/ledger.jsonl`
- `state/cst.json`
- `knowledge/graph.json`
- `artifacts/manifest.json`
- `provenance/provider.json`
- `routing/state.json`
- `policy/authority.json`

New native workspaces default all authority domains to empty: tools, network, filesystem, cloud, deployment, and actuators.

## Build and test

```text
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features
```

The native executable is `almi`.

```text
almi doctor
almi init story --name "My Story" --seed 1
almi inspect story
almi export story story.cosmos
almi verify story.cosmos
almi import story.cosmos restored-story
almi memory verify story
```

Machine-readable output is available through the global `--json` flag.

## Windows

From the repository root:

```text
INSTALL_WINDOWS.bat
RUN_WINDOWS.bat doctor
VERIFY_WINDOWS.bat
```

The installer is user-local and defaults to `%LOCALAPPDATA%\A-LMI`. It does not require admin rights. The uninstaller does not search for or delete user workspaces, `.cosmos` bundles, memory ledgers, or backups.

## Python

The `almi-python` crate builds the `almi_native` extension through PyO3/maturin. It is additive; it does not remove or bypass the Python reference implementation.

## C ABI

`include/almi.h` exposes ABI v1 with opaque context ownership, stable error codes, explicit string freeing, and Rust panic containment.

## Claim boundary

This subsystem demonstrates software interoperability and continuity contracts only. It does not claim consciousness, sentience, biological identity, a soul, resurrection, new physics, physical extra dimensions, quantum advantage, AGI, or production security certification.
