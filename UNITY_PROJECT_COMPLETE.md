# Unity Project Status

> Historical filename retained for compatibility. This file no longer represents a certification that the Unity project is production-ready or fully validated.

## What exists

The repository preserves a Unity COSMIC SYNAPSE source tree under `cosmic_synapse/Unity/` with project settings and C# scripts for simulation, audio analysis, UI, and Python/Unity IPC.

Representative source includes:

- `CosmosManager.cs`
- `AudioManager.cs`
- `FFTAnalyzer.cs`
- `ForceCalculator.cs`
- `MassInfluence.cs`
- `IPCBridgeClient.cs`
- `UIManager.cs`

The restoration repaired the IPC client source so asynchronous receive handling uses a valid Task-based pattern and follows the shared versioned JSON envelope used by the Python bridge.

## What CI verifies

The current restoration workflow performs source-level Unity IPC contract checks. Those tests verify the expected protocol/source structure without requiring a Unity installation on the CI runner.

This is useful evidence that the repaired IPC source matches the restoration contract. It is **not** equivalent to:

- opening/importing the project in the Unity editor;
- compiling every C# script against a specific Unity release;
- building a player for Windows/macOS/Linux/mobile;
- running live Python↔Unity WebSocket traffic in a built player;
- validating microphone behavior on target hardware;
- measuring frame rate, numerical stability, or rendering correctness.

## Simulation terminology

The Unity source preserves project mechanisms involving golden-angle initialization, FFT/audio-derived values, stochastic/noise terms, particle dynamics, and CST/COSMIC SYNAPSE terminology.

These are software/simulation mechanisms. Their implementation does not by itself validate a new physical law, prove golden-ratio superiority, or turn simulation output into a measurement of external physical reality.

## Reproducing a Unity integration result

For a current Unity integration claim, record at minimum:

1. exact repository commit SHA;
2. Unity editor version;
3. target platform and scripting backend;
4. project import/compile output;
5. scene/setup steps;
6. Python IPC server version/configuration;
7. runtime logs showing the shared versioned message envelope;
8. any microphone/device permissions and hardware used;
9. build/player result and known warnings/errors.

A Unity editor/player build is an explicit integration gate and is not silently counted as a pass by the deterministic Python workflow.

## Current classification

**Preserved Unity prototype/source integration.**

The source is suitable for continued editor integration and controlled testing. It should not be described as “production ready,” “fully validated,” or a scientific validation result until those stronger claims have corresponding reproducible evidence.

See:

- `docs/ARCHITECTURE.md`
- `docs/CLAIMS_AND_LIMITATIONS.md`
- `docs/REPRODUCIBILITY.md`
- `cosmic_synapse/tests/test_unity_ipc_source_contract.py`
