# Unity 3D COSMIC SYNAPSE Prototype

This folder preserves the Unity visualization/simulation client for the COSMIC SYNAPSE research project. It contains source for particle simulation, audio analysis, UI controls, and versioned IPC with the Python bridge.

The restoration CI verifies the Unity IPC **source contract**; it does not currently run a Unity editor/player build. Treat the steps below as an integration guide, not a statement that every Unity target has been compiled and validated.

## Source surface

- `CosmosManager.cs` — simulation manager
- `AudioManager.cs` — microphone/audio analysis
- `FFTAnalyzer.cs` — FFT helpers
- `ForceCalculator.cs` — simulation force calculations
- `MassInfluence.cs` — mass/influence component
- `IPCBridgeClient.cs` — versioned WebSocket IPC client
- `UIManager.cs` — UI control surface

Project terms involving golden-angle initialization, resonance, CST, or related historical terminology describe simulation/software choices. Their implementation is not proof of a new physical law or performance advantage.

## Prerequisites

For a live editor integration you will need:

- a compatible Unity editor (historical project metadata targets the Unity 2022.3 LTS era);
- platform microphone permission if testing live audio;
- Python 3.11+ for the active Python IPC package;
- the root `ipc` extra for WebSocket transport:

```bash
python -m pip install '.[ipc]'
```

Record the exact Unity version when publishing an integration result rather than assuming every later editor version is compatible.

## Scene setup

The repository preserves scripts/settings, but a usable scene may still require editor wiring. A representative hierarchy is:

```text
Main Camera
Directional Light
Cosmos Manager
  └─ ParticleSystem
Audio Manager
UI Canvas
```

Attach/configure the relevant scripts and UI references in the inspector for the scene you are testing.

Suggested historical parameters can be used as starting points, but they are experiment/configuration values rather than validated universal constants.

## Running the Python IPC bridge

From the repository root:

```bash
python -m pip install '.[ipc]'
python -m cosmic_synapse.ipc.bridge
```

The default server address is:

```text
ws://localhost:8765
```

The Python bridge loads the optional `websockets` package only when the transport is actually run.

## Versioned IPC protocol

The active protocol envelope is version 1:

```json
{
  "version": 1,
  "type": "command",
  "payload": {
    "command": "spawn_mass",
    "id": "spawn-example",
    "mass_type": "star",
    "position": [0.0, 0.0, 0.0],
    "properties": {}
  }
}
```

A status message uses the same envelope:

```json
{
  "version": 1,
  "type": "status",
  "payload": {
    "simulation_time": 123.45,
    "particle_count": 1000,
    "amplitude": 0.7
  }
}
```

Supported message types are defined by the Python schema. Do not send the older unversioned top-level command/status shapes from historical docs.

The Python bridge validates the envelope before dispatch and replies to commands with a versioned `command_received` acknowledgement containing the command id.

## Integration workflow

1. Install the root package plus `ipc` extra.
2. Start `python -m cosmic_synapse.ipc.bridge`.
3. Open/import the Unity project in the chosen editor version.
4. Configure the scene and set `IPCBridgeClient` to `ws://localhost:8765`.
5. Enter Play mode.
6. Confirm connection/runtime logs.
7. Send/receive versioned messages and capture both Python and Unity logs.
8. If needed, build a target player and repeat the IPC check outside the editor.

## Audio integration

Live microphone behavior depends on the OS, Unity audio device selection, permissions, sample rate, and hardware. The current deterministic workflow does not treat unavailable microphone hardware as a passing live-audio test.

When reporting a result, record the device and relevant audio configuration.

## Build status

A full Unity editor/player compile is an explicit integration gate that is **not** part of the current GitHub Actions workflow. Source-level IPC checks passing should not be reported as a successful Unity build.

For a reproducible Unity result, retain:

- exact repository commit SHA;
- Unity editor version;
- target platform/backend;
- import/compile logs;
- scene/configuration steps;
- runtime IPC logs;
- hardware/permission details;
- build/player result and warnings.

## Troubleshooting

### IPC connection fails

- Confirm the Python bridge is running on `localhost:8765`.
- Confirm the `ipc` Python extra is installed.
- Verify the Unity client URL.
- Inspect both Python and Unity logs.
- Confirm the messages use the v1 envelope.

### No microphone input

- Check OS permissions.
- Confirm Unity selected a valid microphone device.
- Test without IPC first to isolate the device path.

### Performance problems

- Reduce particle count/visual workload.
- Profile in the Unity editor/player rather than inferring performance from source.

## Evidence boundary

See the repository root:

- `docs/ARCHITECTURE.md`
- `docs/CLAIMS_AND_LIMITATIONS.md`
- `docs/REPRODUCIBILITY.md`
- `UNITY_PROJECT_COMPLETE.md` (historical filename, current status note)
