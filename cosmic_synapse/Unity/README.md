# Unity 3D Cosmic Synapse Project

## Overview

This Unity project implements the real-time physics simulation for the Cosmic Synapse component of the Unified Vibrational Intelligence System.

## Features

- Real-time particle physics simulation
- Audio-driven stochastic resonance
- FFT-based spectral analysis
- IPC communication with A-LMI
- Interactive 3D visualization
- WebSocket integration

## Prerequisites

- **Unity 2022.3 LTS** or later
- Microphone access
- Windows/Mac/Linux (target platform)

## Setup Instructions

### 1. Open Project in Unity

1. Open Unity Hub
2. Click "Open" and select the `cosmic_synapse/Unity` folder
3. Wait for Unity to import the project

### 2. Install Required Packages

Unity will automatically install required packages:
- TextMeshPro
- Audio packages (built-in)

### 3. Set Up the Scene

1. Create a new scene: `File > New Scene > Basic`
2. Save as `Assets/Scenes/Main.unity`

### 4. Add GameObject Hierarchy

Create the following GameObject structure in the scene:

```
Main Camera
Directional Light
Cosmos Manager (Empty GameObject with CosmosManager script)
  └─ ParticleSystem
  └─ AudioSource
  └─ FFTAnalyzer script
  └─ ForceCalculator script
  └─ IPCBridgeClient script
Audio Manager (Empty GameObject with AudioManager script)
UI Canvas (with UIManager script)
  └─ Panel
     └─ Omega Slider
     └─ Lambda Slider
     └─ Damping Slider
     └─ Time Scale Slider
     └─ Start Button
     └─ Stop Button
     └─ Spawn Button
     └─ Microphone Toggle
     └─ Statistics Text
```

### 5. Configure Components

#### CosmosManager Component:
- Particle Count: 1000
- Omega: 0.5
- Lambda: 0.1
- Damping: 0.98
- Time Scale: 1.0

#### AudioManager Component:
- Sample Rate: 44100
- Buffer Size: 512
- Trigger Threshold: 0.5

#### IPCBridgeClient Component:
- Server URL: ws://localhost:8765
- Auto Connect: true

#### UIManager Component:
- Wire up all UI references in the inspector

### 6. Configure Particle System

Select the ParticleSystem under Cosmos Manager:
- Max Particles: 1000
- Start Lifetime: 10
- Start Speed: 0
- Start Size: 0.05
- Simulation Space: World
- Emission Rate: 1000 (over time)

### 7. Build Settings

1. Go to `File > Build Settings`
2. Select your target platform
3. Click "Switch Platform"
4. Click "Build" to create executable

## Running the Simulation

### Standalone Mode

1. Press Play in Unity Editor
2. Or build and run the executable
3. Toggle microphone on to enable audio-driven resonance
4. Adjust parameters using UI sliders

### Integrated Mode (with A-LMI)

1. Start A-LMI system first: `python main.py`
2. Start IPC bridge: `python cosmic_synapse/ipc/bridge.py`
3. Launch Unity simulation
4. Unity will connect automatically via WebSocket

## Controls

- **Omega Slider**: Adjust golden ratio frequency (φ parameter)
- **Lambda Slider**: Adjust Lyapunov exponent for stochastic resonance
- **Damping Slider**: Adjust velocity damping factor
- **Time Scale Slider**: Adjust simulation speed
- **Microphone Toggle**: Enable/disable audio input
- **Spawn Button**: Manually spawn a mass at center

## IPC Communication

The Unity client communicates with the A-LMI system via WebSocket:

### Commands from A-LMI:
```json
{
  "type": "command",
  "command": "spawn_mass",
  "position": [0, 0, 0],
  "properties": {
    "mass_type": "star"
  }
}
```

### Status to A-LMI:
```json
{
  "type": "status",
  "simulation_time": 123.45,
  "particle_count": 1000,
  "amplitude": 0.7
}
```

## Troubleshooting

### No Audio Input
- Check microphone permissions in OS settings
- Verify microphone device is selected in AudioManager
- Test microphone in system settings

### IPC Connection Failed
- Ensure A-LMI is running
- Check WebSocket URL is correct
- Verify firewall allows localhost connections

### Performance Issues
- Reduce particle count
- Lower FFT size
- Disable audio processing if not needed

## Scripts Reference

### Core Scripts:
- **CosmosManager.cs**: Main simulation manager
- **AudioManager.cs**: Microphone input and audio analysis
- **FFTAnalyzer.cs**: FFT standardization and spectral analysis
- **ForceCalculator.cs**: Physics force calculations
- **IPCBridgeClient.cs**: WebSocket communication
- **UIManager.cs**: User interface management
- **MassInfluence.cs**: Gravitational influence component

### Key Features Implemented:
- Golden angle particle initialization
- Conservative bowl potential
- Swirl forces
- Stochastic resonance from audio
- IPC bridge for A-LMI communication
- Real-time parameter adjustment

## Performance Optimization

For large particle counts:
- Use GPU particles (Unity's VFX Graph)
- Implement spatial partitioning
- Reduce update frequency for non-critical components
- Use object pooling for spawned masses

## Next Steps

- Implement shader-based particle rendering
- Add trail rendering for particle paths
- Create field visualization (heat map)
- Add camera controls (orbit, zoom, pan)
- Export simulation data for analysis

