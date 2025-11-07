# 12D Cosmic Synapse Engine - Extension Summary

## Overview
Extended the Python engine with audio input integration, Streamlit UI, real-time visualization, and token export functionality.

## Files Created/Modified

### 1. `12d_cosmic_synapse_streamlit.py` (NEW)
Complete Streamlit dashboard application with:
- **Audio Input Integration**: PyAudio support with automatic fallback to simulated audio
- **Interactive Controls**: All physics, adaptive state, synchronization, and timestep parameters
- **Real-time Visualization**: 3D particle plots and multi-panel metrics dashboard
- **Token Export**: JSON export with complete metadata
- **Recording/Replay**: Deterministic audio frame recording and replay

### 2. `requirements.txt` (NEW)
Python dependencies:
- streamlit>=1.28.0
- numpy>=1.24.0
- plotly>=5.17.0
- pandas>=2.0.0
- pyaudio>=0.2.11

### 3. `README_STREAMLIT.md` (NEW)
Complete documentation for the Streamlit UI including installation, usage, and features.

## Features Added

### 🎤 Audio Input Integration
- **PyAudio Support**: Real-time microphone capture
- **FFT Analysis**: Frequency extraction and spectral analysis
- **φ-Harmonic Generation**: Golden ratio-based harmonic series
- **Simulated Audio**: Automatic fallback when PyAudio unavailable
- **Threading**: Separate threads for audio capture and processing

### 📊 Real-time Visualization
- **3D Particle Scatter**: Interactive Plotly 3D visualization with energy coloring
- **Metrics Dashboard**: Multi-panel real-time charts:
  - Psi total over time
  - Synchronization order parameter (r)
  - Total energy
  - Token generation rate
  - Particle count
- **History Tracking**: Last 200 data points for smooth visualization

### 🎛️ Interactive Controls (Sidebar)
- **Audio Controls**: Start/stop audio capture
- **Determinism**: Seed setting and recording/replay controls
- **Physics**: Blend Lorenz, gravity, dark matter, epsilon, cutoff radius
- **Adaptive State**: k, γ, α, σ (similarity)
- **Synchronization**: K_sync parameter
- **Timestep**: Max dt and adaptive toggle
- **Dark Matter**: ρ₀ and r_s parameters
- **Particles**: Add/clear particles

### 📈 Diagnostics Display
- **Psi Breakdown**: All normalized terms displayed in real-time
- **Synchronization Metrics**: Order parameter r and mean theta
- **Conservation Diagnostics**: Energy, momentum, angular momentum, virial ratio
- **Live Updates**: All metrics update continuously

### 🎫 Token Management
- **Token Stream View**: Display recent tokens (last 50)
- **Token Export**: Download as JSON with complete metadata including:
  - Export date and time
  - Total tokens and generation rate
  - Simulation mode and seed
  - Physics configuration
  - All token data
- **Recording Export**: Export recorded audio frames for replay

### 🎲 Deterministic Replay
- **Recording**: Capture audio frames with complete data
- **Replay**: Feed recorded frames back into simulation
- **Seed Control**: Set deterministic seed for reproducible results
- **Export/Import**: Save and load recordings as JSON

## Technical Implementation

### Audio Processing Pipeline
1. **Capture Thread**: Continuously captures audio from microphone (or simulates)
2. **Processing Thread**: Performs FFT analysis and generates audio frames
3. **Simulation Integration**: Frames fed into simulator for token generation

### Visualization Architecture
- **Plotly Integration**: Interactive, web-native visualizations
- **Real-time Updates**: Auto-refresh mechanism when simulation active
- **Efficient Rendering**: Bounded history (200 points) for performance

### State Management
- **Session State**: Persistent across Streamlit reruns
- **Thread Safety**: Proper event handling for audio threads
- **Resource Cleanup**: Proper thread termination on stop

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run 12d_cosmic_synapse_streamlit.py
```

### Basic Workflow
1. **Start Audio**: Click "🎤 Start Audio" to begin capture
2. **Add Particles**: Click "➕ Add Particle" to add particles
3. **Adjust Controls**: Modify parameters in sidebar
4. **View Visualizations**: Switch between tabs
5. **Export Data**: Download tokens/recordings from Export tab

## Integration Points

### Engine Integration
- Uses existing `Simulator` class from `12d_cosmic_synapse_engine.py`
- All physics, adaptive state, and diagnostics already implemented
- Token stream and recorder classes fully integrated

### Audio Integration
- Modular design: works with or without PyAudio
- Thread-safe queue-based communication
- Efficient FFT processing

### UI Integration
- Streamlit's reactive framework
- Auto-refresh when simulation active
- Persistent session state

## Performance Considerations

- **Bounded History**: Limits data points to 200 for smooth rendering
- **Efficient Updates**: Only updates when simulation active
- **Thread Management**: Proper cleanup prevents resource leaks
- **Lazy Loading**: Imports only when needed

## Future Enhancements

Potential additions:
- Audio file upload/playback
- Particle trajectory trails
- Frequency spectrum visualization
- Real-time audio waveform display
- Multi-particle interaction visualization
- Export to other formats (CSV, HDF5)

## Compatibility

- **Python**: 3.8+
- **Operating Systems**: Windows, macOS, Linux
- **Browsers**: All modern browsers (Chrome, Firefox, Safari, Edge)
- **PyAudio**: Optional (app works without it)

## Notes

- PyAudio installation may require system dependencies (see README_STREAMLIT.md)
- Auto-refresh uses Streamlit's rerun mechanism
- All controls update simulation immediately
- Export includes complete metadata for deterministic verification

