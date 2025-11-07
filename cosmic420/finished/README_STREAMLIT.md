# 12D Cosmic Synapse - Streamlit UI

Interactive dashboard for the 12D Cosmic Synapse Theory engine with audio input, real-time visualization, and token export.

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Note: PyAudio installation may require additional system dependencies:
- **Windows**: Usually installs directly with pip
- **macOS**: `brew install portaudio` then `pip install pyaudio`
- **Linux**: `sudo apt-get install portaudio19-dev` then `pip install pyaudio`

If PyAudio is not available, the app will simulate audio input automatically.

## Running the App

```bash
streamlit run 12d_cosmic_synapse_streamlit.py
```

The app will open in your default web browser at `http://localhost:8501`

## Features

### 🎤 Audio Input
- Real-time audio capture from microphone (via PyAudio)
- Automatic fallback to simulated audio if PyAudio unavailable
- FFT analysis and frequency extraction
- φ-harmonic generation

### 🎛️ Interactive Controls
- **Physics Controls**: Blend Lorenz, gravity, dark matter, epsilon, cutoff radius
- **Adaptive State**: Coupling (k), decay (γ), memory (α), similarity (σ)
- **Synchronization**: Kuramoto coupling strength
- **Timestep**: Adaptive timestep with max dt control
- **Dark Matter**: Density (ρ₀) and scale radius (r_s)
- **Particles**: Add/clear particles

### 📊 Real-time Visualization
- **3D Particle Visualization**: Interactive 3D scatter plot with energy coloring
- **Metrics Dashboard**: Real-time charts for:
  - Psi total
  - Synchronization (r)
  - Energy
  - Token rate
  - Particle count

### 📈 Diagnostics
- **Psi Breakdown**: All normalized terms (energy, λ, velocity integral, x12 integral, omega, potential)
- **Synchronization Metrics**: Order parameter r and mean theta
- **Conservation Diagnostics**: Energy, momentum, angular momentum, virial ratio

### 🎫 Token Management
- View recent tokens in real-time
- Export tokens as JSON with complete metadata
- Export recordings for deterministic replay

### 🎲 Deterministic Replay
- Record audio frames
- Replay with fixed seed for identical outputs
- Export/import recordings

## Usage

1. **Start Audio**: Click "🎤 Start Audio" to begin capturing audio input
2. **Add Particles**: Click "➕ Add Particle" to add particles to the simulation
3. **Adjust Controls**: Use sidebar sliders to modify physics parameters
4. **View Visualizations**: Switch between tabs to see different visualizations
5. **Export Data**: Go to "Export" tab to download tokens and recordings

## File Structure

- `12d_cosmic_synapse_engine.py` - Core simulation engine
- `12d_cosmic_synapse_streamlit.py` - Streamlit UI application
- `requirements.txt` - Python dependencies

## Notes

- The app auto-refreshes when audio is running or particles are active
- Token generation occurs every 100ms when audio is active
- All controls update the simulation immediately
- Export includes complete metadata for deterministic verification

