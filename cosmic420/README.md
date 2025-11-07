# Cosmic Synapse Theory Simulation (Madsen's Theory)

A real-time particle physics simulation system that visualizes cosmic particles with audio, camera, and neural network integration. The system processes audio frequencies, camera input, and uses machine learning to adapt particle behaviors in real-time.

## Features

- **Real-time 3D Particle Visualization**: Interactive 3D visualization of cosmic particles
- **Audio Processing**: Microphone input processing with FFT frequency analysis
- **Camera Integration**: Real-time camera brightness detection for particle behavior
- **Neural Network Adaptation**: Machine learning-based particle behavior adaptation
- **Multiple Visualization Modes**: Streamlit web UI and Pygame standalone mode
- **Speech Recognition & TTS**: Voice commands and text-to-speech capabilities
- **Dark Matter Simulation**: Advanced physics modeling with NFW density profiles
- **Particle Replication**: Dynamic particle generation based on energy thresholds

## Prerequisites

- Python 3.7 or higher
- Windows, macOS, or Linux

## Installation

### 1. Install Python Dependencies

Install the required packages using pip:

```bash
pip install numpy torch torchvision matplotlib plotly streamlit sounddevice scipy numba pygame speechrecognition pyttsx3 pillow opencv-python
```

### Optional Dependencies

For additional features (optional):

```bash
# For volumetric heatmaps (requires VTK)
pip install mayavi

# For better audio processing
pip install pyaudio
```

### 2. Verify Installation

Check that all critical modules are available:

```bash
python -c "import numpy, torch, matplotlib, pygame; print('Core dependencies installed!')"
```

## Running the Simulation

### Extra Mode (Recommended for Interactive Experience)

Extra mode provides a standalone Pygame window with real-time visualization, audio reactivity, and a Tkinter log window.

**To start in Extra Mode:**

```bash
python Cosmo12Dplot-demo.py tk
```

**Features in Extra Mode:**
- Real-time Pygame visualization window (800x600)
- Tkinter log window showing system logs and audio frequencies
- Microphone audio capture with real-time frequency analysis
- Camera integration (if available)
- Speech recognition and text-to-speech
- Interactive particle system that reacts to sound

**Controls:**
- `F11`: Toggle fullscreen mode
- Close windows to exit

### Streamlit Web UI Mode (Default)

For a web-based interface with more controls:

```bash
python Cosmo12Dplot-demo.py
```

Or simply:

```bash
streamlit run Cosmo12Dplot-demo.py
```

**Features in Streamlit Mode:**
- Web-based interface accessible in browser
- Real-time 3D particle visualization
- Adjustable simulation parameters
- Audio capture controls
- Metrics and analytics
- Animation playback

## Usage Guide

### Starting Extra Mode

1. **Open a terminal/command prompt**
2. **Navigate to the script directory:**
   ```bash
   cd C:\Users\corys\Desktop
   ```

3. **Run with extra mode flag:**
   ```bash
   python Cosmo12Dplot-demo.py tk
   ```

4. **What to expect:**
   - A Pygame window will open showing particle visualization
   - A Tkinter log window will open showing system logs
   - The system will start capturing audio from your microphone
   - Particles will react to sound frequencies in real-time

### Audio Interaction

- **Blow into microphone**: Particles will react to frequency changes
- **Speak or make sounds**: The system processes audio frequencies and updates particle behavior
- **Frequency data**: Check the Tkinter log window for dominant frequency readings

### Camera Integration

If you have a camera connected:
- The system automatically detects camera brightness
- Brightness affects particle movement and behavior
- Camera data is processed in real-time

### Speech Recognition

- The system continuously listens for speech commands
- Recognized speech is processed and can trigger particle formations
- Check logs for recognized speech output

## Configuration

### Data Directory

The script uses a data directory for loading JSON, DB, and BA2 files. Default location:
- Windows: `C:\Users\phera\Desktop\test\data`
- Falls back to: `%USERPROFILE%\Desktop\data`

To change the data directory, modify line 130 in the script or use the Streamlit UI to set a custom path.

### Audio Settings

Default audio settings (can be modified in code):
- Sample Rate: 44100 Hz
- Duration: 0.1 seconds per capture
- Frequency Bins: [500, 2000, 4000, 8000] Hz

### Simulation Parameters

Key parameters (adjustable in Streamlit mode):
- Number of Particles: 100-1000
- Time Step: 0.1-10.0 seconds
- Replication Energy Threshold: 1e40-1e60 J
- Alpha: 1e-12 to 1e-8 J/m

## Troubleshooting

### Audio Not Working

**Problem**: No audio capture or microphone not detected

**Solutions**:
1. Check microphone permissions in system settings
2. Verify `sounddevice` is installed: `pip install sounddevice`
3. List available audio devices: `python -c "import sounddevice as sd; print(sd.query_devices())"`
4. Check that your microphone is set as the default input device

### Pygame Window Not Opening

**Problem**: Pygame window doesn't appear

**Solutions**:
1. Install pygame: `pip install pygame`
2. Check for display/display server issues
3. Try running in Streamlit mode instead: `python Cosmo12Dplot-demo.py`

### Camera Not Working

**Problem**: Camera capture fails

**Solutions**:
1. Install OpenCV: `pip install opencv-python`
2. Check camera permissions
3. Verify camera is connected and not in use by another application
4. The system will continue without camera if unavailable

### Missing Dependencies

**Problem**: Import errors or missing modules

**Solutions**:
1. Install all required packages (see Installation section)
2. Some features are optional - the script will run with warnings if optional modules are missing
3. Check Python version: `python --version` (requires 3.7+)

### Performance Issues

**Problem**: Slow performance or lag

**Solutions**:
1. Reduce number of particles in simulation
2. Close other resource-intensive applications
3. Adjust update intervals in the code
4. Use Streamlit mode for better performance on slower systems

## File Structure

- `Cosmo12Dplot-demo.py`: Main simulation script
- `cosmic_brain.json`: Stores frequency data over time (auto-generated)
- `learning_data.json`: Stores learning data for speech recognition (auto-generated)
- `data/`: Directory for JSON, DB, and BA2 data files

## Advanced Features

### Particle Formation

The system can form specific structures based on recognized words:
- "flower" - Forms flower-like particle arrangement
- "face", "mouth", "hello", "world" - Forms facial structures
- Custom text input generates particles based on character frequencies

### Neural Network Learning

- Particles adapt behavior based on audio frequencies
- Memory system stores frequency patterns
- Continuous learning updates particle energy and movement

### Dark Matter Simulation

- Uses Navarro-Frenk-White (NFW) density profile
- Computes dark matter influence on particle energy
- Advanced gravitational interactions

## Technical Details

### Core Components

- **Simulator**: Main simulation orchestrator
- **Particle**: Individual particle with mass, position, velocity, energy, frequency
- **CosmicNetwork**: Connectivity computation based on gravitational interactions
- **Dynamics**: Force computation and particle movement
- **LearningMechanism**: Memory-based learning system
- **AdaptiveBehavior**: Neural network-driven adaptation
- **Visualizer**: Plotly and matplotlib visualization

### Physics Model

- Gravitational interactions between particles
- Dark matter influence using NFW profile
- Energy-based particle replication
- Entropy calculations
- Frequency-based energy modulation

## License

This project implements the Cosmic Synapse Theory (Madsen's Theory) simulation.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review system logs in the Tkinter window (extra mode)
3. Check console output for error messages
4. Verify all dependencies are installed correctly

## Notes

- The simulation runs continuously until stopped
- Close windows or press Ctrl+C to exit
- Audio capture runs in background threads
- Camera capture is optional and runs if available
- All data is saved automatically to JSON files

---

**Enjoy exploring the Cosmic Synapse Theory simulation!** 🌌

