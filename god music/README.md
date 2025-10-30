# God Music - Professional AI Music Conductor

The world's first AI band system with true predictive intelligence. Built on the Unified Theory of Vibrational Information Architecture.

## Features

- **Predictive Intelligence** - Anticipates your next musical moves
- **Bio-Frequency Matching** - Generates harmonics based on your voice signature
- **φ-Harmonic Generation** - Uses golden ratio for natural harmonic relationships
- **Microphone Isolation** - Microphone is NEVER output to speakers (analysis only)
- **Professional Instruments** - Full drum kit, bass, guitar, piano, strings, pads
- **Groove Lock** - Locks tempo after establishing your rhythm
- **Real-Time Visualization** - Spectrum and waveform displays

## Architecture

The system is fully modular:

- **Core**: AudioEngine, BioSignature, PhiHarmonics, PsiCalculator
- **Analysis**: PitchDetector, SpectralAnalyzer, TempoDetector
- **Prediction**: PredictiveEngine, PhraseTracker, GrooveLock, ChordPredictor
- **Instruments**: Drums, Bass, Guitar, Piano, Strings, Pads
- **Audio**: Mixer, Synthesis utilities
- **UI**: Logger, Visualizer

## Critical: Microphone Routing

The microphone **NEVER** connects to the output. The audio routing is:

- **Analysis Path**: Microphone → Analyzer (read-only)
- **Output Path**: Instruments → Compressor → Master → Speakers

This is enforced in `AudioEngine.js` and validated at runtime.

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Browser Requirements

- Modern browser with Web Audio API support
- Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- Microphone access required

## License

Public GNU 3.0

## Author

Cory Shane Davis - Based on Unified Theory of Vibrational Information Architecture

