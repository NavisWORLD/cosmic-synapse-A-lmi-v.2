# Architecture Documentation

## System Overview

God Music is a modular ES6+ application built with a clear separation of concerns.

## Module Structure

### Core Modules (`src/core/`)
- **AudioEngine.js** - Web Audio API initialization and microphone routing
- **BioSignature.js** - Real-time bio-frequency extraction and tracking
- **PhiHarmonics.js** - Golden ratio harmonic series generation
- **PsiCalculator.js** - Musical information density calculation

### Analysis Modules (`src/analysis/`)
- **PitchDetector.js** - YIN algorithm for fundamental frequency detection
- **SpectralAnalyzer.js** - FFT-based spectral analysis
- **TempoDetector.js** - Beat and tempo detection

### Prediction Modules (`src/prediction/`)
- **PredictiveEngine.js** - Main prediction orchestrator
- **PhraseTracker.js** - Musical phrase position tracking
- **GrooveLock.js** - Tempo locking mechanism
- **ChordPredictor.js** - φ-harmonic chord prediction

### Instrument Modules (`src/instruments/`)
- **InstrumentBase.js** - Base class for all instruments
- **Drums.js** - Drum kit synthesis
- **Bass.js** - Bass guitar synthesis
- **Guitar.js** - Guitar synthesis (Karplus-Strong)
- **Piano.js** - Piano synthesis (FM)
- **Strings.js** - String pad synthesis
- **Pads.js** - Ambient pad synthesis

### Audio Modules (`src/audio/`)
- **Mixer.js** - Master mixer and instrument buses
- **Synthesis.js** - Shared synthesis utilities

### UI Modules (`src/ui/`)
- **Logger.js** - Activity log system
- **Visualizer.js** - Spectrum and waveform displays
- **InstrumentControls.js** - Instrument volume and mute controls

## Audio Routing

### Critical: Microphone Isolation

```
Microphone Input:
  └─> createMediaStreamSource
      └─> Analyzer (read-only, NO connection to output)

Instrument Synthesis:
  └─> Individual Instrument Buses
      └─> Compressor
          └─> Master Gain
              └─> Audio Destination (Speakers)
```

**The microphone NEVER connects to the output chain.** This is enforced by:
1. Structural code separation (mic → analyzer only)
2. Runtime validation in `AudioEngine.validateRouting()`
3. Visual indicator in UI

## Data Flow

1. **User Input** → Microphone → Analyzer
2. **Analysis** → BioSignature extraction
3. **Harmonic Generation** → PhiHarmonics from bio-signature
4. **Prediction** → PredictiveEngine analyzes patterns
5. **Music Generation** → Instruments receive commands
6. **Output** → Mixer → Compressor → Speakers

## Prediction System

The prediction system operates on multiple levels:

1. **Phrase Level**: Tracks position within 4-bar phrases
2. **Beat Level**: Knows current beat position
3. **Harmonic Level**: Predicts next chord using φ-harmonics
4. **Rhythmic Level**: Locks tempo after 4 bars

## Extension Points

### Adding a New Instrument

1. Create class extending `InstrumentBase`
2. Implement synthesis methods
3. Add to `instruments` object in `main.js`
4. Create bus in mixer
5. Add UI controls

### Adding UI Components

1. Create class in `src/ui/`
2. Initialize in `main.js`
3. Add HTML elements if needed
4. Style in CSS files

## Dependencies

- **Web Audio API** - Core audio processing
- **No external libraries** - Pure ES6 modules
- **Vite** (optional) - Development/build tool

## Build Modes

### Standalone
- Open `index.html` directly
- Modules load via ES6 imports
- No build step required

### Vite Development
- `npm run dev` - Development server with HMR
- `npm run build` - Production build
- `npm run preview` - Preview production build

