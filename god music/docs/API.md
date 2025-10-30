# API Documentation

## Core Classes

### AudioEngine

```javascript
const engine = new AudioEngine();
await engine.initialize();
await engine.connectMicrophone(stream);
engine.validateRouting(); // Ensures mic is not routed to output
```

**Methods:**
- `initialize()` - Initialize Web Audio API
- `connectMicrophone(stream)` - Connect mic (analysis only)
- `validateRouting()` - Verify microphone isolation
- `getContext()` - Get AudioContext
- `getAnalyzer()` - Get analyzer node
- `getMasterGain()` - Get master gain node
- `getCompressor()` - Get compressor node
- `resume()` - Resume audio context
- `destroy()` - Cleanup

### BioSignature

```javascript
const bioSig = new BioSignature(audioEngine);
bioSig.addListener((signature) => {
    console.log(signature.fundamental, signature.tempo);
});
bioSig.update(); // Call in loop
```

**Methods:**
- `update()` - Update signature from current audio
- `getSignature()` - Get current signature object
- `addListener(callback)` - Add change listener
- `removeListener(callback)` - Remove listener

### PredictiveEngine

```javascript
const engine = new PredictiveEngine();
engine.start(audioContext.currentTime);
engine.update(currentTime, tempo, harmonics);
engine.addListener((eventType, data) => {
    // Handle prediction events
});
```

**Methods:**
- `start(startTime)` - Start prediction
- `update(currentTime, tempo, harmonics)` - Update (call in loop)
- `stop()` - Stop prediction
- `addListener(callback)` - Add event listener
- `getState()` - Get current prediction state

## Instrument Classes

All instruments extend `InstrumentBase`:

```javascript
const drums = new Drums(audioContext, mixerBus);
drums.setVolume(0.8);
drums.enable();
drums.disable();
drums.stop();
drums.playKick();
drums.playPattern(time, tempo, beats);
```

**Common Methods:**
- `setVolume(0-1)` - Set volume
- `getVolume()` - Get volume
- `enable()` - Enable instrument
- `disable()` - Disable instrument
- `isEnabled()` - Check if enabled
- `stop()` - Stop all active sound
- `destroy()` - Cleanup

## UI Classes

### Logger

```javascript
const logger = new Logger(containerElement);
logger.info('Message');
logger.success('Message');
logger.warning('Message');
logger.error('Message');
logger.clear();
```

### Visualizer

```javascript
const viz = new Visualizer(spectrumCanvas, waveformCanvas, analyzer);
viz.start();
viz.stop();
```

### InstrumentControls

```javascript
const controls = new InstrumentControls(
    containerElement,
    instrumentsObject,
    onVolumeChange,
    onMuteToggle
);
controls.updateDisplay();
```

## Constants

```javascript
import {
    PHI,                    // 1.618...
    C_SOUND,                // 343.0 m/s
    DEFAULT_VOLUMES,        // Object with default volumes
    GENRE_SETTINGS,         // Genre tempo/feel settings
    ANTICIPATION_TIME,      // 0.5 seconds
    GROOVE_LOCK_BARS        // 4 bars
} from './utils/constants.js';
```

## Helpers

```javascript
import {
    frequencyToNote(freq),  // Convert Hz to note name
    noteToFrequency(note),  // Convert note to Hz
    clamp(value, min, max), // Clamp value
    calculateRMS(buffer),   // Calculate RMS energy
    calculateZCR(buffer)    // Calculate zero crossing rate
} from './utils/helpers.js';
```

