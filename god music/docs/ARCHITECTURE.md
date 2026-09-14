# God Music Architecture

God Music is a modular ES6/Web Audio application. The active restoration describes it as an algorithmic reactive/predictive browser music prototype.

Historical class names such as `BioSignature`, `PhiHarmonics`, and `PsiCalculator` are retained for compatibility and project lineage. They should be read as software abstractions, not validated biomedical measurements or new-physics constructs.

## Module structure

### Core (`src/core/`)

- `AudioEngine.js` — Web Audio context, analyzer, synthesis-output routing
- `BioSignature.js` — historical class name for tracked audio-derived features
- `PhiHarmonics.js` — deterministic harmonic-rule generation using the project phi parameter
- `PsiCalculator.js` — project-specific musical information calculation

### Analysis (`src/analysis/`)

- `PitchDetector.js` — pitch/fundamental-frequency estimation
- `SpectralAnalyzer.js` — FFT/spectral helpers
- `TempoDetector.js` — beat/tempo estimation

### Prediction (`src/prediction/`)

- `PredictiveEngine.js` — deterministic orchestration
- `PhraseTracker.js` — phrase/bar/beat position tracking
- `GrooveLock.js` — tempo-lock rule state
- `ChordPredictor.js` — rule-based next-chord helper using the configured harmonic sequence

The current prediction layer is deterministic/rule-based, not a trained machine-learning model.

### Instruments (`src/instruments/`)

Synthesized drums, bass, guitar, piano, strings, pads, and the shared instrument base.

### Audio (`src/audio/`)

Mixer and synthesis utilities.

### UI (`src/ui/`)

Logger, visualizer, and instrument controls.

## Intended audio routing

```text
microphone/media stream
  └─ analyzer path

synthesized instruments
  └─ instrument buses
      └─ compressor/master
          └─ audio destination
```

The source is intentionally structured so the microphone analysis source is not connected to the synthesized output chain. Runtime/source checks are useful, but target-browser/device verification is still required before making an absolute live-audio routing claim.

## Data flow

```text
microphone sample
   ↓
audio analysis (pitch / spectrum / tempo-related features)
   ↓
historical BioSignature feature object
   ↓
deterministic harmonic/timing rules
   ↓
phrase / groove / chord prediction helpers
   ↓
synthesized instruments
   ↓
mixer / output
```

No step in that flow establishes biological-state inference.

## Prediction behavior

The prediction subsystem tracks phrase/beat state, tempo history, and deterministic chord/harmonic rules. “Prediction” means forward rule logic based on current state; it does not imply learned intelligence.

## Extension points

### Add an instrument

1. Extend `InstrumentBase`.
2. Implement the synthesis/play methods.
3. Register the instrument in the application entry point.
4. Add/connect its mixer bus.
5. Add UI controls and deterministic tests where practical.

### Add analysis/prediction behavior

Keep raw audio-analysis features separate from semantic/biological claims. For a learned model, record the model/revision, training/evaluation assumptions, input/output contract, and evidence separately.

## Dependencies

Runtime behavior primarily uses browser Web Audio/ES modules. Vite is the supported development/build path and is what CI verifies.

```bash
npm install
npm test
npm run dev
npm run build
```

A direct `file://` open is not treated as equivalent to the tested Vite/localhost workflow because ES-module and microphone security behavior can differ across browsers.

## Evidence boundary

CI currently verifies deterministic Node utilities and a Vite production build. It does not certify browser microphone routing, latency, mobile compatibility, audio quality, biometric inference, trained AI, or performance advantages from phi/golden-ratio rules.

See the repository root `docs/CLAIMS_AND_LIMITATIONS.md` and `docs/REPRODUCIBILITY.md`.
