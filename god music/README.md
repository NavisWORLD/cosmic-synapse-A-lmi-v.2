# God Music

God Music is a browser-based, algorithmic reactive and predictive music-conductor experiment from the COSMIC SYNAPSE research lineage.

It analyzes microphone input with Web Audio APIs and uses deterministic signal-analysis, timing, harmonic, and rule-based prediction code to drive synthesized accompaniment. The current implementation is not presented as a trained machine-learning model, biological signal reader, or proof of any special physical effect.

## What is implemented

- Pitch, tempo, and spectral analysis helpers
- Phrase tracking and groove locking
- Deterministic next-chord generation from the current harmonic rules
- Synthesized drums, bass, guitar, piano, strings, and pads
- Spectrum and waveform visualization
- Browser microphone analysis path that is kept separate from the synthesized output path

The active deterministic JavaScript tests cover core analysis/prediction behavior, and CI also performs a Vite production build.

## Architecture

- **Core:** `AudioEngine`, `BioSignature`, `PhiHarmonics`, `PsiCalculator`
- **Analysis:** `PitchDetector`, `SpectralAnalyzer`, `TempoDetector`
- **Prediction:** `PredictiveEngine`, `PhraseTracker`, `GrooveLock`, `ChordPredictor`
- **Instruments:** drums, bass, guitar, piano, strings, pads
- **Audio:** mixer and synthesis utilities
- **UI:** logger and visualizer

Names such as `BioSignature`, `PhiHarmonics`, and `PsiCalculator` are preserved historical/project terminology. They describe software mechanisms in this repository; they should not be read as validated biomedical or new-physics claims.

## Microphone routing

The intended routing is:

- **Analysis path:** microphone → analyzer
- **Output path:** synthesized instruments → compressor/master → speakers

The microphone signal is used for analysis and is not intentionally connected to the output graph.

## Development

```bash
npm install
npm test
npm run dev
npm run build
npm run preview
```

A modern browser with Web Audio API support is required for the live application. Microphone permission is required only for live microphone analysis.

## Scope and evidence

The repository currently verifies software behavior and buildability. It does **not** claim that the harmonic rules outperform conventional music systems, that golden-ratio choices have a demonstrated advantage, or that the application infers biological state.

See the root `docs/CLAIMS_AND_LIMITATIONS.md` and `docs/REPRODUCIBILITY.md` for the evidence boundary used by the restoration branch.

## License

GPL-3.0-only. See the repository root `LICENSE`.

## Author

Cory Shane Davis
