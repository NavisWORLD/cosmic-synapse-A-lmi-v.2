# God Music Quick Start

Use the Vite development path; it matches the current tested build workflow.

## Install and verify

```bash
cd "god music"
npm install
npm test
npm run dev
```

Open the localhost URL printed by Vite.

For a production-style preview:

```bash
npm run build
npm run preview
```

## Live use

1. Allow microphone access if you want live audio analysis.
2. Use the calibration control and provide a short voice/instrument sample.
3. Start the synthesized accompaniment.
4. Monitor the visualizers/log for unexpected behavior.

The application analyzes pitch/spectrum/tempo-related features and feeds deterministic/rule-based timing/harmonic logic. Historical UI/code terms such as “bio-signature” or phi harmonics are project names, not validated biometric measurements or evidence of golden-ratio superiority.

## Important boundaries

- The intended graph keeps microphone input on the analysis path rather than intentionally routing it to speaker output. Verify this on the target browser/device when feedback safety matters.
- “Prediction” is deterministic/rule-based in the current implementation, not learned behavior from a trained model.
- CI verifies Node utility tests and a Vite build, not every microphone/browser/mobile combination.

## Troubleshooting

If the app does not behave as expected, first run:

```bash
npm test
npm run build
```

Then check browser console output, microphone permissions, the Web Audio context, mixer controls, and the target device/audio interface.

See `../RUN_LOCALLY.md` and `../README.md` for more detail.
