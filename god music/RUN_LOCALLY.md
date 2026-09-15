# Running God Music Locally

God Music is a Vite/Web Audio browser prototype. The supported development path uses Node/Vite so ES modules, microphone permissions, and build behavior match the tested CI path.

## Recommended setup

```bash
cd "god music"
npm install
npm test
npm run dev
```

Open the localhost URL printed by Vite.

For a production-style local preview:

```bash
npm run build
npm run preview
```

CI verifies both `npm test` and `npm run build`.

## Using the app

When using the live microphone path:

1. Allow microphone access in the browser.
2. Use the calibration/control UI as provided by the current app.
3. Start the synthesized accompaniment.
4. Monitor the application log/visualizers for errors or unexpected audio routing.

The current prediction code is deterministic/rule-based. “Prediction” here does not mean a trained machine-learning model or biological-state inference.

## Microphone routing

The intended architecture keeps the microphone on the analysis path and synthesized instruments on the output path. The microphone is not intentionally routed to the speakers.

Because browser/device audio graphs vary, verify that separation on the target browser/device before relying on it for live use.

## Troubleshooting

### Microphone blocked

- Use a localhost or HTTPS origin.
- Allow microphone permission in the browser/OS.
- Reload after changing permission.
- Check whether another application has exclusive access to the device.

### App/build error

Run:

```bash
npm test
npm run build
```

Resolve deterministic/build errors before diagnosing live microphone behavior.

### No synthesized audio

- Check system/browser volume.
- Confirm the Web Audio context has started after a user gesture.
- Check instrument mixer controls/mutes.
- Inspect the browser console and application log.

## Browser/device status

A modern browser with Web Audio API and microphone support is required for live analysis. The current CI runs Node utility tests and a Vite build; it does not certify every Chrome/Firefox/Safari/Edge version, mobile device, microphone, or audio interface.

## Scope

A successful local browser run demonstrates the application on that browser/device. It does not establish production reliability, learned musical intelligence, biomedical/bio-frequency measurement, or golden-ratio superiority.

See `README.md` in this folder and the repository root `docs/CLAIMS_AND_LIMITATIONS.md` for the current evidence boundary.
