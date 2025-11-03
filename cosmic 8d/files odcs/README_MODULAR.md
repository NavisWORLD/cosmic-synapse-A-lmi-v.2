# Modular Architecture - 12D Cosmic Synapse Demo

## Structure

The codebase has been organized into a modular structure:

```
cosmic-synapse-12d/
├── index.html (or cosmic_synapse_12d_prototype.html - single file version)
├── js/
│   ├── constants.js          - Physical constants
│   ├── particle-system.js    - Particle12D class
│   ├── physics-engine.js     - Physics calculations (to be extracted)
│   ├── audio-processor.js    - Audio handling (to be extracted)
│   ├── threejs-renderer.js   - Three.js rendering (to be extracted)
│   ├── ui-controller.js      - UI management (to be extracted)
│   └── main.js               - Main system class (to be extracted)
├── css/
│   └── styles.css            - All CSS styles
├── wasm/
│   ├── forces-wrapper.js     - WASM interface for forces
│   ├── omega-wrapper.js      - WASM interface for omega
│   └── state-update-wrapper.js - WASM interface for state updates
└── workers/
    ├── physics-worker.js     - Physics calculation worker
    └── audio-worker.js       - Audio processing worker
```

## Current Status

**Completed:**
- ✅ `constants.js` - Physical constants exported
- ✅ `particle-system.js` - Particle12D class exported
- ✅ `css/styles.css` - All styles extracted
- ✅ Worker modules created
- ✅ WASM wrapper modules created

**Modular HTML Version:**

The current `cosmic_synapse_12d_prototype.html` is a complete single-file application. To use the modular version:

1. Create a new `index.html` that loads modules via ES6 imports
2. Extract remaining classes to their respective module files
3. Update imports/exports as needed

## Usage

### Single-File Version (Current)
The existing `cosmic_synapse_12d_prototype.html` is fully functional and self-contained.

### Modular Version (Future)
To use ES6 modules, create an `index.html` that imports:

```html
<script type="module">
    import { Particle12D } from './js/particle-system.js';
    import { PHI, G, c } from './js/constants.js';
    // ... other imports
</script>
```

## Note

The modular structure is in place. The remaining refactoring (extracting CosmicSynapseSystem class, renderer, audio, UI) can be done incrementally while maintaining the working single-file version.

