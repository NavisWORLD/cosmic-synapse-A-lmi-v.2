/**
 * Mathematical and musical constants
 * Based on Cory Shane Davis's Vibrational Information Theory
 */

export const PHI = 1.618033988749895; // Golden Ratio
export const GOLDEN_ANGLE = 137.5077640500378; // degrees
export const C_SOUND = 343.0; // Speed of sound in m/s

// Musical constants
export const A4_FREQUENCY = 440; // Standard tuning
export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
export const SEMITONES_PER_OCTAVE = 12;

// Default audio settings
export const DEFAULT_SAMPLE_RATE = 44100;
export const DEFAULT_FFT_SIZE = 8192;
export const DEFAULT_SMOOTHING = 0.8;

// Prediction constants
export const DEFAULT_PHRASE_LENGTH = 4; // bars
export const DEFAULT_BEATS_PER_BAR = 4;
export const ANTICIPATION_TIME = 0.5; // seconds ahead to prepare
export const GROOVE_LOCK_BARS = 4; // bars before locking tempo

// Instrument defaults
export const DEFAULT_VOLUMES = {
    drums: 0.8,
    bass: 0.75,
    guitar: 0.7,
    piano: 0.65,
    strings: 0.6,
    pads: 0.5
};

// Genre settings
export const GENRE_SETTINGS = {
    jazz: { tempo: 120, feel: 'swing' },
    rock: { tempo: 130, feel: 'straight' },
    pop: { tempo: 115, feel: 'straight' },
    funk: { tempo: 105, feel: 'groove' },
    ambient: { tempo: 70, feel: 'floating' },
    electronic: { tempo: 128, feel: 'precise' }
};

// Audio processing limits
export const MIN_FREQUENCY = 50; // Hz
export const MAX_FREQUENCY = 2000; // Hz
export const HUMAN_VOICE_MIN = 80; // Hz
export const HUMAN_VOICE_MAX = 800; // Hz

