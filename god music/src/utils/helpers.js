/**
 * Utility helper functions
 */

import { NOTE_NAMES, A4_FREQUENCY, SEMITONES_PER_OCTAVE } from './constants.js';

/**
 * Convert frequency to note name
 * @param {number} frequency - Frequency in Hz
 * @returns {string} Note name (e.g., "C4", "A#5")
 */
export function frequencyToNote(frequency) {
    const semitones = 12 * Math.log2(frequency / A4_FREQUENCY);
    const noteIndex = Math.round(semitones) % SEMITONES_PER_OCTAVE;
    const octave = Math.floor((Math.round(semitones) + 57) / SEMITONES_PER_OCTAVE);
    const note = NOTE_NAMES[(noteIndex + SEMITONES_PER_OCTAVE) % SEMITONES_PER_OCTAVE];
    return `${note}${octave}`;
}

/**
 * Convert note name to frequency
 * @param {string} note - Note name (e.g., "C4", "A#5")
 * @returns {number} Frequency in Hz
 */
export function noteToFrequency(note) {
    const match = note.match(/([A-G]#?)(\d+)/);
    if (!match) return A4_FREQUENCY;
    
    const [, noteName, octave] = match;
    const noteIndex = NOTE_NAMES.indexOf(noteName);
    const semitones = (parseInt(octave) - 4) * 12 + noteIndex;
    
    return A4_FREQUENCY * Math.pow(2, semitones / 12);
}

/**
 * Clamp value between min and max
 */
export function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

/**
 * Map value from one range to another
 */
export function mapRange(value, inMin, inMax, outMin, outMax) {
    return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

/**
 * Calculate RMS (Root Mean Square) of audio buffer
 */
export function calculateRMS(buffer) {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
        sum += buffer[i] * buffer[i];
    }
    return Math.sqrt(sum / buffer.length);
}

/**
 * Calculate Zero Crossing Rate
 */
export function calculateZCR(buffer) {
    let crossings = 0;
    for (let i = 1; i < buffer.length; i++) {
        if ((buffer[i - 1] >= 0 && buffer[i] < 0) ||
            (buffer[i - 1] < 0 && buffer[i] >= 0)) {
            crossings++;
        }
    }
    return crossings / buffer.length;
}

/**
 * Linear interpolation
 */
export function lerp(start, end, t) {
    return start + (end - start) * t;
}

/**
 * Exponential moving average
 */
export function exponentialMovingAverage(current, previous, alpha) {
    return alpha * current + (1 - alpha) * previous;
}

/**
 * Format time as MM:SS
 */
export function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Debounce function
 */
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 */
export function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

