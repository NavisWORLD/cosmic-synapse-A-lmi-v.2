/**
 * Input validation utilities
 */

import { MIN_FREQUENCY, MAX_FREQUENCY, HUMAN_VOICE_MIN, HUMAN_VOICE_MAX } from './constants.js';
import { clamp } from './helpers.js';

/**
 * Validate frequency is in valid range
 */
export function validateFrequency(freq) {
    if (typeof freq !== 'number' || !isFinite(freq)) {
        return false;
    }
    return freq >= MIN_FREQUENCY && freq <= MAX_FREQUENCY;
}

/**
 * Validate frequency is in human voice range
 */
export function validateHumanVoiceFrequency(freq) {
    if (typeof freq !== 'number' || !isFinite(freq)) {
        return false;
    }
    return freq >= HUMAN_VOICE_MIN && freq <= HUMAN_VOICE_MAX;
}

/**
 * Validate and clamp volume (0-1)
 */
export function validateVolume(volume) {
    return clamp(volume, 0, 1);
}

/**
 * Validate tempo (BPM)
 */
export function validateTempo(tempo) {
    if (typeof tempo !== 'number' || !isFinite(tempo)) {
        return false;
    }
    return tempo >= 40 && tempo <= 200;
}

/**
 * Validate audio context state
 */
export function validateAudioContext(context) {
    return context instanceof (window.AudioContext || window.webkitAudioContext);
}

/**
 * Validate audio node connection
 */
export function validateNodeConnection(node) {
    return node instanceof AudioNode;
}

