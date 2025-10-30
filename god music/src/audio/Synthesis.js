/**
 * Synthesis - Shared synthesis utilities
 */

/**
 * Create noise buffer
 */
export function createNoiseBuffer(audioContext, duration) {
    const bufferSize = audioContext.sampleRate * duration;
    const buffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
    const data = buffer.getChannelData(0);
    
    for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
    }
    
    return buffer;
}

/**
 * Create filtered noise buffer with decay envelope
 */
export function createFilteredNoise(audioContext, duration, filterFreq, decay) {
    const buffer = createNoiseBuffer(audioContext, duration);
    const data = buffer.getChannelData(0);
    
    // Apply decay envelope
    for (let i = 0; i < data.length; i++) {
        const t = i / audioContext.sampleRate;
        const envelope = Math.exp(-t / decay);
        data[i] *= envelope;
    }
    
    return buffer;
}

/**
 * Apply ADSR envelope to gain node
 */
export function applyADSR(gainNode, time, attack, decay, sustain, release, duration) {
    const startTime = time || gainNode.context.currentTime;
    
    gainNode.gain.setValueAtTime(0, startTime);
    gainNode.gain.linearRampToValueAtTime(1, startTime + attack);
    gainNode.gain.linearRampToValueAtTime(sustain, startTime + attack + decay);
    gainNode.gain.setValueAtTime(sustain, startTime + duration - release);
    gainNode.gain.linearRampToValueAtTime(0, startTime + duration);
}

