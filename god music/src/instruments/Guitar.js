/**
 * Guitar - Synthesized guitar with strumming patterns
 */

import { InstrumentBase } from './InstrumentBase.js';

export class Guitar extends InstrumentBase {
    constructor(audioContext, mixerBus) {
        super('guitar', audioContext, mixerBus);
        this.volume = 0.7;
        if (this.bus) {
            this.bus.gain.value = this.volume;
        }
    }

    /**
     * Play guitar string (Karplus-Strong algorithm)
     */
    playString(frequency, time = null) {
        if (!this.enabled) return;

        const now = time || this.audioContext.currentTime;
        const sampleRate = this.audioContext.sampleRate;
        const duration = 2.0;
        const delayLength = Math.round(sampleRate / frequency);
        const bufferSize = sampleRate * duration;
        const buffer = this.audioContext.createBuffer(1, bufferSize, sampleRate);
        const output = buffer.getChannelData(0);

        // Initial noise burst
        for (let i = 0; i < delayLength; i++) {
            output[i] = Math.random() * 2 - 1;
        }

        // Karplus-Strong feedback loop
        for (let i = delayLength; i < bufferSize; i++) {
            output[i] = 0.996 * 0.5 * (output[i - delayLength] + output[i - delayLength + 1]);
        }

        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;

        const gain = this.audioContext.createGain();
        gain.gain.value = 0.2;

        source.connect(gain);
        gain.connect(this.bus);

        source.start(now);
        source.stop(now + duration);

        this.nodes.push(source, gain);
    }

    /**
     * Play chord (strummed)
     */
    playChord(frequencies, time = null) {
        if (!this.enabled || !frequencies || frequencies.length === 0) return;

        const now = time || this.audioContext.currentTime;
        
        frequencies.forEach((freq, i) => {
            // Strum delay
            const delay = i * 0.02;
            this.playString(freq, now + delay);
        });
    }
}

