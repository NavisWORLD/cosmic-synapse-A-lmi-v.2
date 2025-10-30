/**
 * Drums - Synthesized drum kit with predictive fills
 */

import { InstrumentBase } from './InstrumentBase.js';
import { createFilteredNoise, applyADSR } from '../audio/Synthesis.js';

export class Drums extends InstrumentBase {
    constructor(audioContext, mixerBus) {
        super('drums', audioContext, mixerBus);
        this.volume = 0.8;
        if (this.bus) {
            this.bus.gain.value = this.volume;
        }
    }

    /**
     * Play kick drum
     */
    playKick(time = null) {
        if (!this.enabled) return;

        const now = time || this.audioContext.currentTime;
        const osc = this.audioContext.createOscillator();
        const gain = this.audioContext.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(150, now);
        osc.frequency.exponentialRampToValueAtTime(50, now + 0.1);

        gain.gain.setValueAtTime(1, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);

        osc.connect(gain);
        gain.connect(this.bus);

        osc.start(now);
        osc.stop(now + 0.2);

        this.oscillators.push(osc);
        this.nodes.push(gain);
    }

    /**
     * Play snare drum
     */
    playSnare(time = null) {
        if (!this.enabled) return;

        const now = time || this.audioContext.currentTime;
        const buffer = createFilteredNoise(this.audioContext, 0.15, 1000, 0.05);
        const noise = this.audioContext.createBufferSource();
        noise.buffer = buffer;

        const filter = this.audioContext.createBiquadFilter();
        filter.type = 'highpass';
        filter.frequency.value = 1000;

        const gain = this.audioContext.createGain();
        gain.gain.setValueAtTime(0.7, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);

        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.bus);

        noise.start(now);
        noise.stop(now + 0.15);

        this.nodes.push(noise, filter, gain);
    }

    /**
     * Play hi-hat
     */
    playHiHat(time = null) {
        if (!this.enabled) return;

        const now = time || this.audioContext.currentTime;
        const buffer = createFilteredNoise(this.audioContext, 0.05, 7000, 0.01);
        const noise = this.audioContext.createBufferSource();
        noise.buffer = buffer;

        const filter = this.audioContext.createBiquadFilter();
        filter.type = 'highpass';
        filter.frequency.value = 7000;

        const gain = this.audioContext.createGain();
        gain.gain.setValueAtTime(0.3, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.05);

        noise.connect(filter);
        filter.connect(gain);
        gain.connect(this.bus);

        noise.start(now);
        noise.stop(now + 0.05);

        this.nodes.push(noise, filter, gain);
    }

    /**
     * Play drum pattern
     */
    playPattern(startTime, tempo, beats = 4) {
        if (!this.enabled) return;

        const beatDuration = 60 / tempo;
        const now = startTime || this.audioContext.currentTime;

        // Kick on 1 and 3
        this.playKick(now);
        this.playKick(now + beatDuration * 2);

        // Snare on 2 and 4
        this.playSnare(now + beatDuration);
        this.playSnare(now + beatDuration * 3);

        // Hi-hat pattern (8th notes)
        for (let i = 0; i < beats * 2; i++) {
            this.playHiHat(now + (beatDuration / 2) * i);
        }
    }
}

