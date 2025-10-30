/**
 * Bass - Synthesized bass guitar following root progressions
 */

import { InstrumentBase } from './InstrumentBase.js';

export class Bass extends InstrumentBase {
    constructor(audioContext, mixerBus) {
        super('bass', audioContext, mixerBus);
        this.volume = 0.75;
        this.currentRoot = 440;
        if (this.bus) {
            this.bus.gain.value = this.volume;
        }
    }

    /**
     * Play bass note
     */
    playNote(frequency, duration = 0.5, time = null) {
        if (!this.enabled) return;

        const now = time || this.audioContext.currentTime;
        const osc = this.audioContext.createOscillator();
        const filter = this.audioContext.createBiquadFilter();
        const gain = this.audioContext.createGain();

        osc.type = 'triangle';
        osc.frequency.value = frequency / 2; // Octave below

        filter.type = 'lowpass';
        filter.frequency.value = 800;
        filter.Q.value = 3;

        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(0.4, now + 0.05);
        gain.gain.setValueAtTime(0.4, now + duration - 0.1);
        gain.gain.linearRampToValueAtTime(0, now + duration);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.bus);

        osc.start(now);
        osc.stop(now + duration);

        this.oscillators.push(osc);
        this.nodes.push(filter, gain);
    }

    /**
     * Play bass line following root
     */
    playRoot(rootFrequency, tempo, time = null) {
        if (!this.enabled) return;
        this.currentRoot = rootFrequency;
        
        const beatDuration = 60 / tempo;
        const now = time || this.audioContext.currentTime;

        // Root on beat 1
        this.playNote(rootFrequency, beatDuration * 0.9, now);
        
        // Fifth on beat 3
        this.playNote(rootFrequency * 1.5, beatDuration * 0.9, now + beatDuration * 2);
    }

    /**
     * Update root frequency
     */
    updateRoot(frequency) {
        this.currentRoot = frequency;
    }
}

