/**
 * Piano - FM synthesis piano
 */

import { InstrumentBase } from './InstrumentBase.js';

export class Piano extends InstrumentBase {
    constructor(audioContext, mixerBus) {
        super('piano', audioContext, mixerBus);
        this.volume = 0.65;
        if (this.bus) {
            this.bus.gain.value = this.volume;
        }
    }

    /**
     * Play piano note (FM synthesis)
     */
    playNote(frequency, velocity = 0.3, time = null) {
        if (!this.enabled) return;

        const now = time || this.audioContext.currentTime;
        const carrier = this.audioContext.createOscillator();
        const modulator = this.audioContext.createOscillator();
        const modGain = this.audioContext.createGain();
        const gain = this.audioContext.createGain();

        carrier.frequency.value = frequency;
        modulator.frequency.value = frequency * 2;
        modGain.gain.value = frequency * 2;

        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(velocity, now + 0.01);
        gain.gain.exponentialRampToValueAtTime(velocity * 0.3, now + 0.2);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 1.5);

        modulator.connect(modGain);
        modGain.connect(carrier.frequency);
        carrier.connect(gain);
        gain.connect(this.bus);

        carrier.start(now);
        modulator.start(now);
        carrier.stop(now + 1.5);
        modulator.stop(now + 1.5);

        this.oscillators.push(carrier, modulator);
        this.nodes.push(modGain, gain);
    }

    /**
     * Play chord
     */
    playChord(frequencies, velocity = 0.3, time = null) {
        if (!this.enabled || !frequencies || frequencies.length === 0) return;

        const now = time || this.audioContext.currentTime;
        
        frequencies.forEach((freq, i) => {
            // Slight delay for natural chord attack
            setTimeout(() => {
                this.playNote(freq, velocity, now + i * 0.03);
            }, i * 30);
        });
    }
}

