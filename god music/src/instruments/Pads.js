/**
 * Pads - Ambient pad synthesis
 */

import { InstrumentBase } from './InstrumentBase.js';

export class Pads extends InstrumentBase {
    constructor(audioContext, mixerBus) {
        super('pads', audioContext, mixerBus);
        this.volume = 0.5;
        if (this.bus) {
            this.bus.gain.value = this.volume;
        }
    }

    /**
     * Play ambient pad
     */
    playPad(frequency, duration = 10, time = null) {
        if (!this.enabled) return;

        const now = time || this.audioContext.currentTime;
        const osc = this.audioContext.createOscillator();
        const filter = this.audioContext.createBiquadFilter();
        const gain = this.audioContext.createGain();

        osc.type = 'sine';
        osc.frequency.value = frequency;

        filter.type = 'lowpass';
        filter.frequency.value = 2000;
        filter.Q.value = 1;

        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(0.15, now + 2);
        gain.gain.setValueAtTime(0.15, now + duration - 2);
        gain.gain.linearRampToValueAtTime(0, now + duration);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.bus);

        osc.start(now);
        osc.stop(now + duration);

        this.oscillators.push(osc);
        this.nodes.push(filter, gain);
    }
}

