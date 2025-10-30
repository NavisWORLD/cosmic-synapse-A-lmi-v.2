/**
 * Strings - Sustained string pad synthesis
 */

import { InstrumentBase } from './InstrumentBase.js';

export class Strings extends InstrumentBase {
    constructor(audioContext, mixerBus) {
        super('strings', audioContext, mixerBus);
        this.volume = 0.6;
        if (this.bus) {
            this.bus.gain.value = this.volume;
        }
    }

    /**
     * Play sustained string note
     */
    playPad(frequencies, duration = 4, time = null) {
        if (!this.enabled || !frequencies || frequencies.length === 0) return;

        const now = time || this.audioContext.currentTime;

        frequencies.forEach((freq, i) => {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();

            osc.type = 'sine';
            osc.frequency.value = freq;

            gain.gain.setValueAtTime(0, now);
            gain.gain.linearRampToValueAtTime(0.1, now + 2);
            gain.gain.setValueAtTime(0.1, now + duration - 0.5);
            gain.gain.linearRampToValueAtTime(0, now + duration);

            osc.connect(gain);
            gain.connect(this.bus);

            osc.start(now);
            osc.stop(now + duration);

            this.oscillators.push(osc);
            this.nodes.push(gain);
        });
    }
}

