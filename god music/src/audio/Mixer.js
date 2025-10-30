/**
 * Mixer - Master audio mixer for all instruments
 */

export class Mixer {
    constructor(audioContext, compressor, masterGain) {
        this.audioContext = audioContext;
        this.compressor = compressor;
        this.masterGain = masterGain;
        
        // Individual instrument buses
        this.buses = new Map();
    }

    /**
     * Create a bus for an instrument
     */
    createBus(instrumentName) {
        const bus = this.audioContext.createGain();
        bus.connect(this.compressor);
        this.buses.set(instrumentName, bus);
        return bus;
    }

    /**
     * Get bus for instrument
     */
    getBus(instrumentName) {
        if (!this.buses.has(instrumentName)) {
            return this.createBus(instrumentName);
        }
        return this.buses.get(instrumentName);
    }

    /**
     * Set master volume (0-1)
     */
    setMasterVolume(volume) {
        this.masterGain.gain.value = Math.max(0, Math.min(1, volume));
    }

    /**
     * Get master volume
     */
    getMasterVolume() {
        return this.masterGain.gain.value;
    }
}

