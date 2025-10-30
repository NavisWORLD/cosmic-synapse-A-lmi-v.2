/**
 * InstrumentBase - Base class for all instruments
 */

export class InstrumentBase {
    constructor(name, audioContext, mixerBus) {
        this.name = name;
        this.audioContext = audioContext;
        this.bus = mixerBus;
        
        this.enabled = true;
        this.volume = 0.5;
        this.oscillators = [];
        this.nodes = [];
        
        // Update bus gain
        if (this.bus) {
            this.bus.gain.value = this.volume;
        }
    }

    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        if (this.bus) {
            this.bus.gain.value = this.volume;
        }
    }

    getVolume() {
        return this.volume;
    }

    enable() {
        this.enabled = true;
    }

    disable() {
        this.enabled = false;
        this.stop();
    }

    isEnabled() {
        return this.enabled;
    }

    stop() {
        this.oscillators.forEach(osc => {
            try {
                osc.stop();
                osc.disconnect();
            } catch (e) {}
        });
        this.oscillators = [];
        
        this.nodes.forEach(node => {
            try {
                node.disconnect();
            } catch (e) {}
        });
        this.nodes = [];
    }

    destroy() {
        this.stop();
        if (this.bus) {
            this.bus.disconnect();
        }
    }
}

