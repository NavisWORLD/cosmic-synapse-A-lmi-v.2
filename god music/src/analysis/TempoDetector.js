/**
 * TempoDetector - Detect tempo from audio signal
 */

export class TempoDetector {
    constructor() {
        this.beatTimes = [];
        this.intervals = [];
        this.threshold = 0.3;
        this.maxHistory = 8;
    }

    /**
     * Detect beat from audio buffer
     */
    detectBeat(buffer, currentTime) {
        const energy = this.calculateEnergy(buffer);
        
        if (energy > this.threshold) {
            this.beatTimes.push(currentTime);
            
            if (this.beatTimes.length >= 2) {
                const lastInterval = currentTime - this.beatTimes[this.beatTimes.length - 2];
                this.intervals.push(lastInterval);
                
                if (this.intervals.length > this.maxHistory) {
                    this.intervals.shift();
                }
            }
            
            // Keep only recent beats
            if (this.beatTimes.length > this.maxHistory * 2) {
                this.beatTimes.shift();
            }
        }
    }

    /**
     * Get stable tempo in BPM
     */
    getStableTempo() {
        if (this.intervals.length < 4) {
            return null;
        }
        
        // Calculate median interval (robust to outliers)
        const sorted = [...this.intervals].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        
        // Convert to BPM
        const bpm = 60 / median;
        
        // Clamp to reasonable range
        return Math.max(40, Math.min(200, bpm));
    }

    /**
     * Calculate energy (RMS)
     */
    calculateEnergy(buffer) {
        let sum = 0;
        for (let i = 0; i < buffer.length; i++) {
            sum += buffer[i] * buffer[i];
        }
        return Math.sqrt(sum / buffer.length);
    }

    /**
     * Reset detector
     */
    reset() {
        this.beatTimes = [];
        this.intervals = [];
    }
}

