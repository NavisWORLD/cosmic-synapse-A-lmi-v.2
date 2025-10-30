/**
 * GrooveLock - Lock tempo after establishing groove
 */

import { GROOVE_LOCK_BARS } from '../utils/constants.js';

export class GrooveLock {
    constructor() {
        this.isLocked = false;
        this.lockedTempo = null;
        this.lockBars = GROOVE_LOCK_BARS;
        this.tempoHistory = [];
    }

    /**
     * Record tempo sample
     */
    recordTempo(tempo, currentBar) {
        if (tempo && tempo > 0) {
            this.tempoHistory.push({ tempo, bar: currentBar });
            
            // Keep only recent history
            if (this.tempoHistory.length > 20) {
                this.tempoHistory.shift();
            }
        }

        // Check if we should lock
        if (!this.isLocked && currentBar >= this.lockBars) {
            this.lock();
        }
    }

    /**
     * Lock the groove
     */
    lock() {
        if (this.isLocked) return;

        // Calculate average tempo from history
        if (this.tempoHistory.length >= 4) {
            const tempos = this.tempoHistory
                .slice(-8)
                .map(item => item.tempo)
                .sort((a, b) => a - b);
            
            // Use median for robustness
            const medianIndex = Math.floor(tempos.length / 2);
            this.lockedTempo = tempos[medianIndex];
            this.isLocked = true;
            
            return true;
        }
        
        return false;
    }

    /**
     * Get locked tempo or current average
     */
    getTempo() {
        if (this.isLocked && this.lockedTempo) {
            return this.lockedTempo;
        }
        
        // Return average if not locked yet
        if (this.tempoHistory.length > 0) {
            const sum = this.tempoHistory.reduce((acc, item) => acc + item.tempo, 0);
            return sum / this.tempoHistory.length;
        }
        
        return 120; // Default
    }

    /**
     * Check if locked
     */
    isGrooveLocked() {
        return this.isLocked;
    }

    /**
     * Reset
     */
    reset() {
        this.isLocked = false;
        this.lockedTempo = null;
        this.tempoHistory = [];
    }
}

