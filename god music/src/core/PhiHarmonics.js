/**
 * PhiHarmonics - Generate φ-harmonic series using golden ratio
 */

import { PHI } from '../utils/constants.js';
import { clamp } from '../utils/helpers.js';

export class PhiHarmonics {
    constructor(fundamental = 440) {
        this.fundamental = fundamental;
        this.harmonics = [];
        this.generate();
    }

    /**
     * Generate φ-harmonic series
     */
    generate(fundamental = null) {
        if (fundamental !== null) {
            this.fundamental = fundamental;
        }

        this.harmonics = [];
        const numHarmonics = 12;

        for (let i = 0; i < numHarmonics; i++) {
            // Exponent relative to center
            const exponent = i - (numHarmonics / 2);
            
            // Generate frequency: f₀ × φⁿ
            let freq = this.fundamental * Math.pow(PHI, exponent);
            
            // Fold into musical range (2 octaves around fundamental)
            while (freq > this.fundamental * 4) freq /= 2;
            while (freq < this.fundamental / 4) freq *= 2;
            
            // Add stochastic modulation (±1%)
            const noise = (Math.random() - 0.5) * 0.02;
            freq *= (1 + noise);
            
            // Clamp to audible range
            freq = clamp(freq, 50, 2000);
            
            this.harmonics.push(freq);
        }

        // Sort by frequency
        this.harmonics.sort((a, b) => a - b);
        
        return this.harmonics;
    }

    /**
     * Get all harmonics
     */
    getHarmonics() {
        return [...this.harmonics];
    }

    /**
     * Get harmonic at index
     */
    getHarmonic(index) {
        return this.harmonics[index] || null;
    }

    /**
     * Update fundamental and regenerate
     */
    updateFundamental(fundamental) {
        if (fundamental > 0) {
            this.generate(fundamental);
        }
    }

    /**
     * Get harmonics for chord (root, third, fifth, etc.)
     */
    getChordHarmonics(rootIndex = 0, chordType = 'major') {
        const chordIndices = {
            major: [0, 2, 4, 7],
            minor: [0, 2, 4, 6],
            seventh: [0, 2, 4, 7, 10],
            maj7: [0, 2, 4, 7, 11],
            sus4: [0, 3, 4]
        };

        const indices = chordIndices[chordType] || chordIndices.major;
        const chord = indices.map(idx => this.harmonics[(rootIndex + idx) % this.harmonics.length]);
        
        return chord.filter(freq => freq !== undefined);
    }
}

