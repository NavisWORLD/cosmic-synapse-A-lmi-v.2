/**
 * ChordPredictor - Predict next chord using φ-harmonics
 */

import { PHI } from '../utils/constants.js';

export class ChordPredictor {
    constructor() {
        this.recentChords = [];
        this.maxHistory = 8;
    }

    /**
     * Predict next chord based on previous chord and φ-harmonics
     */
    predictNextChord(previousChord, phiHarmonics) {
        if (!previousChord || !phiHarmonics || phiHarmonics.length === 0) {
            return null;
        }

        // Use φ-harmonic sequence
        const nextRoot = previousChord.root * PHI;
        
        // Fold into octave
        let folded = nextRoot;
        while (folded > previousChord.root * 2) folded /= 2;
        while (folded < previousChord.root / 2) folded *= 2;

        // Select chord type
        const chordTypes = ['major', 'minor', 'seventh', 'maj7', 'sus4'];
        const typeIndex = Math.floor(folded / 100) % chordTypes.length;

        const chord = {
            root: folded,
            type: chordTypes[typeIndex],
            frequencies: this.generateChordFrequencies(folded, chordTypes[typeIndex], phiHarmonics)
        };

        // Store in history
        this.recentChords.push(chord);
        if (this.recentChords.length > this.maxHistory) {
            this.recentChords.shift();
        }

        return chord;
    }

    /**
     * Generate chord frequencies from φ-harmonics
     */
    generateChordFrequencies(root, type, phiHarmonics) {
        const chordIndices = {
            major: [0, 2, 4, 7],
            minor: [0, 2, 4, 6],
            seventh: [0, 2, 4, 7, 10],
            maj7: [0, 2, 4, 7, 11],
            sus4: [0, 3, 4]
        };

        const indices = chordIndices[type] || chordIndices.major;
        
        // Find closest harmonics to root and chord tones
        const frequencies = indices.map(idx => {
            // Find harmonic closest to expected frequency
            const expectedFreq = root * Math.pow(2, idx / 12);
            let closest = phiHarmonics[0];
            let minDiff = Math.abs(phiHarmonics[0] - expectedFreq);
            
            for (const harmonic of phiHarmonics) {
                const diff = Math.abs(harmonic - expectedFreq);
                if (diff < minDiff) {
                    minDiff = diff;
                    closest = harmonic;
                }
            }
            
            return closest;
        });

        return frequencies;
    }

    /**
     * Get recent chord history
     */
    getRecentChords() {
        return [...this.recentChords];
    }

    /**
     * Reset
     */
    reset() {
        this.recentChords = [];
    }
}

