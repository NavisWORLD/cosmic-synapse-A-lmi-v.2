/**
 * PsiCalculator - Calculate ψ (psi) musical information density
 * Based on Unified Theory of Vibrational Information Architecture
 */

import { PHI, C_SOUND } from '../utils/constants.js';

export class PsiCalculator {
    constructor() {
        this.rhythmIntegral = 0;
        this.timeStep = 0.1;
    }

    /**
     * Calculate ψ (psi) - musical information density
     * ψ = [(φ × E_acoustic)/c² + λ(t) + ∫rhythm(t)dt + Ω_harmonic(t)] / ρ_ref
     */
    calculatePsi(state) {
        // Term 1: φ-scaled acoustic energy
        const phiEnergy = (PHI * state.energy) / (C_SOUND * C_SOUND);
        
        // Term 2: Chaos parameter (variation)
        const chaos = state.chaosLevel || (Math.random() * 0.3);
        
        // Term 3: Rhythmic integral (accumulated momentum)
        this.rhythmIntegral += (state.tempo / 120.0) * this.timeStep;
        this.rhythmIntegral *= 0.98; // Decay
        
        // Term 4: Harmonic connectivity
        const omega = (state.activeVoices / 6.0) * state.energy;
        
        // Combine and normalize
        const psi = phiEnergy + chaos + this.rhythmIntegral + omega;
        
        // Return components and total
        return {
            total: Math.max(0, Math.min(1, psi / 1.5)), // Clamp to [0, 1]
            phiEnergy,
            chaos,
            rhythmIntegral: this.rhythmIntegral,
            harmonicOmega: omega
        };
    }

    /**
     * Check if complexity should be added
     */
    shouldAddComplexity(psi, threshold = 0.3) {
        return psi < threshold;
    }

    /**
     * Check if complexity should be reduced
     */
    shouldReduceComplexity(psi, threshold = 0.7) {
        return psi > threshold;
    }

    /**
     * Reset calculator
     */
    reset() {
        this.rhythmIntegral = 0;
    }
}

