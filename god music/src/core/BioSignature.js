/**
 * BioSignature - Extract and track user's bio-frequency signature
 */

import { SpectralAnalyzer } from '../analysis/SpectralAnalyzer.js';
import { PitchDetector } from '../analysis/PitchDetector.js';
import { TempoDetector } from '../analysis/TempoDetector.js';
import { frequencyToNote } from '../utils/helpers.js';
import { HUMAN_VOICE_MIN, HUMAN_VOICE_MAX } from '../utils/constants.js';

export class BioSignature {
    constructor(audioEngine) {
        this.audioEngine = audioEngine;
        this.analyzer = new SpectralAnalyzer(audioEngine.getAnalyzer());
        this.pitchDetector = new PitchDetector(audioEngine.getContext().sampleRate);
        this.tempoDetector = new TempoDetector();
        
        this.signature = {
            fundamental: null,
            spectralCentroid: null,
            spectralSpread: null,
            energy: 0,
            tempo: 120,
            key: null,
            note: null,
            lastUpdate: null
        };
        
        this.listeners = [];
    }

    /**
     * Update signature from current audio
     */
    update() {
        if (!this.audioEngine.isInitialized) {
            return;
        }

        const timeData = this.analyzer.getFloatTimeDomainData();
        const freqData = this.analyzer.getFloatFrequencyData();
        const currentTime = this.audioEngine.getContext().currentTime;

        // Detect pitch
        const pitch = this.pitchDetector.detect(timeData);
        
        if (pitch && pitch >= HUMAN_VOICE_MIN && pitch <= HUMAN_VOICE_MAX) {
            this.signature.fundamental = pitch;
            this.signature.note = frequencyToNote(pitch);
            this.signature.key = this.signature.note[0]; // Extract note name
        }

        // Spectral analysis
        this.signature.spectralCentroid = this.analyzer.calculateSpectralCentroid(freqData);
        this.signature.spectralSpread = this.analyzer.calculateSpectralSpread(freqData);
        this.signature.energy = this.analyzer.calculateEnergy();

        // Tempo detection
        this.tempoDetector.detectBeat(timeData, currentTime);
        const detectedTempo = this.tempoDetector.getStableTempo();
        if (detectedTempo) {
            this.signature.tempo = Math.round(detectedTempo);
        }

        this.signature.lastUpdate = Date.now();

        // Notify listeners
        this.notifyListeners();
    }

    /**
     * Get current signature
     */
    getSignature() {
        return { ...this.signature };
    }

    /**
     * Add listener for signature changes
     */
    addListener(callback) {
        this.listeners.push(callback);
    }

    /**
     * Remove listener
     */
    removeListener(callback) {
        const index = this.listeners.indexOf(callback);
        if (index > -1) {
            this.listeners.splice(index, 1);
        }
    }

    /**
     * Notify all listeners
     */
    notifyListeners() {
        const signature = this.getSignature();
        this.listeners.forEach(listener => {
            try {
                listener(signature);
            } catch (error) {
                console.error('Error in bio signature listener:', error);
            }
        });
    }
}

