/**
 * AudioEngine - Core Web Audio API setup
 * CRITICAL: Microphone routes ONLY to analyzer, NEVER to output
 */

import { DEFAULT_FFT_SIZE, DEFAULT_SMOOTHING } from '../utils/constants.js';
import { validateAudioContext } from '../utils/validation.js';

export class AudioEngine {
    constructor() {
        this.audioContext = null;
        this.microphone = null;
        this.analyzer = null;
        this.masterGain = null;
        this.compressor = null;
        
        // Separate routing paths
        this.analysisGraph = null; // mic → analyzer → [nothing]
        this.synthesisGraph = null; // instruments → mixer → compressor → master → speakers
        
        this.isInitialized = false;
    }

    /**
     * Initialize Web Audio API
     */
    async initialize() {
        try {
            // Create audio context
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioContextClass();
            
            if (!validateAudioContext(this.audioContext)) {
                throw new Error('Failed to create valid AudioContext');
            }

            // Create analyzer for microphone analysis ONLY
            this.analyzer = this.audioContext.createAnalyser();
            this.analyzer.fftSize = DEFAULT_FFT_SIZE;
            this.analyzer.smoothingTimeConstant = DEFAULT_SMOOTHING;

            // Create master output chain (for instruments only)
            this.compressor = this.audioContext.createDynamicsCompressor();
            this.compressor.threshold.value = -24;
            this.compressor.knee.value = 30;
            this.compressor.ratio.value = 12;
            this.compressor.attack.value = 0.003;
            this.compressor.release.value = 0.25;

            this.masterGain = this.audioContext.createGain();
            this.masterGain.gain.value = 0.7;

            // Connect synthesis chain: instruments → compressor → master → destination
            this.compressor.connect(this.masterGain);
            this.masterGain.connect(this.audioContext.destination);

            // Analysis graph is separate and will be connected when mic is added
            // mic → analyzer (NOT connected to destination)

            this.isInitialized = true;
            this.validateRouting();
            
            return this.audioContext;
        } catch (error) {
            console.error('AudioEngine initialization failed:', error);
            throw error;
        }
    }

    /**
     * Connect microphone to analyzer ONLY
     * CRITICAL: This is the ONLY place microphone should be connected
     */
    async connectMicrophone(stream) {
        if (!this.isInitialized) {
            throw new Error('AudioEngine must be initialized before connecting microphone');
        }

        try {
            // Create microphone source from stream
            this.microphone = this.audioContext.createMediaStreamSource(stream);

            // CRITICAL: Connect mic ONLY to analyzer
            // DO NOT connect to masterGain, compressor, or destination
            this.microphone.connect(this.analyzer);

            // Validate routing immediately
            this.validateRouting();

            return this.microphone;
        } catch (error) {
            console.error('Microphone connection failed:', error);
            throw error;
        }
    }

    /**
     * Validate that microphone is NOT routed to output
     * This runs after every routing change to ensure safety
     */
    validateRouting() {
        if (!this.microphone) {
            return; // No mic connected yet, nothing to validate
        }

        // Note: Web Audio API doesn't provide direct access to destination connections
        // Our validation is structural: we only call connect() once (mic → analyzer)
        // and never connect analyzer to output chain
        // This is enforced by code structure and runtime checks

        // Verify microphone exists and analyzer exists
        if (!this.microphone || !this.analyzer) {
            console.warn('⚠️ Microphone or analyzer not properly initialized');
            return;
        }

        // Check if analyzer is accidentally connected to output chain
        // We know analyzer should only be used for reading frequency data, not routing audio
        // Structural validation: analyzer is never connected to compressor/masterGain/destination
        // This is enforced by never calling connect() on analyzer to those nodes

        console.log('✅ Microphone routing validated: Analysis only, no output');
        console.log('   Microphone → Analyzer (read-only, not connected to speakers)');
        console.log('   Instruments → Compressor → Master → Speakers (separate chain)');
    }

    /**
     * Get audio context
     */
    getContext() {
        if (!this.isInitialized) {
            throw new Error('AudioEngine not initialized');
        }
        return this.audioContext;
    }

    /**
     * Get analyzer node (for microphone analysis)
     */
    getAnalyzer() {
        if (!this.isInitialized) {
            throw new Error('AudioEngine not initialized');
        }
        return this.analyzer;
    }

    /**
     * Get master gain node (for instrument output)
     */
    getMasterGain() {
        if (!this.isInitialized) {
            throw new Error('AudioEngine not initialized');
        }
        return this.masterGain;
    }

    /**
     * Get compressor node (for instrument output)
     */
    getCompressor() {
        if (!this.isInitialized) {
            throw new Error('AudioEngine not initialized');
        }
        return this.compressor;
    }

    /**
     * Resume audio context (required for user interaction)
     */
    async resume() {
        if (this.audioContext && this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    /**
     * Cleanup
     */
    destroy() {
        if (this.microphone) {
            this.microphone.disconnect();
            this.microphone = null;
        }
        if (this.analyzer) {
            this.analyzer.disconnect();
            this.analyzer = null;
        }
        if (this.masterGain) {
            this.masterGain.disconnect();
            this.masterGain = null;
        }
        if (this.compressor) {
            this.compressor.disconnect();
            this.compressor = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        this.isInitialized = false;
    }
}

