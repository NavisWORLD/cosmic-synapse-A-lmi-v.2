/**
 * Main entry point - God Music Professional AI Music Conductor
 */

import { AudioEngine } from './core/AudioEngine.js';
import { BioSignature } from './core/BioSignature.js';
import { PhiHarmonics } from './core/PhiHarmonics.js';
import { PsiCalculator } from './core/PsiCalculator.js';
import { Mixer } from './audio/Mixer.js';
import { PredictiveEngine } from './prediction/PredictiveEngine.js';
import { Drums } from './instruments/Drums.js';
import { Bass } from './instruments/Bass.js';
import { Guitar } from './instruments/Guitar.js';
import { Piano } from './instruments/Piano.js';
import { Strings } from './instruments/Strings.js';
import { Pads } from './instruments/Pads.js';
import { Logger } from './ui/Logger.js';
import { Visualizer } from './ui/Visualizer.js';
import { InstrumentControls } from './ui/InstrumentControls.js';
import { DEFAULT_VOLUMES } from './utils/constants.js';

/**
 * Main application class
 */
class GodMusicConductor {
    constructor() {
        this.audioEngine = null;
        this.bioSignature = null;
        this.phiHarmonics = null;
        this.psiCalculator = null;
        this.mixer = null;
        this.predictiveEngine = null;
        this.instruments = {};
        this.logger = null;
        this.visualizer = null;
        
        this.isPlaying = false;
        this.microphoneStream = null;
        this.grooveLocked = false;
    }

    /**
     * Initialize application
     */
    async init() {
        try {
            // Initialize audio engine
            this.audioEngine = new AudioEngine();
            await this.audioEngine.initialize();
            
            // Initialize mixer
            this.mixer = new Mixer(
                this.audioEngine.getContext(),
                this.audioEngine.getCompressor(),
                this.audioEngine.getMasterGain()
            );

            // Initialize core systems
            this.bioSignature = new BioSignature(this.audioEngine);
            this.phiHarmonics = new PhiHarmonics();
            this.psiCalculator = new PsiCalculator();
            this.predictiveEngine = new PredictiveEngine();

            // Initialize instruments
            this.instruments.drums = new Drums(
                this.audioEngine.getContext(),
                this.mixer.getBus('drums')
            );
            this.instruments.bass = new Bass(
                this.audioEngine.getContext(),
                this.mixer.getBus('bass')
            );
            this.instruments.guitar = new Guitar(
                this.audioEngine.getContext(),
                this.mixer.getBus('guitar')
            );
            this.instruments.piano = new Piano(
                this.audioEngine.getContext(),
                this.mixer.getBus('piano')
            );
            this.instruments.strings = new Strings(
                this.audioEngine.getContext(),
                this.mixer.getBus('strings')
            );
            this.instruments.pads = new Pads(
                this.audioEngine.getContext(),
                this.mixer.getBus('pads')
            );

            // Set default volumes
            Object.keys(this.instruments).forEach(key => {
                if (DEFAULT_VOLUMES[key]) {
                    this.instruments[key].setVolume(DEFAULT_VOLUMES[key]);
                }
            });

            // Initialize UI
            this.logger = new Logger(document.getElementById('logPanel'));
            this.visualizer = new Visualizer(
                document.getElementById('spectrumCanvas'),
                document.getElementById('waveformCanvas'),
                this.audioEngine.getAnalyzer()
            );
            
            // Initialize instrument controls
            this.instrumentControls = new InstrumentControls(
                document.getElementById('instrumentGrid'),
                this.instruments,
                (instName, volume) => {
                    this.logger.info(`🔊 ${instName} volume: ${Math.round(volume * 100)}%`);
                },
                (instName, enabled) => {
                    this.logger.info(`${enabled ? '🔊' : '🔇'} ${instName} ${enabled ? 'enabled' : 'disabled'}`);
                }
            );

                    // Setup event listeners
            this.setupEventListeners();

            // Update status indicator
            this.updateStatus('System initialized successfully');

            this.logger.success('✅ System initialized successfully');
            this.logger.info('🎤 Microphone: Analysis Only ✓ (Never outputs to speakers)');

            // Setup bio signature listener
            this.bioSignature.addListener((signature) => {
                this.updateBioDisplay(signature);
                if (signature.fundamental) {
                    this.phiHarmonics.updateFundamental(signature.fundamental);
                }
            });

            // Setup prediction listener
            this.predictiveEngine.addListener((eventType, data) => {
                this.handlePredictionEvent(eventType, data);
            });

            this.logger.success('✅ System initialized successfully');
            this.logger.info('🎤 Microphone: Analysis Only ✓ (Never outputs to speakers)');

        } catch (error) {
            console.error('Initialization failed:', error);
            this.updateStatus(`❌ Initialization failed: ${error.message}`, 'error');
            if (this.logger) {
                this.logger.error(`❌ Initialization failed: ${error.message}`);
            }
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Calibrate button
        const calibrateBtn = document.getElementById('calibrateBtn');
        if (calibrateBtn) {
            calibrateBtn.addEventListener('click', () => {
                this.calibrate();
            });
        }

        // Start button
        const startBtn = document.getElementById('startBtn');
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                this.startBand();
            });
        }

        // Stop button
        const stopBtn = document.getElementById('stopBtn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                this.stopBand();
            });
        }

        // Test button
        const testBtn = document.getElementById('testBtn');
        if (testBtn) {
            testBtn.addEventListener('click', () => {
                this.testInstruments();
            });
        }

        // Genre buttons
        const genreSelector = document.getElementById('genreSelector');
        if (genreSelector) {
            genreSelector.addEventListener('click', (e) => {
                if (e.target.classList.contains('genre-btn')) {
                    const genre = e.target.textContent.toLowerCase().replace(/[^a-z]/g, '');
                    this.setGenre(genre);
                    // Update UI
                    document.querySelectorAll('.genre-btn').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    e.target.classList.add('active');
                }
            });
        }

        // Instrument controls (volume sliders and mute buttons)
        ['drums', 'bass', 'guitar', 'piano', 'strings', 'pads'].forEach(inst => {
            const slider = document.getElementById(`${inst}-vol`);
            const toggle = document.getElementById(`${inst}-toggle`);

            if (slider) {
                slider.addEventListener('input', (e) => {
                    const volume = parseInt(e.target.value) / 100;
                    if (this.instruments[inst]) {
                        this.instruments[inst].setVolume(volume);
                        this.logger.info(`🔊 ${inst} volume: ${Math.round(volume * 100)}%`);
                    }
                });
            }

            if (toggle) {
                toggle.addEventListener('click', () => {
                    const instObj = this.instruments[inst];
                    if (instObj) {
                        if (instObj.isEnabled()) {
                            instObj.disable();
                            toggle.textContent = 'OFF';
                            toggle.classList.add('muted');
                            this.logger.info(`🔇 ${inst} disabled`);
                        } else {
                            instObj.enable();
                            toggle.textContent = 'ON';
                            toggle.classList.remove('muted');
                            this.logger.info(`🔊 ${inst} enabled`);
                        }
                    }
                });
            }
        });
    }

    /**
     * Calibrate bio-signature
     */
    async calibrate() {
        this.logger.info('🎤 Starting calibration...');
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                } 
            });
            
            this.microphoneStream = stream;
            await this.audioEngine.connectMicrophone(stream);
            
            this.logger.warning('🗣️ Please speak, hum, or sing for 3 seconds...');
            this.updateStatus('Calibrating...');

            // Sample for 3 seconds
            setTimeout(() => {
                this.bioSignature.update();
                this.logger.success('✅ Calibration complete!');
                this.updateStatus('Calibration complete');
            }, 3000);

        } catch (error) {
            this.logger.error(`❌ Calibration failed: ${error.message}`);
        }
    }

    /**
     * Start band
     */
    async startBand() {
        if (this.isPlaying) return;

        if (!this.bioSignature.getSignature().fundamental) {
            this.logger.warning('⚠️ Please calibrate bio-signature first!');
            return;
        }

        try {
            await this.audioEngine.resume();
            
            this.isPlaying = true;
            const startTime = this.audioEngine.getContext().currentTime;
            
            this.predictiveEngine.start(startTime);
            this.visualizer.start();
            
            // Start main loop
            this.mainLoop();

            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;

            this.logger.success('🎼 BAND STARTED - Playing with you!');
            this.updateStatus('Band is playing...');

        } catch (error) {
            this.logger.error(`❌ Failed to start: ${error.message}`);
            this.updateStatus(`Failed to start: ${error.message}`, 'error');
        }
    }

    /**
     * Stop band
     */
    stopBand() {
        this.isPlaying = false;
        this.predictiveEngine.stop();
        this.visualizer.stop();

        // Stop all instruments
        Object.values(this.instruments).forEach(inst => inst.stop());

        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;

        this.logger.warning('⏹️ Band stopped');
        this.updateStatus('Band stopped');
    }

    /**
     * Main loop
     */
    mainLoop() {
        if (!this.isPlaying) return;

        const now = this.audioEngine.getContext().currentTime;

        // Update bio signature
        this.bioSignature.update();

        // Update prediction
        const signature = this.bioSignature.getSignature();
        const harmonics = this.phiHarmonics.getHarmonics();
        this.predictiveEngine.update(now, signature.tempo || 120, harmonics);

        // Calculate and display Psi
        this.updatePsi(signature);

        // Generate music based on prediction
        this.generateMusic(now, signature, harmonics);

        // Continue loop
        setTimeout(() => this.mainLoop(), 50);
    }

    /**
     * Update Psi calculation display
     */
    updatePsi(signature) {
        const activeVoices = Object.values(this.instruments)
            .filter(inst => inst.isEnabled()).length;

        const psiState = {
            energy: signature.energy || 0.5,
            tempo: signature.tempo || 120,
            activeVoices
        };

        const psi = this.psiCalculator.calculatePsi(psiState);

        // Update UI
        document.getElementById('psiValue').textContent = psi.total.toFixed(3);
        document.getElementById('psiFill').style.width = `${psi.total * 100}%`;
        document.getElementById('phiEnergy').textContent = psi.phiEnergy.toFixed(4);
        document.getElementById('chaosParam').textContent = psi.chaos.toFixed(3);
        document.getElementById('rhythmIntegral').textContent = psi.rhythmIntegral.toFixed(3);
        document.getElementById('connectivity').textContent = (psi.harmonicOmega / 10).toFixed(3);
    }

    /**
     * Generate music
     */
    generateMusic(currentTime, signature, harmonics) {
        const tempo = signature.tempo || 120;
        const beatDuration = 60 / tempo;
        const state = this.predictiveEngine.getState();

        // Play drums on beat
        if (state.currentBeat < 0.1) {
            const barInPhrase = state.barInPhrase;
            if (barInPhrase % 2 === 0) {
                this.instruments.drums.playPattern(currentTime, tempo);
            }
        }

        // Play bass following root
        if (harmonics && harmonics.length > 0 && state.currentBeat < 0.1) {
            const root = harmonics[0];
            this.instruments.bass.playRoot(root, tempo, currentTime);
        }

        // Play chords when they change
        if (state.currentChord && state.currentBeat < 0.1) {
            const chord = state.currentChord;
            if (chord.frequencies && chord.frequencies.length > 0) {
                this.instruments.piano.playChord(chord.frequencies, 0.3, currentTime);
                this.instruments.guitar.playChord(chord.frequencies.slice(0, 3), currentTime);
            }
        }
    }

    /**
     * Test all instruments
     */
    testInstruments() {
        this.logger.info('🧪 Testing instruments...');
        const now = this.audioEngine.getContext().currentTime;
        
        this.instruments.drums.playKick(now);
        setTimeout(() => this.instruments.drums.playSnare(now + 0.25), 250);
        setTimeout(() => this.instruments.drums.playHiHat(now + 0.5), 500);
        
        if (this.phiHarmonics.getHarmonics().length > 0) {
            const root = this.phiHarmonics.getHarmonic(0);
            setTimeout(() => this.instruments.bass.playNote(root, 0.5, now + 0.75), 750);
        }

        setTimeout(() => {
                this.logger.success('✅ Instrument test complete');
                this.updateStatus('Test complete');
        }, 1500);
    }

    /**
     * Update bio display
     */
    updateBioDisplay(signature) {
        if (signature.fundamental) {
            document.getElementById('bioFreq').textContent =
                `${signature.fundamental.toFixed(1)} Hz`;
        }
        if (signature.key) {
            document.getElementById('keyDetected').textContent = signature.key;
        }
        if (signature.tempo) {
            document.getElementById('tempoDetected').textContent = `${signature.tempo} BPM`;
        }
        if (signature.energy !== undefined) {
            document.getElementById('energyLevel').textContent = `${(signature.energy * 100).toFixed(1)}%`;
        }
        if (signature.spectralCentroid) {
            document.getElementById('spectralCentroid').textContent = `${signature.spectralCentroid.toFixed(1)} Hz`;
        }
    }

    /**
     * Set genre and adjust instrument parameters
     */
    setGenre(genre) {
        this.currentGenre = genre;
        this.logger.info(`🎵 Genre set to: ${genre.toUpperCase()}`);

        // Adjust instrument parameters based on genre
        switch (genre) {
            case 'jazz':
                this.adjustInstrumentsForJazz();
                break;
            case 'rock':
                this.adjustInstrumentsForRock();
                break;
            case 'pop':
                this.adjustInstrumentsForPop();
                break;
            case 'funk':
                this.adjustInstrumentsForFunk();
                break;
            case 'ambient':
                this.adjustInstrumentsForAmbient();
                break;
            case 'electronic':
                this.adjustInstrumentsForElectronic();
                break;
            default:
                this.logger.warning(`Unknown genre: ${genre}`);
        }
    }

    /**
     * Adjust instruments for Jazz
     */
    adjustInstrumentsForJazz() {
        if (this.instruments.drums) this.instruments.drums.setVolume(0.7);
        if (this.instruments.bass) this.instruments.bass.setVolume(0.8);
        if (this.instruments.piano) this.instruments.piano.setVolume(0.9);
        if (this.instruments.guitar) this.instruments.guitar.setVolume(0.3);
        if (this.instruments.strings) this.instruments.strings.setVolume(0.4);
        if (this.instruments.pads) this.instruments.pads.setVolume(0.2);
    }

    /**
     * Adjust instruments for Rock
     */
    adjustInstrumentsForRock() {
        if (this.instruments.drums) this.instruments.drums.setVolume(0.9);
        if (this.instruments.bass) this.instruments.bass.setVolume(0.8);
        if (this.instruments.guitar) this.instruments.guitar.setVolume(1.0);
        if (this.instruments.piano) this.instruments.piano.setVolume(0.2);
        if (this.instruments.strings) this.instruments.strings.setVolume(0.3);
        if (this.instruments.pads) this.instruments.pads.setVolume(0.1);
    }

    /**
     * Adjust instruments for Pop
     */
    adjustInstrumentsForPop() {
        if (this.instruments.drums) this.instruments.drums.setVolume(0.8);
        if (this.instruments.bass) this.instruments.bass.setVolume(0.7);
        if (this.instruments.piano) this.instruments.piano.setVolume(0.6);
        if (this.instruments.guitar) this.instruments.guitar.setVolume(0.5);
        if (this.instruments.strings) this.instruments.strings.setVolume(0.6);
        if (this.instruments.pads) this.instruments.pads.setVolume(0.7);
    }

    /**
     * Adjust instruments for Funk
     */
    adjustInstrumentsForFunk() {
        if (this.instruments.drums) this.instruments.drums.setVolume(0.9);
        if (this.instruments.bass) this.instruments.bass.setVolume(1.0);
        if (this.instruments.guitar) this.instruments.guitar.setVolume(0.7);
        if (this.instruments.piano) this.instruments.piano.setVolume(0.4);
        if (this.instruments.strings) this.instruments.strings.setVolume(0.2);
        if (this.instruments.pads) this.instruments.pads.setVolume(0.3);
    }

    /**
     * Adjust instruments for Ambient
     */
    adjustInstrumentsForAmbient() {
        if (this.instruments.drums) this.instruments.drums.setVolume(0.1);
        if (this.instruments.bass) this.instruments.bass.setVolume(0.3);
        if (this.instruments.piano) this.instruments.piano.setVolume(0.4);
        if (this.instruments.guitar) this.instruments.guitar.setVolume(0.2);
        if (this.instruments.strings) this.instruments.strings.setVolume(0.8);
        if (this.instruments.pads) this.instruments.pads.setVolume(1.0);
    }

    /**
     * Adjust instruments for Electronic
     */
    adjustInstrumentsForElectronic() {
        if (this.instruments.drums) this.instruments.drums.setVolume(0.8);
        if (this.instruments.bass) this.instruments.bass.setVolume(0.9);
        if (this.instruments.piano) this.instruments.piano.setVolume(0.3);
        if (this.instruments.guitar) this.instruments.guitar.setVolume(0.2);
        if (this.instruments.strings) this.instruments.strings.setVolume(0.4);
        if (this.instruments.pads) this.instruments.pads.setVolume(0.8);
    }

    /**
     * Handle prediction events
     */
    handlePredictionEvent(eventType, data) {
        if (eventType === 'prediction') {
            this.logger.info(`🎯 PREDICTION: ${data.type} incoming`);
        } else if (eventType === 'chordChange') {
            this.logger.info(`🎵 Chord changed`);
        } else if (eventType === 'position') {
            document.getElementById('activePredictions').textContent = data.predictions || 0;
            document.getElementById('phrasePosition').textContent = `${data.barInPhrase + 1} / ${4}`;
            document.getElementById('nextChangeIn').textContent = `${data.beatsUntilEnd?.toFixed(1) || '--'} beats`;

            if (data.grooveLocked && !this.grooveLocked) {
                this.grooveLocked = true;
                document.getElementById('grooveLock').textContent = 'LOCKED ✓';
                document.getElementById('grooveLock').style.color = 'var(--success)';
                this.logger.success('🔒 GROOVE LOCKED - Band locked to your rhythm!');
                this.updateStatus('Groove locked!');
            }
        }
    }
}

// Status update helper
GodMusicConductor.prototype.updateStatus = function(text, type = 'info') {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    if (statusText) {
        statusText.textContent = text;
        statusText.className = type;
    }
};

// Initialize on load
window.addEventListener('load', async () => {
    const conductor = new GodMusicConductor();
    conductor.updateStatus('Initializing...');
    await conductor.init();
    window.conductor = conductor; // For debugging
});


