/**
 * PredictiveEngine - Main prediction system orchestrator
 */

import { PhraseTracker } from './PhraseTracker.js';
import { GrooveLock } from './GrooveLock.js';
import { ChordPredictor } from './ChordPredictor.js';
import { ANTICIPATION_TIME } from '../utils/constants.js';

export class PredictiveEngine {
    constructor() {
        this.phraseTracker = new PhraseTracker();
        this.grooveLock = new GrooveLock();
        this.chordPredictor = new ChordPredictor();
        
        this.isActive = false;
        this.predictedNextChord = null;
        this.predictedNextFill = null;
        this.currentChord = null;
        
        this.listeners = [];
    }

    /**
     * Start prediction engine
     */
    start(startTime) {
        this.isActive = true;
        this.phraseTracker.start(startTime);
        this.grooveLock.reset();
        this.chordPredictor.reset();
    }

    /**
     * Update prediction (call in loop)
     */
    update(currentTime, tempo, phiHarmonics) {
        if (!this.isActive) return;

        // Update phrase tracker
        this.phraseTracker.setTempo(tempo);
        this.phraseTracker.update(currentTime);

        // Record tempo for groove lock
        const currentBar = this.phraseTracker.currentBar;
        this.grooveLock.recordTempo(tempo, currentBar);

        // Use locked tempo if available
        const lockedTempo = this.grooveLock.getTempo();
        if (this.grooveLock.isGrooveLocked()) {
            this.phraseTracker.setTempo(lockedTempo);
        }

        // Calculate beats until phrase end
        const beatsUntilEnd = this.phraseTracker.getBeatsUntilPhraseEnd();

        // Predict changes 2 beats ahead
        if (beatsUntilEnd < 2 && beatsUntilEnd > 0.5) {
            if (!this.predictedNextChord && phiHarmonics) {
                this.predictedNextChord = this.chordPredictor.predictNextChord(
                    this.currentChord || { root: phiHarmonics[0] },
                    phiHarmonics
                );
                this.notifyListeners('prediction', {
                    type: 'chord',
                    chord: this.predictedNextChord
                });
            }
        }

        // Execute predictions
        const beatDuration = this.phraseTracker.getBeatDuration();
        if (beatsUntilEnd < ANTICIPATION_TIME / beatDuration) {
            if (this.predictedNextChord) {
                this.currentChord = this.predictedNextChord;
                this.predictedNextChord = null;
                this.notifyListeners('chordChange', { chord: this.currentChord });
            }
        }

        // Notify listeners of position updates
        this.notifyListeners('position', {
            bar: this.phraseTracker.currentBar,
            beat: this.phraseTracker.currentBeat,
            barInPhrase: this.phraseTracker.getBarInPhrase(),
            beatsUntilEnd,
            grooveLocked: this.grooveLock.isGrooveLocked()
        });
    }

    /**
     * Stop prediction
     */
    stop() {
        this.isActive = false;
    }

    /**
     * Add listener
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
     * Notify listeners
     */
    notifyListeners(eventType, data) {
        this.listeners.forEach(listener => {
            try {
                listener(eventType, data);
            } catch (error) {
                console.error('Error in prediction listener:', error);
            }
        });
    }

    /**
     * Get current state
     */
    getState() {
        return {
            isActive: this.isActive,
            currentBar: this.phraseTracker.currentBar,
            currentBeat: this.phraseTracker.currentBeat,
            barInPhrase: this.phraseTracker.getBarInPhrase(),
            beatsUntilEnd: this.phraseTracker.getBeatsUntilPhraseEnd(),
            grooveLocked: this.grooveLock.isGrooveLocked(),
            predictedNextChord: this.predictedNextChord,
            currentChord: this.currentChord
        };
    }
}

