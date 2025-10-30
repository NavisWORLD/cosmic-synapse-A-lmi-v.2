/**
 * PhraseTracker - Track position within musical phrases
 */

import { DEFAULT_PHRASE_LENGTH, DEFAULT_BEATS_PER_BAR } from '../utils/constants.js';

export class PhraseTracker {
    constructor() {
        this.phraseLength = DEFAULT_PHRASE_LENGTH;
        this.beatsPerBar = DEFAULT_BEATS_PER_BAR;
        this.startTime = 0;
        this.currentBar = 0;
        this.currentBeat = 0;
        this.tempo = 120;
    }

    /**
     * Start tracking
     */
    start(startTime) {
        this.startTime = startTime;
        this.currentBar = 0;
        this.currentBeat = 0;
    }

    /**
     * Update position based on current time
     */
    update(currentTime) {
        const elapsed = currentTime - this.startTime;
        const beatDuration = 60 / this.tempo;
        const totalBeats = elapsed / beatDuration;
        
        this.currentBeat = totalBeats % this.beatsPerBar;
        this.currentBar = Math.floor(totalBeats / this.beatsPerBar);
    }

    /**
     * Get current position in phrase
     */
    getBarInPhrase() {
        return this.currentBar % this.phraseLength;
    }

    /**
     * Get beats until phrase end
     */
    getBeatsUntilPhraseEnd() {
        const barInPhrase = this.getBarInPhrase();
        const beatsInPhrase = this.phraseLength * this.beatsPerBar;
        const beatsPassed = barInPhrase * this.beatsPerBar + this.currentBeat;
        return beatsInPhrase - beatsPassed;
    }

    /**
     * Set tempo
     */
    setTempo(tempo) {
        this.tempo = tempo;
    }

    /**
     * Get current beat duration
     */
    getBeatDuration() {
        return 60 / this.tempo;
    }
}

