import test from 'node:test';
import assert from 'node:assert/strict';

import { SpectralAnalyzer } from '../src/analysis/SpectralAnalyzer.js';
import { ChordPredictor } from '../src/prediction/ChordPredictor.js';

class FakeAnalyser {
    constructor() {
        this.frequencyBinCount = 4;
        this.context = { sampleRate: 8000 };
    }

    getByteFrequencyData(target) {
        target.set([0, 64, 128, 255]);
    }

    getFloatFrequencyData(target) {
        target.set([-90, -30, 0, -60]);
    }

    getByteTimeDomainData(target) {
        target.set([0, 64, 128, 255]);
    }

    getFloatTimeDomainData(target) {
        target.set([-1, 1, -1, 1]);
    }
}

test('spectral analyzer reports deterministic RMS energy and peak bin', () => {
    const analyzer = new SpectralAnalyzer(new FakeAnalyser());
    assert.equal(analyzer.calculateEnergy(), 1);
    assert.equal(analyzer.findPeakFrequency(0, 4000, new Float32Array([-90, -30, 0, -60])), 2000);
});

test('spectral centroid and spread remain finite for deterministic fixture', () => {
    const analyzer = new SpectralAnalyzer(new FakeAnalyser());
    const data = new Float32Array([-90, -30, 0, -60]);
    const centroid = analyzer.calculateSpectralCentroid(data);
    const spread = analyzer.calculateSpectralSpread(data);
    assert.ok(Number.isFinite(centroid));
    assert.ok(Number.isFinite(spread));
    assert.ok(centroid > 1000 && centroid < 3000);
    assert.ok(spread >= 0);
});

test('chord predictor is deterministic and keeps bounded history', () => {
    const predictor = new ChordPredictor();
    const harmonics = [220, 330, 440, 550, 660, 880];
    const first = predictor.predictNextChord({ root: 440 }, harmonics);
    const secondPredictor = new ChordPredictor();
    const second = secondPredictor.predictNextChord({ root: 440 }, harmonics);
    assert.deepEqual(first, second);

    for (let index = 0; index < 12; index += 1) {
        predictor.predictNextChord({ root: 220 + index }, harmonics);
    }
    assert.equal(predictor.getRecentChords().length, predictor.maxHistory);
});
