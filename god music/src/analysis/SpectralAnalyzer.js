/**
 * SpectralAnalyzer - FFT-based spectral analysis
 */

export class SpectralAnalyzer {
    constructor(analyzerNode) {
        this.analyzer = analyzerNode;
        this.bufferLength = analyzerNode.frequencyBinCount;
        this.dataArray = new Uint8Array(this.bufferLength);
        this.floatDataArray = new Float32Array(this.bufferLength);
    }

    /**
     * Get frequency domain data
     */
    getFrequencyData() {
        this.analyzer.getByteFrequencyData(this.dataArray);
        return new Uint8Array(this.dataArray);
    }

    /**
     * Get frequency domain data as float
     */
    getFloatFrequencyData() {
        this.analyzer.getFloatFrequencyData(this.floatDataArray);
        return new Float32Array(this.floatDataArray);
    }

    /**
     * Get time domain data
     */
    getTimeDomainData() {
        const timeData = new Uint8Array(this.bufferLength);
        this.analyzer.getByteTimeDomainData(timeData);
        return timeData;
    }

    /**
     * Get time domain data as float
     */
    getFloatTimeDomainData() {
        const timeData = new Float32Array(this.bufferLength);
        this.analyzer.getFloatTimeDomainData(timeData);
        return timeData;
    }

    /**
     * Calculate spectral centroid (brightness)
     */
    calculateSpectralCentroid(frequencyData = null) {
        const freqData = frequencyData || this.getFloatFrequencyData();
        const sampleRate = this.analyzer.context.sampleRate;
        const nyquist = sampleRate / 2;
        const freqStep = nyquist / this.bufferLength;

        let weightedSum = 0;
        let totalMagnitude = 0;

        for (let i = 0; i < this.bufferLength; i++) {
            const magnitude = Math.pow(10, freqData[i] / 20); // Convert dB to linear
            const frequency = i * freqStep;
            weightedSum += frequency * magnitude;
            totalMagnitude += magnitude;
        }

        return totalMagnitude > 0 ? weightedSum / totalMagnitude : 0;
    }

    /**
     * Calculate spectral spread
     */
    calculateSpectralSpread(frequencyData = null) {
        const centroid = this.calculateSpectralCentroid(frequencyData);
        const freqData = frequencyData || this.getFloatFrequencyData();
        const sampleRate = this.analyzer.context.sampleRate;
        const nyquist = sampleRate / 2;
        const freqStep = nyquist / this.bufferLength;

        let variance = 0;
        let totalMagnitude = 0;

        for (let i = 0; i < this.bufferLength; i++) {
            const magnitude = Math.pow(10, freqData[i] / 20);
            const frequency = i * freqStep;
            const diff = frequency - centroid;
            variance += diff * diff * magnitude;
            totalMagnitude += magnitude;
        }

        return totalMagnitude > 0 ? Math.sqrt(variance / totalMagnitude) : 0;
    }

    /**
     * Calculate energy (RMS)
     */
    calculateEnergy() {
        const timeData = this.getFloatTimeDomainData();
        let sum = 0;
        for (let i = 0; i < timeData.length; i++) {
            sum += timeData[i] * timeData[i];
        }
        return Math.sqrt(sum / timeData.length);
    }

    /**
     * Find peak frequency in range
     */
    findPeakFrequency(minFreq = 80, maxFreq = 800, frequencyData = null) {
        const freqData = frequencyData || this.getFloatFrequencyData();
        const sampleRate = this.analyzer.context.sampleRate;
        const nyquist = sampleRate / 2;
        const freqStep = nyquist / this.bufferLength;

        const minIdx = Math.floor(minFreq / freqStep);
        const maxIdx = Math.floor(maxFreq / freqStep);

        let maxMagnitude = -Infinity;
        let peakIndex = 0;

        for (let i = minIdx; i < maxIdx && i < this.bufferLength; i++) {
            if (freqData[i] > maxMagnitude) {
                maxMagnitude = freqData[i];
                peakIndex = i;
            }
        }

        return peakIndex * freqStep;
    }
}

