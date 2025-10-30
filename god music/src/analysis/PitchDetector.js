/**
 * PitchDetector - YIN algorithm for accurate pitch detection
 */

export class PitchDetector {
    constructor(sampleRate, bufferSize = 4096) {
        this.sampleRate = sampleRate;
        this.bufferSize = bufferSize;
        this.threshold = 0.15;
    }

    /**
     * Detect fundamental frequency using YIN algorithm
     * @param {Float32Array} buffer - Audio buffer
     * @returns {number|null} - Frequency in Hz, or null if not detected
     */
    detect(buffer) {
        if (!buffer || buffer.length < this.bufferSize / 2) {
            return null;
        }

        const yinBuffer = this.differenceFunction(buffer);
        this.cumulativeMeanNormalizedDifference(yinBuffer);
        const tau = this.absoluteThreshold(yinBuffer);

        if (tau === -1) {
            return null;
        }

        const betterTau = this.parabolicInterpolation(tau, yinBuffer);
        const frequency = this.sampleRate / betterTau;

        // Validate frequency is in reasonable range
        if (frequency < 50 || frequency > 2000) {
            return null;
        }

        return frequency;
    }

    /**
     * Step 1: Difference function
     */
    differenceFunction(buffer) {
        const bufferLength = Math.min(buffer.length, this.bufferSize / 2);
        const yinBuffer = new Float32Array(bufferLength);

        for (let tau = 0; tau < bufferLength; tau++) {
            yinBuffer[tau] = 0;
            for (let j = 0; j < bufferLength; j++) {
                const delta = buffer[j] - buffer[j + tau];
                yinBuffer[tau] += delta * delta;
            }
        }

        return yinBuffer;
    }

    /**
     * Step 2: Cumulative mean normalized difference
     */
    cumulativeMeanNormalizedDifference(yinBuffer) {
        yinBuffer[0] = 1;
        let runningSum = 0;

        for (let tau = 1; tau < yinBuffer.length; tau++) {
            runningSum += yinBuffer[tau];
            if (runningSum !== 0) {
                yinBuffer[tau] *= tau / runningSum;
            }
        }
    }

    /**
     * Step 3: Absolute threshold
     */
    absoluteThreshold(yinBuffer) {
        for (let tau = 2; tau < yinBuffer.length; tau++) {
            if (yinBuffer[tau] < this.threshold) {
                // Find local minimum
                while (tau + 1 < yinBuffer.length && 
                       yinBuffer[tau + 1] < yinBuffer[tau]) {
                    tau++;
                }
                return tau;
            }
        }
        return -1;
    }

    /**
     * Step 4: Parabolic interpolation for better accuracy
     */
    parabolicInterpolation(tau, yinBuffer) {
        if (tau <= 0 || tau >= yinBuffer.length - 1) {
            return tau;
        }

        const s0 = yinBuffer[tau - 1];
        const s1 = yinBuffer[tau];
        const s2 = yinBuffer[tau + 1];

        if (2 * s1 - s2 - s0 === 0) {
            return tau;
        }

        return tau + (s2 - s0) / (2 * (2 * s1 - s2 - s0));
    }
}

