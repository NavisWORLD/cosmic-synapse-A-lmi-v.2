// Audio Processing Web Worker
// Handles FFT analysis and frequency processing off the main thread

self.addEventListener('message', async (e) => {
    const { type, data } = e.data;
    
    switch(type) {
        case 'init':
            await initializeAudio(data);
            break;
        case 'process':
            processAudioData(data);
            break;
        case 'close':
            closeAudio();
            break;
    }
});

let audioContext = null;
let analyser = null;
let microphone = null;
let audioData = null;
let stream = null;

async function initializeAudio(config) {
    try {
        // Create audio context in worker (if supported)
        // Note: Some browsers require AudioContext on main thread
        // We'll process raw audio data instead
        self.postMessage({
            type: 'initialized',
            success: true
        });
    } catch (error) {
        self.postMessage({
            type: 'error',
            error: error.message
        });
    }
}

function processAudioData(rawAudioData) {
    if (!rawAudioData || !rawAudioData.buffer) {
        return;
    }
    
    try {
        // Perform FFT analysis on the audio buffer
        const buffer = rawAudioData.buffer;
        const sampleRate = rawAudioData.sampleRate || 44100;
        
        // Simple FFT using existing data
        // In a full implementation, we'd use Web Audio API or WASM FFT
        const frequencyData = analyzeFrequency(buffer, sampleRate);
        
        // Generate phi harmonics
        const PHI = 1.618033988749895;
        const dominantFreq = findDominantFrequency(frequencyData, sampleRate);
        const harmonics = generatePhiHarmonics(dominantFreq, PHI);
        
        // Send results back to main thread
        self.postMessage({
            type: 'audioData',
            data: {
                frequency: dominantFreq,
                magnitude: frequencyData.maxMagnitude || 0,
                spectrum: frequencyData.spectrum,
                harmonics: harmonics,
                rawData: buffer
            }
        });
    } catch (error) {
        self.postMessage({
            type: 'error',
            error: error.message
        });
    }
}

function analyzeFrequency(buffer, sampleRate) {
    // Simple frequency analysis
    // In production, use proper FFT (WASM or Web Audio API)
    const spectrum = new Float32Array(buffer.length / 2);
    let maxMagnitude = 0;
    let dominantBin = 0;
    
    // Simplified magnitude calculation
    for (let i = 0; i < spectrum.length; i++) {
        const real = buffer[i * 2] || 0;
        const imag = buffer[i * 2 + 1] || 0;
        const magnitude = Math.sqrt(real * real + imag * imag);
        spectrum[i] = magnitude;
        
        if (magnitude > maxMagnitude) {
            maxMagnitude = magnitude;
            dominantBin = i;
        }
    }
    
    return {
        spectrum: spectrum,
        maxMagnitude: maxMagnitude,
        dominantBin: dominantBin
    };
}

function findDominantFrequency(frequencyData, sampleRate) {
    const nyquist = sampleRate / 2;
    const binSize = nyquist / frequencyData.spectrum.length;
    return frequencyData.dominantBin * binSize;
}

function generatePhiHarmonics(baseFreq, phi) {
    const harmonics = [];
    const numHarmonics = 5;
    
    for (let n = 0; n < numHarmonics; n++) {
        harmonics.push({
            frequency: baseFreq * Math.pow(phi, n),
            amplitude: 1 / (n + 1)
        });
    }
    
    return harmonics;
}

function closeAudio() {
    if (microphone) {
        microphone.disconnect();
        microphone = null;
    }
    if (analyser) {
        analyser = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

