/**
 * Visualizer - Spectrum and waveform visualizations
 */

export class Visualizer {
    constructor(spectrumCanvas, waveformCanvas, analyzer) {
        this.spectrumCanvas = spectrumCanvas;
        this.waveformCanvas = waveformCanvas;
        this.analyzer = analyzer;
        this.isRunning = false;
        
        this.ctxSpectrum = spectrumCanvas.getContext('2d');
        this.ctxWaveform = waveformCanvas.getContext('2d');
    }

    /**
     * Start visualization loop
     */
    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.draw();
    }

    /**
     * Stop visualization
     */
    stop() {
        this.isRunning = false;
    }

    /**
     * Draw loop
     */
    draw() {
        if (!this.isRunning) return;

        this.drawSpectrum();
        this.drawWaveform();

        requestAnimationFrame(() => this.draw());
    }

    /**
     * Draw frequency spectrum
     */
    drawSpectrum() {
        const canvas = this.spectrumCanvas;
        const ctx = this.ctxSpectrum;
        
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        const bufferLength = this.analyzer.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyzer.getByteFrequencyData(dataArray);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height * 0.9;
            const hue = (i / bufferLength) * 270 + 180;
            ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
            ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
    }

    /**
     * Draw waveform
     */
    drawWaveform() {
        const canvas = this.waveformCanvas;
        const ctx = this.ctxWaveform;
        
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        const bufferLength = this.analyzer.fftSize;
        const dataArray = new Uint8Array(bufferLength);
        this.analyzer.getByteTimeDomainData(dataArray);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgb(102, 126, 234)';
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }

        ctx.stroke();
    }
}

