/**
 * InstrumentControls - UI for instrument volume and mute controls
 */

export class InstrumentControls {
    constructor(containerElement, instruments, onVolumeChange, onMuteToggle) {
        this.container = containerElement;
        this.instruments = instruments;
        this.onVolumeChange = onVolumeChange;
        this.onMuteToggle = onMuteToggle;
        this.controls = {};
        this.render();
    }

    /**
     * Render instrument controls
     */
    render() {
        const instrumentNames = {
            drums: '🥁 Drums',
            bass: '🎸 Bass',
            guitar: '🎸 Guitar',
            piano: '🎹 Piano',
            strings: '🎻 Strings',
            pads: '🎵 Pads'
        };

        Object.keys(this.instruments).forEach(instName => {
            const inst = this.instruments[instName];
            const controlDiv = document.createElement('div');
            controlDiv.className = 'instrument-control';
            controlDiv.innerHTML = `
                <div class="instrument-name">
                    ${instrumentNames[instName] || instName}
                    <button class="toggle-btn" data-instrument="${instName}" id="${instName}-toggle">
                        ${inst.isEnabled() ? 'ON' : 'OFF'}
                    </button>
                </div>
                <input 
                    type="range" 
                    class="volume-slider" 
                    min="0" 
                    max="100" 
                    value="${Math.round(inst.getVolume() * 100)}"
                    data-instrument="${instName}"
                    id="${instName}-vol"
                >
                <div class="volume-value" id="${instName}-volume-value">${Math.round(inst.getVolume() * 100)}%</div>
            `;

            this.container.appendChild(controlDiv);
            this.controls[instName] = {
                slider: controlDiv.querySelector('.volume-slider'),
                toggle: controlDiv.querySelector('.toggle-btn'),
                volumeValue: controlDiv.querySelector('.volume-value')
            };

            // Setup event listeners
            this.controls[instName].slider.addEventListener('input', (e) => {
                const value = parseInt(e.target.value);
                this.updateVolume(instName, value);
            });

            this.controls[instName].toggle.addEventListener('click', () => {
                this.toggleMute(instName);
            });
        });
    }

    /**
     * Update volume for instrument
     */
    updateVolume(instrumentName, value) {
        const volume = value / 100;
        this.instruments[instrumentName].setVolume(volume);
        this.controls[instrumentName].volumeValue.textContent = `${value}%`;
        
        if (this.onVolumeChange) {
            this.onVolumeChange(instrumentName, volume);
        }
    }

    /**
     * Toggle mute for instrument
     */
    toggleMute(instrumentName) {
        const inst = this.instruments[instrumentName];
        if (inst.isEnabled()) {
            inst.disable();
            this.controls[instrumentName].toggle.textContent = 'OFF';
            this.controls[instrumentName].toggle.classList.add('muted');
        } else {
            inst.enable();
            this.controls[instrumentName].toggle.textContent = 'ON';
            this.controls[instrumentName].toggle.classList.remove('muted');
        }

        if (this.onMuteToggle) {
            this.onMuteToggle(instrumentName, inst.isEnabled());
        }
    }

    /**
     * Update control display
     */
    updateDisplay() {
        Object.keys(this.instruments).forEach(instName => {
            const inst = this.instruments[instName];
            const control = this.controls[instName];
            if (control) {
                control.slider.value = Math.round(inst.getVolume() * 100);
                control.volumeValue.textContent = `${Math.round(inst.getVolume() * 100)}%`;
                control.toggle.textContent = inst.isEnabled() ? 'ON' : 'OFF';
                if (!inst.isEnabled()) {
                    control.toggle.classList.add('muted');
                } else {
                    control.toggle.classList.remove('muted');
                }
            }
        });
    }
}

