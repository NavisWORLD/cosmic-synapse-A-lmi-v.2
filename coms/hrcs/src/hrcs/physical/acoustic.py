"""
Acoustic Modem Implementation
Golden ratio OFDM-based acoustic communication (20Hz-20kHz)
"""

import numpy as np
from typing import Optional

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

from ..core.math import UnifiedMath
from .base import BaseModem


# Default parameters
SAMPLE_RATE = 48000      # Audio sample rate (Hz)
BASE_FREQ = 432         # Base frequency (Hz)
CHANNELS = 32           # Number of OFDM subcarriers
SYMBOL_DURATION = 0.02  # Symbol duration (seconds)


class AcousticModem(BaseModem):
    """
    Acoustic communication using golden ratio OFDM
    
    Uses orthogonal frequency division multiplexing (OFDM) with golden ratio
    subcarrier spacing for robust acoustic communication in the 20Hz-20kHz range.
    """
    
    def __init__(self, sample_rate=SAMPLE_RATE, base_freq=BASE_FREQ, channels=CHANNELS):
        super().__init__()
        self.sample_rate = sample_rate
        self.base_freq = base_freq
        self.band_name = "acoustic"
        
        # Generate golden ratio frequency channels
        self.channels = UnifiedMath.golden_ratio_frequencies(base_freq, channels)
        
        # Keep channels within audible range
        self.channels = np.array([f for f in self.channels if f < 18000])
        self.num_channels = len(self.channels)
        
        self.symbol_duration = SYMBOL_DURATION
        self.samples_per_symbol = int(self.sample_rate * self.symbol_duration)
    
    def is_available(self) -> bool:
        """Check if audio hardware is available"""
        return SOUNDDEVICE_AVAILABLE
    
    def modulate(self, data_bytes: bytes) -> np.ndarray:
        """
        Modulate data onto acoustic carriers using BPSK OFDM
        
        Args:
            data_bytes: Data to modulate
            
        Returns:
            Array of audio samples
        """
        # Convert bytes to bits
        bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
        
        # Calculate symbols needed
        bits_per_symbol = self.num_channels
        num_symbols = int(np.ceil(len(bits) / bits_per_symbol))
        
        # Pad bits to multiple of bits_per_symbol
        total_bits = num_symbols * bits_per_symbol
        bits = np.pad(bits, (0, total_bits - len(bits)))
        
        # Generate signal
        signal = []
        
        for sym_idx in range(num_symbols):
            # Get bits for this symbol
            start = sym_idx * bits_per_symbol
            end = start + bits_per_symbol
            symbol_bits = bits[start:end]
            
            # Generate OFDM symbol
            t = np.linspace(0, self.symbol_duration, self.samples_per_symbol)
            symbol_signal = np.zeros(self.samples_per_symbol)
            
            for i, freq in enumerate(self.channels):
                if i < len(symbol_bits):
                    # BPSK modulation: bit 0 = phase 0, bit 1 = phase π
                    phase = np.pi if symbol_bits[i] else 0
                    carrier = np.sin(2 * np.pi * freq * t + phase)
                    symbol_signal += carrier
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(symbol_signal))
            if max_val > 0:
                symbol_signal = symbol_signal / max_val
            
            # Apply stochastic resonance enhancement
            symbol_signal = UnifiedMath.stochastic_enhance(symbol_signal, 0.05)
            
            signal.extend(symbol_signal)
        
        return np.array(signal)
    
    def demodulate(self, received_signal: np.ndarray) -> bytes:
        """
        Demodulate acoustic signal
        
        Args:
            received_signal: Array of received audio samples
            
        Returns:
            Demodulated data bytes
        """
        bits = []
        num_symbols = len(received_signal) // self.samples_per_symbol
        
        for sym_idx in range(num_symbols):
            # Extract symbol
            start = sym_idx * self.samples_per_symbol
            end = start + self.samples_per_symbol
            
            if end > len(received_signal):
                break
                
            symbol = received_signal[start:end]
            
            # Correlate with each carrier
            t = np.linspace(0, self.symbol_duration, self.samples_per_symbol)
            
            for freq in self.channels:
                ref_0 = np.sin(2 * np.pi * freq * t)
                ref_1 = np.sin(2 * np.pi * freq * t + np.pi)
                
                corr_0 = np.abs(np.sum(symbol * ref_0))
                corr_1 = np.abs(np.sum(symbol * ref_1))
                
                # Decision: choose bit based on stronger correlation
                bit = 1 if corr_1 > corr_0 else 0
                bits.append(bit)
        
        # Convert bits to bytes
        bit_array = np.array(bits)
        # Pad to multiple of 8
        remainder = len(bit_array) % 8
        if remainder > 0:
            bit_array = np.pad(bit_array, (0, 8 - remainder))
        
        return np.packbits(bit_array).tobytes()
    
    def transmit(self, data: bytes) -> bool:
        """
        Transmit data acoustically
        
        Args:
            data: Data to transmit
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            signal = self.modulate(data)
            sd.play(signal, self.sample_rate)
            sd.wait()
            return True
        except Exception as e:
            print(f"Acoustic transmit error: {e}")
            return False
    
    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Record and demodulate acoustic signal
        
        Args:
            timeout: Recording duration in seconds
            
        Returns:
            Demodulated data bytes, or None on error
        """
        if not self.is_available():
            return None
        
        try:
            recording = sd.rec(int(timeout * self.sample_rate),
                              samplerate=self.sample_rate,
                              channels=1)
            sd.wait()
            
            # Extract mono channel
            if len(recording.shape) > 1:
                recording = recording[:, 0]
            else:
                recording = recording.flatten()
            
            return self.demodulate(recording)
        except Exception as e:
            print(f"Acoustic receive error: {e}")
            return None

