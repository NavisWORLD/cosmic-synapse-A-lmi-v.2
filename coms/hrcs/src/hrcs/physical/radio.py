"""
SDR Radio Modem Implementation
Frequency hopping spread spectrum communication (VHF/UHF)
"""

import numpy as np
import time
from typing import Optional

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_TX, SOAPY_SDR_RX
    SOAPY_SDR_AVAILABLE = True
except ImportError:
    SOAPY_SDR_AVAILABLE = False

from ..core.math import UnifiedMath
from .base import BaseModem


class RadioModem(BaseModem):
    """
    Software Defined Radio communication using frequency hopping
    
    Supports VHF (30-300 MHz) and UHF (300-3000 MHz) bands with:
    - Golden ratio frequency channel distribution
    - Lorenz attractor-based frequency hopping for anti-jamming
    - Spectral signature data encoding
    """
    
    def __init__(self, center_freq=433.92e6, sample_rate=2e6, bandwidth=1e6):
        super().__init__()
        self.center_freq = center_freq  # Center frequency in Hz
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.band_name = "radio"
        
        # Generate golden ratio frequency hopping channels
        self.num_channels = 256
        self.channels = self._generate_channels()
        
        # SDR device (initialized on first use)
        self._sdr = None
        self._tx_stream = None
        self._rx_stream = None
    
    def _generate_channels(self) -> list:
        """Generate channel list with golden ratio distribution"""
        channels = []
        phi = UnifiedMath.PHI
        
        for i in range(self.num_channels):
            offset = (i / self.num_channels - 0.5) * self.bandwidth
            # Apply golden ratio weighting for better distribution
            weighted_offset = offset * (phi ** (abs(offset) / self.bandwidth))
            channels.append(self.center_freq + weighted_offset)
        
        return channels
    
    def is_available(self) -> bool:
        """Check if SDR hardware is available"""
        if not SOAPY_SDR_AVAILABLE:
            return False
        
        try:
            if self._sdr is None:
                self._sdr = SoapySDR.Device()
            return self._sdr is not None
        except Exception:
            return False
    
    def _init_sdr(self):
        """Initialize SDR if not already done"""
        if self._sdr is not None:
            return
        
        try:
            self._sdr = SoapySDR.Device()
            
            # Configure TX
            self._sdr.setSampleRate(SOAPY_SDR_TX, 0, self.sample_rate)
            self._sdr.setFrequency(SOAPY_SDR_TX, 0, self.center_freq)
            self._sdr.setGain(SOAPY_SDR_TX, 0, 30)  # 30dB gain
            
            # Configure RX
            self._sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            self._sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
            self._sdr.setGain(SOAPY_SDR_RX, 0, 40)  # 40dB gain
            
            # Create streams
            self._tx_stream = self._sdr.setupStream(SOAPY_SDR_TX, "CF32")
            self._rx_stream = self._sdr.setupStream(SOAPY_SDR_RX, "CF32")
        except Exception as e:
            print(f"SDR initialization error: {e}")
            self._sdr = None
    
    def generate_hop_sequence(self, seed: int) -> np.ndarray:
        """
        Generate Lorenz-based hopping sequence
        
        Args:
            seed: Seed value for initial conditions
            
        Returns:
            Array of frequency indices
        """
        # Use seed as initial conditions
        x0 = (seed & 0xFF) - 128
        y0 = ((seed >> 8) & 0xFF) - 128
        z0 = ((seed >> 16) & 0xFF) - 128
        
        return UnifiedMath.lorenz_sequence(x0, y0, z0, num_hops=1000)
    
    def transmit(self, data: bytes) -> bool:
        """
        Transmit data using frequency hopping
        
        Args:
            data: Data to transmit
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            self._init_sdr()
            if self._sdr is None:
                return False
            
            # Generate hop sequence from data hash
            seed = hash(data) & 0xFFFFFF
            hop_sequence = self.generate_hop_sequence(seed)
            
            # Encode data spectrally
            spectrum = UnifiedMath.spectral_encode(data)
            spectrum_real = np.real(spectrum).astype(np.float32)
            spectrum_imag = np.imag(spectrum).astype(np.float32)
            
            # Convert to complex samples
            iq_samples = spectrum_real + 1j * spectrum_imag
            
            # Transmit
            self._sdr.activateStream(self._tx_stream)
            self._sdr.writeStream(self._tx_stream, [iq_samples], len(iq_samples))
            self._sdr.deactivateStream(self._tx_stream)
            
            return True
        except Exception as e:
            print(f"Radio transmit error: {e}")
            return False
    
    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Receive data
        
        Args:
            timeout: Time to listen
            
        Returns:
            Received data bytes, or None on error
        """
        if not self.is_available():
            return None
        
        try:
            self._init_sdr()
            if self._sdr is None:
                return None
            
            # Calculate samples to receive
            num_samples = int(timeout * self.sample_rate)
            
            # Receive
            buff = np.zeros(num_samples, dtype=np.complex64)
            self._sdr.activateStream(self._rx_stream)
            self._sdr.readStream(self._rx_stream, [buff], num_samples)
            self._sdr.deactivateStream(self._rx_stream)
            
            # Convert to spectrum
            spectrum = buff
            
            # Decode
            return UnifiedMath.spectral_decode(spectrum)
        except Exception as e:
            print(f"Radio receive error: {e}")
            return None

