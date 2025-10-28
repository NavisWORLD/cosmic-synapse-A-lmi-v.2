"""
Unified Mathematical Framework for HRCS
Implements golden ratio frequencies, Lorenz attractor, stochastic resonance, and spectral encoding
"""

import numpy as np

# Constants
PHI = 1.618033988749895  # Golden Ratio
SIGMA = 10.0  # Prandtl number for Lorenz
RHO = 28.0  # Rayleigh number for Lorenz
BETA = 8/3  # Geometric factor for Lorenz


class UnifiedMath:
    """
    Implements the 8D Unified Mathematical Framework
    """

    @staticmethod
    def golden_ratio_frequencies(base_freq, num_channels):
        """
        Generate golden ratio spaced frequencies
        
        Args:
            base_freq: Base frequency in Hz
            num_channels: Number of frequency channels to generate
            
        Returns:
            Array of frequencies in Hz
        """
        freqs = []
        for i in range(num_channels):
            freq = base_freq * (PHI ** (i / 4))
            if freq < 18000:  # Keep in audible range
                freqs.append(freq)
        return np.array(freqs)
    
    @staticmethod
    def lorenz_sequence(x0, y0, z0, length, dt=0.01):
        """
        Generate Lorenz attractor sequence for frequency hopping
        
        Args:
            x0, y0, z0: Initial conditions (shared secret between nodes)
            length: Number of hops to generate
            dt: Time step
            
        Returns:
            Array of frequency indices (0-255)
        """
        x, y, z = x0, y0, z0
        sequence = []
        
        for _ in range(length):
            # Lorenz equations
            dx = SIGMA * (y - x) * dt
            dy = (x * (RHO - z) - y) * dt
            dz = (x * y - BETA * z) * dt
            
            x += dx
            y += dy
            z += dz
            
            # Map to 0-255
            value = int((x + 30) / 60 * 255) % 256
            sequence.append(value)
        
        return np.array(sequence)
    
    @staticmethod
    def stochastic_enhance(signal, noise_level=0.1):
        """
        Apply stochastic resonance enhancement
        
        Stochastic resonance enhances signal detection by adding optimal noise levels.
        
        Args:
            signal: Input signal array
            noise_level: Optimal noise intensity (typically 0.1-0.3 of signal amplitude)
            
        Returns:
            Enhanced signal
        """
        # Add white noise at calculated optimal level
        noise = np.random.normal(0, noise_level, len(signal))
        enhanced = signal + noise
        
        # Nonlinear transformation (bistable potential)
        threshold = 0.3
        output = np.where(enhanced > threshold, 1.0,
                         np.where(enhanced < -threshold, -1.0, enhanced))
        
        return output
    
    @staticmethod
    def spectral_encode(data_bytes, num_channels=256):
        """
        Encode data as spectral signature
        
        Instead of traditional binary encoding, encodes information in the frequency spectrum.
        
        Args:
            data_bytes: Raw data to transmit
            num_channels: Number of frequency channels available (for compatibility)
            
        Returns:
            Complex spectral signature
        """
        # Convert bytes to bit array
        bit_array = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
        
        # Pad to match FFT size (power of 2)
        fft_size = 2 ** int(np.ceil(np.log2(len(bit_array))))
        padded = np.pad(bit_array, (0, fft_size - len(bit_array)))
        
        # Apply FFT to create spectral signature
        spectrum = np.fft.fft(padded.astype(float))
        
        # Apply golden ratio weighting for robustness
        weights = np.array([PHI ** (i / fft_size) for i in range(fft_size)])
        weighted_spectrum = spectrum * weights
        
        return weighted_spectrum
    
    @staticmethod
    def spectral_decode(spectrum):
        """
        Decode data from spectral signature
        
        Args:
            spectrum: Received spectral signature
            
        Returns:
            Original data bytes
        """
        # Remove golden ratio weighting
        fft_size = len(spectrum)
        weights = np.array([PHI ** (i / fft_size) for i in range(fft_size)])
        unweighted = spectrum / weights
        
        # Inverse FFT
        recovered_signal = np.fft.ifft(unweighted).real
        
        # Threshold to binary
        bit_array = (recovered_signal > 0.5).astype(np.uint8)
        
        # Convert back to bytes
        byte_array = np.packbits(bit_array)
        
        return byte_array.tobytes()

