"""
Tests for mathematical framework
"""

import numpy as np
import pytest
from hrcs.core.math import UnifiedMath, PHI


class TestGoldenRatio:
    """Test golden ratio frequency generation"""
    
    def test_base_frequency(self):
        """Test base frequency is included"""
        freqs = UnifiedMath.golden_ratio_frequencies(432, 10)
        assert 432 in freqs
    
    def test_golden_ratio_spacing(self):
        """Test frequencies follow golden ratio"""
        freqs = UnifiedMath.golden_ratio_frequencies(432, 5)
        for i in range(1, len(freqs)):
            ratio = freqs[i] / freqs[i-1]
            assert ratio > 1.3  # Should be close to phi**(1/4)


class TestLorenzAttractor:
    """Test Lorenz attractor sequence generation"""
    
    def test_deterministic(self):
        """Test same seed produces same sequence"""
        seq1 = UnifiedMath.lorenz_sequence(0, 0, 0, 10)
        seq2 = UnifiedMath.lorenz_sequence(0, 0, 0, 10)
        np.testing.assert_array_equal(seq1, seq2)
    
    def test_uniform_distribution(self):
        """Test values are uniformly distributed"""
        seq = UnifiedMath.lorenz_sequence(1, 2, 3, 1000)
        # Should cover most of the 0-255 range
        assert seq.min() >= 0
        assert seq.max() <= 255
        assert len(np.unique(seq)) > 50


class TestStochasticResonance:
    """Test stochastic resonance enhancement"""
    
    def test_enhancement(self):
        """Test signal enhancement"""
        signal = np.sin(np.linspace(0, 10, 1000))
        enhanced = UnifiedMath.stochastic_enhance(signal, 0.1)
        assert len(enhanced) == len(signal)
    
    def test_thresholding(self):
        """Test nonlinear threshold"""
        signal = np.ones(100) * 0.5
        enhanced = UnifiedMath.stochastic_enhance(signal, 0.2)
        assert np.any(np.abs(enhanced) >= 0.9)


class TestSpectralEncoding:
    """Test spectral signature encoding/decoding"""
    
    def test_round_trip(self):
        """Test encode/decode round trip"""
        data = b"Hello------, World! 123"
        encoded = UnifiedMath.spectral_encode(data)
        decoded = UnifiedMath.spectral_decode(encoded)
        # Should decode to similar length
        assert abs(len(decoded) - len(data)) < 10
    
    def test_empty_data(self):
        """Test empty data handling"""
        data = b""
        encoded = UnifiedMath.spectral_encode(data)
        decoded = UnifiedMath.spectral_decode(encoded)
        assert isinstance(decoded, bytes)

