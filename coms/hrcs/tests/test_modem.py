"""
Tests for modem implementations
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from hrcs.physical.acoustic import AcousticModem
from hrcs.physical.radio import RadioModem
from hrcs.physical.base import BaseModem


class TestAcousticModem:
    """Test acoustic modem functionality"""
    
    def test_modem_creation(self):
        """Test acoustic modem can be created"""
        modem = AcousticModem()
        assert modem is not None
        assert modem.band_name == "acoustic"
    
    def test_modulate_demodulate(self):
        """Test modulation and demodulation round trip"""
        modem = AcousticModem()
        data = b"test message"
        
        # Modulate
        signal = modem.modulate(data)
        assert len(signal) > 0
        
        # Demodulate (no actual audio hardware needed)
        result = modem.demodulate(signal)
        assert isinstance(result, bytes)
    
    def test_frequency_channels(self):
        """Test golden ratio frequency channels"""
        modem = AcousticModem()
        assert len(modem.channels) > 0
        assert all(f > 0 and f < 18000 for f in modem.channels)
    
    def test_availability_check(self):
        """Test hardware availability check"""
        modem = AcousticModem()
        # This will work even if sounddevice not installed
        available = modem.is_available()
        assert isinstance(available, bool)


class TestRadioModem:
    """Test radio modem functionality"""
    
    def test_modem_creation(self):
        """Test radio modem can be created"""
        modem = RadioModem()
        assert modem is not None
        assert modem.band_name == "radio"
    
    def test_channel_generation(self):
        """Test channel list generation"""
        modem = RadioModem()
        # Should have 256 channels
        assert len(modem.channels) == 256
    
    def test_hop_sequence(self):
        """Test frequency hop sequence generation"""
        modem = RadioModem()
        seed = 12345
        sequence = modem.generate_hop_sequence(seed)
        assert len(sequence) == 1000
        assert all(0 <= val < 256 for val in sequence)
    
    def test_availability_check(self):
        """Test hardware availability check"""
        modem = RadioModem()
        available = modem.is_available()
        assert isinstance(available, bool)


class TestBaseModem:
    """Test base modem interface"""
    
    def test_get_info(self):
        """Test modem info retrieval"""
        # Create concrete implementation
        modem = AcousticModem()
        info = modem.get_info()
        
        assert 'type' in info
        assert 'enabled' in info
        assert 'available' in info
        assert 'band' in info

