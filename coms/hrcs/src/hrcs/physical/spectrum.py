"""
Spectral Analysis Tools for HRCS
Frequency domain analysis and signal processing utilities
"""

import numpy as np
from typing import Tuple, List


def compute_power_spectral_density(signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density of signal
    
    Args:
        signal: Time domain signal
        sample_rate: Sample rate in Hz
        
    Returns:
        Tuple of (frequencies, power_density)
    """
    # FFT
    spectrum = np.fft.fft(signal)
    power = np.abs(spectrum) ** 2
    
    # Frequency axis
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    return freqs, power


def detect_peaks(frequencies: np.ndarray, power: np.ndarray, threshold: float = None) -> List[int]:
    """
    Detect frequency peaks in power spectrum
    
    Args:
        frequencies: Frequency values
        power: Power spectral density values
        threshold: Minimum power for peak (None = 3 std dev above mean)
        
    Returns:
        List of peak indices
    """
    if threshold is None:
        threshold = np.mean(power) + 3 * np.std(power)
    
    # Find local maxima
    peaks = []
    for i in range(1, len(power) - 1):
        if power[i] > power[i-1] and power[i] > power[i+1] and power[i] > threshold:
            peaks.append(i)
    
    return peaks


def calculate_snr(signal: np.ndarray, noise_floor: float = None) -> float:
    """
    Calculate signal-to-noise ratio
    
    Args:
        signal: Signal array
        noise_floor: Noise floor level (None = estimate from signal)
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    
    if noise_floor is None:
        # Estimate noise floor from lower percentile
        noise_floor = np.percentile(np.abs(signal) ** 2, 10)
    
    snr = signal_power / (noise_floor + 1e-10)
    snr_db = 10 * np.log10(snr)
    
    return snr_db


def golden_ratio_channel_allocation(base_freq: float, num_channels: int, bandwidth: float) -> np.ndarray:
    """
    Allocate channels using golden ratio spacing
    
    Args:
        base_freq: Base frequency in Hz
        num_channels: Number of channels
        bandwidth: Total bandwidth in Hz
        
    Returns:
        Array of channel frequencies
    """
    phi = 1.618033988749895
    channels = []
    
    for i in range(num_channels):
        # Golden ratio spacing within bandwidth
        position = (phi ** (i / num_channels) - 1) / (phi - 1)
        channels.append(base_freq + position * bandwidth)
    
    return np.array(channels)

