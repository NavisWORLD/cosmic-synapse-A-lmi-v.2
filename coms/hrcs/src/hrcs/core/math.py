"""Experimental mathematical utilities used by HRCS prototypes.

Golden-ratio channel placement, Lorenz-derived index sequences, stochastic
noise experiments, and FFT codecs are preserved as software mechanisms. This
module does not claim those choices outperform conventional communications
methods without comparative reproducible measurements.
"""

from __future__ import annotations

import numpy as np

PHI = 1.618033988749895
SIGMA = 10.0
RHO = 28.0
BETA = 8 / 3


class UnifiedMath:
    PHI = PHI
    SIGMA = SIGMA
    RHO = RHO
    BETA = BETA

    @staticmethod
    def golden_ratio_frequencies(base_freq, num_channels):
        freqs = []
        for index in range(num_channels):
            frequency = base_freq * (PHI ** (index / 4))
            if frequency < 18000:
                freqs.append(frequency)
        return np.asarray(freqs, dtype=float)

    @staticmethod
    def lorenz_sequence(
        x0,
        y0,
        z0,
        length=None,
        dt=0.01,
        *,
        num_hops=None,
    ):
        if length is None:
            length = num_hops
        if length is None or int(length) < 0:
            raise ValueError("Lorenz sequence length must be a non-negative integer")
        x, y, z = float(x0), float(y0), float(z0)
        sequence = []
        for _ in range(int(length)):
            dx = SIGMA * (y - x) * dt
            dy = (x * (RHO - z) - y) * dt
            dz = (x * y - BETA * z) * dt
            x += dx
            y += dy
            z += dz
            value = int((x + 30) / 60 * 255) % 256
            sequence.append(value)
        return np.asarray(sequence, dtype=np.uint8)

    @staticmethod
    def stochastic_enhance(signal, noise_level=0.1, rng=None):
        """Apply the historical nonlinear noise transform as an experiment.

        A caller-provided RNG makes the experiment reproducible. No performance
        advantage is implied by this function alone.
        """

        signal = np.asarray(signal, dtype=float)
        generator = np.random.default_rng() if rng is None else rng
        noise = generator.normal(0.0, noise_level, len(signal))
        enhanced = signal + noise
        threshold = 0.3
        return np.where(
            enhanced > threshold,
            1.0,
            np.where(enhanced < -threshold, -1.0, enhanced),
        )

    @staticmethod
    def spectral_encode(data_bytes, num_channels=256):
        bit_array = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
        if len(bit_array) == 0:
            return np.asarray([], dtype=np.complex128)
        fft_size = 2 ** int(np.ceil(np.log2(len(bit_array))))
        padded = np.pad(bit_array, (0, fft_size - len(bit_array)))
        spectrum = np.fft.fft(padded.astype(float))
        weights = np.asarray([PHI ** (i / fft_size) for i in range(fft_size)])
        return spectrum * weights

    @staticmethod
    def spectral_decode(spectrum):
        spectrum = np.asarray(spectrum)
        if spectrum.size == 0:
            return b""
        fft_size = len(spectrum)
        weights = np.asarray([PHI ** (i / fft_size) for i in range(fft_size)])
        recovered_signal = np.fft.ifft(spectrum / weights).real
        return np.packbits((recovered_signal > 0.5).astype(np.uint8)).tobytes()
