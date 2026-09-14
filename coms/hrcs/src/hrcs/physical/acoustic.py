"""Experimental phi-spaced multicarrier acoustic modem.

The carriers are not assumed to be orthogonal, so the active implementation
uses a least-squares/matched basis decoder rather than labeling the scheme
OFDM. BPSK phase/sign is preserved during demodulation. A length prefix makes
synthetic round trips byte-exact even when the final carrier group is padded.

Optional historical stochastic-noise processing is disabled by default and is
not represented as a validated communications advantage.
"""

from __future__ import annotations

import math
import struct
from typing import Optional

import numpy as np

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

from ..core.math import UnifiedMath
from .base import BaseModem

SAMPLE_RATE = 48000
BASE_FREQ = 432
CHANNELS = 32
SYMBOL_DURATION = 0.02
FRAME_LENGTH_BYTES = 4


class AcousticModem(BaseModem):
    """Phi-spaced multicarrier BPSK modem with optional physical audio I/O."""

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        base_freq=BASE_FREQ,
        channels=CHANNELS,
        stochastic_noise_level: float = 0.0,
        stochastic_seed: int | None = None,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.base_freq = float(base_freq)
        self.band_name = "acoustic"
        self.channels = UnifiedMath.golden_ratio_frequencies(base_freq, channels)
        self.channels = np.asarray([f for f in self.channels if f < 18000], dtype=float)
        self.num_channels = len(self.channels)
        if self.num_channels == 0:
            raise ValueError("Acoustic modem requires at least one carrier")
        self.symbol_duration = SYMBOL_DURATION
        self.samples_per_symbol = int(self.sample_rate * self.symbol_duration)
        if self.samples_per_symbol <= self.num_channels:
            raise ValueError("Not enough samples per symbol for configured carrier basis")
        self.stochastic_noise_level = float(stochastic_noise_level)
        self._rng = np.random.default_rng(stochastic_seed)

        t = np.arange(self.samples_per_symbol, dtype=np.float64) / self.sample_rate
        self._carrier_basis = np.sin(2 * np.pi * self.channels[:, None] * t[None, :])
        # A is samples x carriers. pinv(A) maps a received symbol back to
        # signed carrier coefficients even though the carriers are non-orthogonal.
        self._decoder = np.linalg.pinv(self._carrier_basis.T, rcond=1e-10)

    def is_available(self) -> bool:
        return SOUNDDEVICE_AVAILABLE

    def modulate(self, data_bytes: bytes) -> np.ndarray:
        if not isinstance(data_bytes, bytes):
            raise TypeError("Acoustic modem payload must be bytes")
        framed = struct.pack("!I", len(data_bytes)) + data_bytes
        bits = np.unpackbits(np.frombuffer(framed, dtype=np.uint8))
        num_symbols = int(math.ceil(len(bits) / self.num_channels))
        bits = np.pad(bits, (0, num_symbols * self.num_channels - len(bits)))

        symbols = []
        for symbol_index in range(num_symbols):
            start = symbol_index * self.num_channels
            symbol_bits = bits[start : start + self.num_channels]
            coefficients = np.where(symbol_bits == 0, 1.0, -1.0)
            symbol = coefficients @ self._carrier_basis
            peak = float(np.max(np.abs(symbol)))
            if peak > 0:
                symbol = symbol / peak
            if self.stochastic_noise_level > 0:
                symbol = UnifiedMath.stochastic_enhance(
                    symbol,
                    self.stochastic_noise_level,
                    rng=self._rng,
                )
            symbols.append(symbol.astype(np.float64, copy=False))
        return np.concatenate(symbols) if symbols else np.asarray([], dtype=float)

    def demodulate(self, received_signal: np.ndarray) -> bytes:
        signal = np.asarray(received_signal, dtype=float).reshape(-1)
        num_symbols = len(signal) // self.samples_per_symbol
        if num_symbols == 0:
            raise ValueError("Acoustic signal does not contain a complete symbol")

        bits: list[int] = []
        for symbol_index in range(num_symbols):
            start = symbol_index * self.samples_per_symbol
            symbol = signal[start : start + self.samples_per_symbol]
            coefficients = self._decoder @ symbol
            bits.extend((coefficients < 0).astype(np.uint8).tolist())

        raw = np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()
        if len(raw) < FRAME_LENGTH_BYTES:
            raise ValueError("Acoustic frame is missing its length prefix")
        payload_length = struct.unpack("!I", raw[:FRAME_LENGTH_BYTES])[0]
        available = len(raw) - FRAME_LENGTH_BYTES
        if payload_length > available:
            raise ValueError(
                f"Acoustic frame declares {payload_length} bytes but only {available} are available"
            )
        return raw[FRAME_LENGTH_BYTES : FRAME_LENGTH_BYTES + payload_length]

    def transmit(self, data: bytes) -> bool:
        if not self.is_available():
            return False
        try:
            signal = self.modulate(data)
            sd.play(signal, self.sample_rate)
            sd.wait()
            return True
        except Exception:
            return False

    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        if not self.is_available():
            return None
        try:
            recording = sd.rec(
                int(timeout * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
            )
            sd.wait()
            return self.demodulate(np.asarray(recording).reshape(-1))
        except Exception:
            return None
