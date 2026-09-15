"""Optional SDR radio modem for HRCS experiments.

The active implementation provides deterministic, digest-derived transmit hop
planning and actually retunes the SDR on each TX chunk. Receive-side
synchronized hopping/framing is not implemented here, so this module does not
claim a verified frequency-hopping link or anti-jamming advantage.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_TX

    SOAPY_SDR_AVAILABLE = True
except ImportError:
    SoapySDR = None
    SOAPY_SDR_RX = None
    SOAPY_SDR_TX = None
    SOAPY_SDR_AVAILABLE = False

from ..core.math import PHI, UnifiedMath
from .base import BaseModem


def stable_data_seed(data: bytes, bits: int = 24) -> int:
    """Derive a stable cross-process integer seed from bytes."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    if bits <= 0 or bits > 256:
        raise ValueError("bits must be between 1 and 256")
    digest_value = int.from_bytes(hashlib.sha256(bytes(data)).digest(), "big")
    return digest_value & ((1 << bits) - 1)


class RadioModem(BaseModem):
    """Experimental SDR transport with deterministic TX hop planning."""

    frequency_hopping_status = "experimental_tx_only"

    def __init__(self, center_freq=433.92e6, sample_rate=2e6, bandwidth=1e6):
        super().__init__()
        self.center_freq = float(center_freq)
        self.sample_rate = float(sample_rate)
        self.bandwidth = float(bandwidth)
        self.band_name = "radio"
        self.num_channels = 256
        self.channels = self._generate_channels()
        self._sdr = None
        self._tx_stream = None
        self._rx_stream = None

    def _generate_channels(self) -> list[float]:
        channels: list[float] = []
        for index in range(self.num_channels):
            offset = (index / self.num_channels - 0.5) * self.bandwidth
            weighted_offset = offset * (PHI ** (abs(offset) / self.bandwidth))
            channels.append(self.center_freq + weighted_offset)
        return channels

    def is_available(self) -> bool:
        if not SOAPY_SDR_AVAILABLE:
            return False
        try:
            if self._sdr is None:
                self._sdr = SoapySDR.Device()
            return self._sdr is not None
        except Exception:
            return False

    def _init_sdr(self) -> None:
        if self._sdr is not None and self._tx_stream is not None and self._rx_stream is not None:
            return
        if not SOAPY_SDR_AVAILABLE:
            raise RuntimeError("SoapySDR is not installed")
        try:
            if self._sdr is None:
                self._sdr = SoapySDR.Device()
            self._sdr.setSampleRate(SOAPY_SDR_TX, 0, self.sample_rate)
            self._sdr.setFrequency(SOAPY_SDR_TX, 0, self.center_freq)
            self._sdr.setGain(SOAPY_SDR_TX, 0, 30)
            self._sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            self._sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
            self._sdr.setGain(SOAPY_SDR_RX, 0, 40)
            if self._tx_stream is None:
                self._tx_stream = self._sdr.setupStream(SOAPY_SDR_TX, "CF32")
            if self._rx_stream is None:
                self._rx_stream = self._sdr.setupStream(SOAPY_SDR_RX, "CF32")
        except Exception:
            self._sdr = None
            self._tx_stream = None
            self._rx_stream = None
            raise

    @staticmethod
    def _seed_initial_conditions(seed: int) -> tuple[float, float, float]:
        """Map three seed bytes to a bounded Lorenz initial-condition range."""

        components = [
            seed & 0xFF,
            (seed >> 8) & 0xFF,
            (seed >> 16) & 0xFF,
        ]
        return tuple((value / 255.0) * 30.0 - 15.0 for value in components)

    def generate_hop_sequence(self, seed: int, hop_count: int = 1000) -> np.ndarray:
        if hop_count <= 0:
            raise ValueError("hop_count must be positive")
        x0, y0, z0 = self._seed_initial_conditions(int(seed))
        return UnifiedMath.lorenz_sequence(x0, y0, z0, length=hop_count)

    def frequency_plan(self, data: bytes, hop_count: int) -> list[float]:
        """Return deterministic configured-channel frequencies for TX chunks."""

        sequence = self.generate_hop_sequence(stable_data_seed(data), hop_count=hop_count)
        if not self.channels:
            raise RuntimeError("No radio channels are configured")
        return [self.channels[int(index) % len(self.channels)] for index in sequence]

    def transmit(self, data: bytes) -> bool:
        """Transmit with deterministic per-chunk retuning when SDR hardware exists."""

        if not data or not self.is_available():
            return False
        try:
            self._init_sdr()
            spectrum = UnifiedMath.spectral_encode(data)
            iq_samples = np.asarray(spectrum, dtype=np.complex64)
            if iq_samples.size == 0:
                return False

            hop_count = min(32, int(iq_samples.size))
            plan = self.frequency_plan(data, hop_count=hop_count)
            chunks = np.array_split(iq_samples, hop_count)

            self._sdr.activateStream(self._tx_stream)
            try:
                for frequency, chunk in zip(plan, chunks):
                    if chunk.size == 0:
                        continue
                    self._sdr.setFrequency(SOAPY_SDR_TX, 0, float(frequency))
                    result = self._sdr.writeStream(
                        self._tx_stream,
                        [np.ascontiguousarray(chunk, dtype=np.complex64)],
                        int(chunk.size),
                    )
                    # SoapySDR returns a stream-result object on many backends.
                    returned = getattr(result, "ret", None)
                    if returned is not None and returned < 0:
                        return False
            finally:
                self._sdr.deactivateStream(self._tx_stream)
                self._sdr.setFrequency(SOAPY_SDR_TX, 0, self.center_freq)
            return True
        except Exception:
            return False

    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """Receive at the fixed center frequency.

        This path intentionally does not claim synchronized frequency hopping;
        a shared framing/clock/channel schedule must be implemented and tested
        before that property can be asserted for RX.
        """

        if timeout <= 0 or not self.is_available():
            return None
        try:
            self._init_sdr()
            self._sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
            num_samples = max(1, int(timeout * self.sample_rate))
            buffer = np.zeros(num_samples, dtype=np.complex64)
            self._sdr.activateStream(self._rx_stream)
            try:
                result = self._sdr.readStream(self._rx_stream, [buffer], num_samples)
                returned = getattr(result, "ret", num_samples)
                if returned is not None and returned <= 0:
                    return None
                spectrum = buffer[: int(returned)]
            finally:
                self._sdr.deactivateStream(self._rx_stream)
            return UnifiedMath.spectral_decode(spectrum)
        except Exception:
            return None
