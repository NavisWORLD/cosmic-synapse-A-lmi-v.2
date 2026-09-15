"""Perceptual hashing helpers for similarity and near-duplicate detection.

The legacy SimHash/fallback paths intentionally retain MD5 output compatibility.
Those digests are locality/deduplication identifiers only and are never used for
credentials, signatures, integrity verification, or other security decisions.
"""

import hashlib
import logging
from typing import Union

import imagehash
import numpy as np
from PIL import Image


class PerceptualHasher:
    """Generate perceptual hashes for supported content types."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def hash_image(self, image: Union[Image.Image, np.ndarray, bytes]) -> str:
        """Compute a perceptual hash for image content."""
        try:
            if isinstance(image, bytes):
                import io

                image = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            return str(imagehash.phash(image))
        except Exception as e:
            self.logger.error(f"Error hashing image: {e}")
            return self._fallback_hash(image)

    def hash_audio(self, audio: Union[np.ndarray, bytes], sample_rate: int = 44100) -> str:
        """Compute a compact spectral hash for audio content."""
        try:
            if isinstance(audio, bytes):
                audio = np.frombuffer(audio, dtype=np.int16)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size == 0:
                return self._fallback_hash(audio)

            # Deterministic spectral summary; this is a similarity feature, not
            # a claim about a physical frequency model.
            spectrum = np.abs(np.fft.rfft(audio))
            if spectrum.size == 0:
                return self._fallback_hash(audio)
            bins = min(64, spectrum.size)
            edges = np.linspace(0, spectrum.size, bins + 1, dtype=int)
            pooled = np.array(
                [spectrum[edges[i] : edges[i + 1]].mean() for i in range(bins)],
                dtype=np.float32,
            )
            threshold = float(np.median(pooled))
            bits = pooled > threshold
            binary = "".join("1" if bit else "0" for bit in bits)
            return hex(int(binary or "0", 2))[2:].zfill((bins + 3) // 4)
        except Exception as e:
            self.logger.error(f"Error hashing audio: {e}")
            return self._fallback_hash(audio)

    def hash_text(self, text: str, num_bits: int = 64) -> str:
        """
        Compute SimHash for text content.

        SimHash is a locality-sensitive hashing algorithm that produces similar
        hashes for similar text, useful for near-duplicate detection.
        """
        try:
            tokens = text.lower().split()
            v = np.zeros(num_bits, dtype=int)

            for token in tokens:
                # MD5 is retained solely to preserve the historical SimHash bit
                # distribution/output. It is not used for a security property.
                digest = hashlib.md5(token.encode(), usedforsecurity=False)
                h = int(digest.hexdigest(), 16)
                for i in range(num_bits):
                    if h & (1 << i):
                        v[i] += 1
                    else:
                        v[i] -= 1

            binary_hash = 0
            for i in range(num_bits):
                if v[i] > 0:
                    binary_hash |= 1 << i
            return hex(binary_hash)[2:].zfill(num_bits // 4)
        except Exception as e:
            self.logger.error(f"Error hashing text: {e}")
            return self._fallback_hash(text)

    def _fallback_hash(self, data: Union[Image.Image, np.ndarray, bytes, str]) -> str:
        """Return the legacy deterministic MD5 identifier for non-security use."""
        if isinstance(data, str):
            payload = data.encode()
        elif isinstance(data, (Image.Image, np.ndarray)):
            payload = np.array(data).tobytes()
        else:
            payload = data
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()

    def similarity_image(self, hash1: str, hash2: str) -> float:
        """Compute normalized similarity between two image perceptual hashes."""
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            max_distance = len(h1.hash.flatten())
            return 1.0 - ((h1 - h2) / max_distance)
        except Exception as e:
            self.logger.error(f"Error comparing image hashes: {e}")
            return 0.0

    def similarity_text(self, hash1: str, hash2: str, num_bits: int = 64) -> float:
        """Compute normalized Hamming similarity between two SimHashes."""
        try:
            h1 = int(hash1, 16)
            h2 = int(hash2, 16)
            distance = (h1 ^ h2).bit_count()
            return 1.0 - (distance / num_bits)
        except Exception as e:
            self.logger.error(f"Error comparing text hashes: {e}")
            return 0.0
