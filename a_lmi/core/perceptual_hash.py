"""
Perceptual Hashing Module

Implements perceptual hashing for deduplication and similarity detection:
- Images: DCT-based perceptual hash (pHash)
- Audio: Audio fingerprinting using chromaprint
- Text: SimHash algorithm
"""

import hashlib
import imagehash
import numpy as np
import logging
from typing import Union, Tuple
import pyacoustid
import chromaprint
from PIL import Image
import io


class PerceptualHash:
    """
    Perceptual hashing for different modalities.
    
    Unlike cryptographic hashes, perceptual hashes are designed to detect
    similar content even when exact bytes differ.
    """
    
    def __init__(self):
        """Initialize perceptual hash module."""
        self.logger = logging.getLogger(__name__)
    
    def hash_image(
        self,
        image: Union[Image.Image, np.ndarray, bytes],
        hash_size: int = 16
    ) -> str:
        """
        Compute perceptual hash for image using DCT-based algorithm.
        
        This is resistant to scaling, rotation (to some extent), and format changes.
        
        Args:
            image: PIL Image, numpy array, or bytes
            hash_size: Size of hash (higher = more accurate, slower)
            
        Returns:
            Hex string of perceptual hash
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Compute perceptual hash using DCT
            phash = imagehash.phash(image, hash_size=hash_size)
            
            return str(phash)
            
        except Exception as e:
            self.logger.error(f"Error hashing image: {e}")
            return self._fallback_hash(image)
    
    def hash_audio(
        self,
        audio: Union[np.ndarray, bytes, str],
        sample_rate: int = 44100
    ) -> str:
        """
        Compute audio fingerprint using chromaprint/acoustid.
        
        Args:
            audio: Audio data as numpy array, bytes, or file path
            sample_rate: Sample rate
            
        Returns:
            Hex string of audio fingerprint
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio, bytes):
                import soundfile as sf
                audio, sr = sf.read(io.BytesIO(audio))
                sample_rate = sr
            elif isinstance(audio, str):
                import soundfile as sf
                audio, sample_rate = sf.read(audio)
            
            # Ensure audio is int16 (chromaprint requirement)
            if audio.dtype != np.int16:
                # Normalize and convert
                if audio.dtype == np.float32 or audio.dtype == np.float64:
                    audio = (audio * 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)
            
            # Compute fingerprint
            duration, fingerprint = chromaprint.fingerprint(audio, sample_rate)
            
            # Convert to hex string
            return fingerprint
            
        except Exception as e:
            self.logger.error(f"Error hashing audio: {e}")
            return self._fallback_hash(audio)
    
    def hash_text(self, text: str, num_bits: int = 64) -> str:
        """
        Compute SimHash for text content.
        
        SimHash is a locality-sensitive hashing algorithm that produces similar
        hashes for similar text, useful for near-duplicate detection.
        
        Args:
            text: Text string
            num_bits: Number of bits in hash (typically 64 or 128)
            
        Returns:
            Hex string of SimHash
        """
        try:
            # Tokenize text (simple whitespace tokenization)
            tokens = text.lower().split()
            
            # Create binary hash for each token
            v = np.zeros(num_bits, dtype=int)
            
            for token in tokens:
                # Hash the token
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                
                # Add bits to vector
                for i in range(num_bits):
                    if h & (1 << i):
                        v[i] += 1
                    else:
                        v[i] -= 1
            
            # Generate binary hash
            binary_hash = 0
            for i in range(num_bits):
                if v[i] > 0:
                    binary_hash |= 1 << i
            
            # Convert to hex
            return hex(binary_hash)[2:].zfill(num_bits // 4)
            
        except Exception as e:
            self.logger.error(f"Error hashing text: {e}")
            return self._fallback_hash(text)
    
    def _fallback_hash(self, data: Union[Image.Image, np.ndarray, bytes, str]) -> str:
        """
        Fallback hash using MD5 for any failure case.
        
        Args:
            data: Data to hash
            
        Returns:
            MD5 hash as hex string
        """
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        elif isinstance(data, (Image.Image, np.ndarray)):
            return hashlib.md5(np.array(data).tobytes()).hexdigest()
        else:
            return hashlib.md5(data).hexdigest()
    
    def similarity_image(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between two image perceptual hashes.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Similarity score [0, 1] where 1 is identical
        """
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            
            # Hamming distance
            distance = h1 - h2
            
            # Normalize to [0, 1]
            max_distance = len(hash1) * 4  # Each hex char represents 4 bits
            similarity = 1.0 - (distance / max_distance)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Error computing image similarity: {e}")
            # Binary comparison fallback
            return 1.0 if hash1 == hash2 else 0.0
    
    def similarity_text(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between two SimHashes.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Similarity score [0, 1] where 1 is identical
        """
        try:
            # Convert hex to binary
            bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
            bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
            
            # Compute Hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            
            # Normalize to [0, 1]
            similarity = 1.0 - (distance / len(bin1))
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Error computing text similarity: {e}")
            return 1.0 if hash1 == hash2 else 0.0
    
    def similarity_audio(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between two audio fingerprints.
        
        Args:
            hash1: First fingerprint
            hash2: Second fingerprint
            
        Returns:
            Similarity score [0, 1] where 1 is identical
        """
        try:
            # For chromaprint, we can decode and compare
            # This is a simplified version
            if hash1 == hash2:
                return 1.0
            
            # Compare fingerprint structures (simplified)
            # Real implementation would decode chromaprint format
            # For now, use string similarity as proxy
            matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
            similarity = matches / max(len(hash1), len(hash2))
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing audio similarity: {e}")
            return 1.0 if hash1 == hash2 else 0.0


# Global instance
_perceptual_hasher = None


def get_hasher() -> PerceptualHash:
    """
    Get global perceptual hasher instance.
    
    Returns:
        PerceptualHash instance
    """
    global _perceptual_hasher
    if _perceptual_hasher is None:
        _perceptual_hasher = PerceptualHash()
    return _perceptual_hasher

