"""
Light Token: Universal Multimodal Information Container

Implements the three-layer representation:
1. Semantic Core (Joint Embedding)
2. Perceptual Fingerprint (Perceptual Hash)
3. Spectral Signature (FFT of embedding)

Based on Graph Fourier Transform principles for vibrational information processing.
"""

import uuid
import hashlib
import numpy as np
from datetime import datetime, timezone
from scipy.fft import fft
from typing import Optional, Dict, Any
import json


class LightToken:
    """
    Universal multimodal information container implementing vibrational information theory.
    
    This is the atomic unit of knowledge in the A-LMI system - a unified representation
    that can encode any information type through spectral signature analysis.
    """
    
    def __init__(
        self,
        source_uri: str,
        modality: str,
        raw_data_ref: str,
        content_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new Light Token.
        
        Args:
            source_uri: Origin identifier (URL, mic stream ID, file path)
            modality: Data type ('text', 'image', 'audio', 'video', 'speech')
            raw_data_ref: Pointer to raw data in object store
            content_text: Textual content or transcription
            metadata: Additional context (ESC classification, resolution, author)
        """
        self.token_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.source_uri = source_uri
        self.modality = modality
        self.raw_data_ref = raw_data_ref
        self.content_text = content_text
        self.metadata = metadata or {}
        
        # Three-layer representation (to be populated)
        self.joint_embedding: Optional[np.ndarray] = None
        self.perceptual_hash: Optional[str] = None
        self.spectral_signature: Optional[np.ndarray] = None
        
        # Processed flags
        self._embedding_set = False
        self._hash_computed = False
        self._spectral_computed = False
    
    def set_embedding(self, embedding: np.ndarray) -> None:
        """
        Set the semantic embedding and automatically compute spectral signature.
        
        Args:
            embedding: 1536-dimensional semantic vector from joint-embedding model
        """
        if embedding.shape[0] != 1536:
            raise ValueError(f"Expected embedding dimension 1536, got {embedding.shape[0]}")
        
        self.joint_embedding = embedding.astype(np.float32)
        self._embedding_set = True
        
        # Automatically compute spectral signature
        self.spectral_signature = self._compute_spectral_signature(embedding)
        self._spectral_computed = True
    
    def set_perceptual_hash(self, phash: str) -> None:
        """
        Set the perceptual fingerprint for duplicate detection.
        
        Args:
            phash: Hex string of perceptual hash
        """
        self.perceptual_hash = phash
        self._hash_computed = True
    
    def _compute_spectral_signature(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute frequency-domain representation of semantic embedding.
        
        This implements the novel Graph Fourier Transform approach: treating
        semantic vectors as signals and analyzing their frequency characteristics.
        
        Args:
            embedding: Semantic embedding vector
            
        Returns:
            Complex-valued frequency spectrum
        """
        # Apply Discrete Fourier Transform
        spectral = fft(embedding)
        
        # Return spectral signature (complex-valued)
        return spectral
    
    def get_spectral_power(self) -> np.ndarray:
        """
        Get the power spectral density (magnitude) of the spectral signature.
        
        Returns:
            Real-valued array of frequency power
        """
        if not self._spectral_computed:
            raise ValueError("Spectral signature not computed. Call set_embedding first.")
        
        return np.abs(self.spectral_signature)
    
    def get_dominant_frequency(self) -> tuple[int, float]:
        """
        Get the dominant frequency component of the semantic content.
        
        This identifies the primary "frequency" at which this information
        naturally vibrates, enabling resonance-based retrieval.
        
        Returns:
            (frequency_index, power) tuple
        """
        power = self.get_spectral_power()
        dominant_idx = np.argmax(power)
        return int(dominant_idx), float(power[dominant_idx])
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Light Token to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        result = {
            "token_id": self.token_id,
            "timestamp": self.timestamp,
            "source_uri": self.source_uri,
            "modality": self.modality,
            "raw_data_ref": self.raw_data_ref,
            "content_text": self.content_text,
            "metadata": self.metadata,
            "perceptual_hash": self.perceptual_hash,
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if self.joint_embedding is not None:
            result["joint_embedding"] = self.joint_embedding.tolist()
        
        if self.spectral_signature is not None:
            # Store spectral signature as magnitude and phase
            result["spectral_signature_magnitude"] = np.abs(self.spectral_signature).tolist()
            result["spectral_signature_phase"] = np.angle(self.spectral_signature).tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LightToken':
        """
        Reconstruct Light Token from dictionary.
        
        Args:
            data: Dictionary from to_dict()
            
        Returns:
            Reconstructed LightToken instance
        """
        token = cls(
            source_uri=data["source_uri"],
            modality=data["modality"],
            raw_data_ref=data["raw_data_ref"],
            content_text=data.get("content_text"),
            metadata=data.get("metadata")
        )
        
        # Restore fields
        token.token_id = data["token_id"]
        token.timestamp = data["timestamp"]
        token.perceptual_hash = data.get("perceptual_hash")
        
        # Restore numpy arrays
        if "joint_embedding" in data:
            token.joint_embedding = np.array(data["joint_embedding"], dtype=np.float32)
            token._embedding_set = True
        
        if "spectral_signature_magnitude" in data:
            magnitude = np.array(data["spectral_signature_magnitude"])
            phase = np.array(data["spectral_signature_phase"])
            token.spectral_signature = magnitude * np.exp(1j * phase)
            token._spectral_computed = True
        
        return token
    
    def to_json(self) -> str:
        """
        Serialize Light Token to JSON string.
        
        Returns:
            JSON representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LightToken':
        """
        Deserialize Light Token from JSON string.
        
        Args:
            json_str: JSON representation
            
        Returns:
            Reconstructed LightToken instance
        """
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        """String representation of Light Token."""
        status = []
        if self._embedding_set:
            status.append("EMB")
        if self._hash_computed:
            status.append("HASH")
        if self._spectral_computed:
            status.append("SPEC")
        
        status_str = "/".join(status) if status else "RAW"
        return f"LightToken({self.modality}, {status_str}, ID={self.token_id[:8]}...)"


def spectral_similarity(token_a: LightToken, token_b: LightToken, method: str = "power_correlation") -> float:
    """
    Compute spectral similarity between two Light Tokens.
    
    This enables frequency-based retrieval - finding information that operates
    at compatible vibrational frequencies, even if semantically different.
    
    Args:
        token_a: First Light Token
        token_b: Second Light Token
        method: Similarity method ('power_correlation', 'cosine', 'euclidean')
        
    Returns:
        Similarity score [0, 1] where 1 is identical spectra
    """
    if not (token_a._spectral_computed and token_b._spectral_computed):
        raise ValueError("Both tokens must have computed spectral signatures")
    
    power_a = token_a.get_spectral_power()
    power_b = token_b.get_spectral_power()
    
    if method == "power_correlation":
        # Pearson correlation of power spectral densities
        return float(np.corrcoef(power_a, power_b)[0, 1])
    
    elif method == "cosine":
        # Cosine similarity of power spectra
        return float(np.dot(power_a, power_b) / (np.linalg.norm(power_a) * np.linalg.norm(power_b)))
    
    elif method == "euclidean":
        # Euclidean distance (normalized to [0, 1])
        distance = np.linalg.norm(power_a - power_b)
        max_distance = np.linalg.norm(power_a) + np.linalg.norm(power_b)
        return 1.0 - (distance / max_distance)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def resonance_match(query_token: LightToken, candidate_tokens: list[LightToken], threshold: float = 0.7) -> list[tuple[LightToken, float]]:
    """
    Find candidate tokens that resonate with the query token.
    
    Args:
        query_token: Token to find resonances for
        candidate_tokens: List of candidate tokens to check
        threshold: Minimum similarity to be considered a match
        
    Returns:
        List of (token, similarity_score) tuples, sorted by similarity
    """
    matches = []
    
    for candidate in candidate_tokens:
        similarity = spectral_similarity(query_token, candidate)
        if similarity >= threshold:
            matches.append((candidate, similarity))
    
    # Sort by similarity (descending)
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches

