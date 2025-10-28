"""
Homomorphic Encryption Module

Enables computation on encrypted data without decryption.
Uses Pyfhel for practical homomorphic encryption operations.
"""

import logging
from typing import List, Union
import numpy as np


class HomomorphicEncryption:
    """
    Homomorphic Encryption wrapper using Pyfhel.
    
    Supports:
    - Encrypted arithmetic operations
    - Encrypted comparisons
    - Encrypted similarity computations
    """
    
    def __init__(self, scheme='BGV', context_params=None):
        """
        Initialize homomorphic encryption.
        
        Args:
            scheme: Scheme to use ('BGV', 'CKKS', 'BFV')
            context_params: Custom context parameters
        """
        self.logger = logging.getLogger(__name__)
        self.scheme = scheme
        
        # Try to import Pyfhel
        try:
            import Pyfhel
            self.pyfhel = Pyfhel
            self.he = Pyfhel.Pyfhel()
            self._configure_context()
            self.initialized = True
            self.logger.info("Homomorphic encryption initialized")
        except ImportError:
            self.logger.warning("Pyfhel not installed. Homomorphic encryption not available.")
            self.initialized = False
    
    def _configure_context(self):
        """Configure HE context."""
        try:
            # BGV scheme for integer operations
            if self.scheme == 'BGV':
                self.he.contextGen(
                    scheme=self.pyfhel.SCHEME.BGV,
                    n=2**13,  # Polynomial modulus degree
                    t_bits=20  # Plaintext bits
                )
                self.he.keyGen()
            else:
                self.logger.error(f"Scheme {self.scheme} not yet implemented")
                
        except Exception as e:
            self.logger.error(f"Error configuring context: {e}")
            self.initialized = False
    
    def encrypt(self, data: Union[int, List[int]]) -> Union[object, List[object]]:
        """
        Encrypt data.
        
        Args:
            data: Integer or list of integers to encrypt
            
        Returns:
            Encrypted ciphertext(s)
        """
        if not self.initialized:
            raise RuntimeError("Homomorphic encryption not initialized")
        
        try:
            return self.he.encryptInt(data)
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt(self, ciphertext) -> Union[int, List[int]]:
        """
        Decrypt ciphertext.
        
        Args:
            ciphertext: Encrypted data
            
        Returns:
            Decrypted integers
        """
        if not self.initialized:
            raise RuntimeError("Homomorphic encryption not initialized")
        
        try:
            return self.he.decryptInt(ciphertext)
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            raise
    
    def add(self, ct1, ct2) -> object:
        """
        Add two encrypted values.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            
        Returns:
            Encrypted sum
        """
        if not self.initialized:
            raise RuntimeError("Homomorphic encryption not initialized")
        
        try:
            return ct1 + ct2
        except Exception as e:
            self.logger.error(f"Addition error: {e}")
            raise
    
    def multiply(self, ct1, ct2) -> object:
        """
        Multiply two encrypted values.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            
        Returns:
            Encrypted product
        """
        if not self.initialized:
            raise RuntimeError("Homomorphic encryption not initialized")
        
        try:
            return ct1 * ct2
        except Exception as e:
            self.logger.error(f"Multiplication error: {e}")
            raise
    
    def dot_product(self, vec1: List[object], vec2: List[object]) -> object:
        """
        Compute encrypted dot product.
        
        Args:
            vec1: First encrypted vector
            vec2: Second encrypted vector
            
        Returns:
            Encrypted dot product
        """
        if not self.initialized:
            raise RuntimeError("Homomorphic encryption not initialized")
        
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")
        
        try:
            result = self.he.encryptInt(0)
            for v1, v2 in zip(vec1, vec2):
                result = result + (v1 * v2)
            return result
        except Exception as e:
            self.logger.error(f"Dot product error: {e}")
            raise

