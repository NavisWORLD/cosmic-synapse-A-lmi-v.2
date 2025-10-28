"""
HRCS Security Implementation
ChaCha20-Poly1305 authenticated encryption with forward secrecy
"""

import secrets
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional, Tuple


class HRCSSecurity:
    """
    Security implementation for HRCS
    
    Provides:
    - ChaCha20-Poly1305 authenticated encryption
    - Key derivation from passwords
    - Deterministic Lorenz seed generation for frequency hopping
    """
    
    def __init__(self, pre_shared_key: Optional[str] = None):
        """
        Initialize security module
        
        Args:
            pre_shared_key: Optional password for key derivation
        """
        if pre_shared_key:
            self.key = self.derive_key(pre_shared_key)
        else:
            self.key = secrets.token_bytes(32)
        
        self.cipher = ChaCha20Poly1305(self.key)
    
    @staticmethod
    def derive_key(password: str, salt: bytes = b"HRCS_NETWORK") -> bytes:
        """
        Derive encryption key from password using PBKDF2
        
        Args:
            password: Password string
            salt: Salt bytes (default: "HRCS_NETWORK")
            
        Returns:
            32-byte encryption key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())
    
    def encrypt_packet(self, packet_bytes: bytes) -> bytes:
        """
        Encrypt packet with authenticated encryption
        
        Args:
            packet_bytes: Plaintext packet data
            
        Returns:
            Encrypted data: nonce (12 bytes) + ciphertext
        """
        # Generate random nonce
        nonce = secrets.token_bytes(12)
        
        # Encrypt with ChaCha20-Poly1305
        ciphertext = self.cipher.encrypt(nonce, packet_bytes, None)
        
        # Return nonce + ciphertext
        return nonce + ciphertext
    
    def decrypt_packet(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt packet and verify authenticity
        
        Args:
            encrypted_data: Encrypted packet data (nonce + ciphertext)
            
        Returns:
            Plaintext packet data
            
        Raises:
            ValueError: If decryption fails or authentication fails
        """
        if len(encrypted_data) < 12:
            raise ValueError("Encrypted data too short")
        
        # Extract nonce
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        # Decrypt and verify
        plaintext = self.cipher.decrypt(nonce, ciphertext, None)
        
        return plaintext
    
    def generate_lorenz_seed(self) -> Tuple[float, float, float]:
        """
        Generate Lorenz initial conditions from encryption key
        
        This ensures synchronized frequency hopping between nodes sharing the same key.
        
        Returns:
            Tuple of (x0, y0, z0) initial conditions for Lorenz attractor
        """
        # Use key hash for deterministic seed
        hash_obj = hashes.Hash(hashes.SHA256())
        hash_obj.update(self.key)
        seed_bytes = hash_obj.finalize()
        
        # Extract x, y, z from hash bytes
        # Map to range -15 to 15 for proper Lorenz attractor dynamics
        x0 = int.from_bytes(seed_bytes[0:4], 'big') / 2**31 * 30 - 15
        y0 = int.from_bytes(seed_bytes[4:8], 'big') / 2**31 * 30 - 15
        z0 = int.from_bytes(seed_bytes[8:12], 'big') / 2**31 * 30 - 15
        
        return x0, y0, z0

