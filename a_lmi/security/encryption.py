"""
Core Encryption Module

AES-256-GCM encryption for data at rest and in transit.
Implements secure key management and encryption/decryption utilities.
"""

import os
import base64
import logging
from typing import Union, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend


class AESCipher:
    """
    AES-256-GCM encryption for secure data storage.
    
    Uses:
    - AES-256-GCM (authenticated encryption)
    - PBKDF2 for key derivation
    - Random nonces for each encryption
    """
    
    def __init__(self, key: bytes = None):
        """
        Initialize AES cipher.
        
        Args:
            key: Encryption key (32 bytes for AES-256). If None, generates random key.
        """
        self.logger = logging.getLogger(__name__)
        
        if key is None:
            # Generate random key
            key = os.urandom(32)
        
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        
        self.key = key
        self.aesgcm = AESGCM(self.key)
        self.logger.info("AES cipher initialized")
    
    def encrypt(self, plaintext: Union[str, bytes]) -> Tuple[str, str]:
        """
        Encrypt plaintext data.
        
        Args:
            plaintext: Data to encrypt (string or bytes)
            
        Returns:
            Tuple of (encrypted_data_base64, nonce_base64)
        """
        try:
            # Convert to bytes if needed
            if isinstance(plaintext, str):
                plaintext = plaintext.encode('utf-8')
            
            # Generate random nonce
            nonce = os.urandom(12)  # 96 bits for GCM
            
            # Encrypt
            ciphertext = self.aesgcm.encrypt(nonce, plaintext, None)
            
            # Encode to base64 for storage
            ciphertext_b64 = base64.b64encode(ciphertext).decode('utf-8')
            nonce_b64 = base64.b64encode(nonce).decode('utf-8')
            
            return ciphertext_b64, nonce_b64
            
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt(self, ciphertext_b64: str, nonce_b64: str) -> bytes:
        """
        Decrypt ciphertext data.
        
        Args:
            ciphertext_b64: Encrypted data (base64 encoded)
            nonce_b64: Nonce (base64 encoded)
            
        Returns:
            Decrypted bytes
        """
        try:
            # Decode from base64
            ciphertext = base64.b64decode(ciphertext_b64)
            nonce = base64.b64decode(nonce_b64)
            
            # Decrypt
            plaintext = self.aesgcm.decrypt(nonce, ciphertext, None)
            
            return plaintext
            
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            raise
    
    def encrypt_to_string(self, plaintext: Union[str, bytes]) -> str:
        """
        Encrypt and return as single base64 string (includes nonce).
        
        Args:
            plaintext: Data to encrypt
            
        Returns:
            Combined encrypted data + nonce as base64 string
        """
        ciphertext, nonce = self.encrypt(plaintext)
        
        # Combine nonce and ciphertext
        combined = f"{nonce}:{ciphertext}"
        return combined
    
    def decrypt_from_string(self, encrypted_data: str) -> bytes:
        """
        Decrypt from combined base64 string.
        
        Args:
            encrypted_data: Combined nonce:ciphertext base64 string
            
        Returns:
            Decrypted bytes
        """
        parts = encrypted_data.split(':', 1)
        if len(parts) != 2:
            raise ValueError("Invalid encrypted data format")
        
        nonce_b64, ciphertext_b64 = parts
        return self.decrypt(ciphertext_b64, nonce_b64)


def derive_key(password: Union[str, bytes], salt: bytes = None, iterations: int = 100000) -> Tuple[bytes, bytes]:
    """
    Derive encryption key from password using PBKDF2.
    
    Args:
        password: Password string or bytes
        salt: Salt bytes (if None, generates random)
        iterations: PBKDF2 iterations
        
    Returns:
        Tuple of (derived_key, salt)
    """
    if isinstance(password, str):
        password = password.encode('utf-8')
    
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
        backend=default_backend()
    )
    
    key = kdf.derive(password)
    return key, salt


def encrypt_data(data: Union[str, bytes], key: bytes = None, password: Union[str, bytes] = None) -> Tuple[str, str]:
    """
    Convenience function to encrypt data.
    
    Args:
        data: Data to encrypt
        key: Encryption key (32 bytes)
        password: Password for key derivation (if key not provided)
        
    Returns:
        Tuple of (encrypted_data_base64, nonce_base64)
    """
    if key is None:
        if password is None:
            # Generate random key
            key = os.urandom(32)
        else:
            key, _ = derive_key(password)
    
    cipher = AESCipher(key)
    return cipher.encrypt(data)


def decrypt_data(encrypted_data: str, nonce: str, key: bytes = None, password: Union[str, bytes] = None) -> bytes:
    """
    Convenience function to decrypt data.
    
    Args:
        encrypted_data: Encrypted data (base64)
        nonce: Nonce (base64)
        key: Encryption key
        password: Password for key derivation
        
    Returns:
        Decrypted bytes
    """
    if key is None:
        if password is None:
            raise ValueError("Either key or password must be provided")
        key, _ = derive_key(password)
    
    cipher = AESCipher(key)
    return cipher.decrypt(encrypted_data, nonce)


def generate_key() -> bytes:
    """
    Generate a random encryption key.
    
    Returns:
        32-byte random key suitable for AES-256
    """
    return os.urandom(32)


def generate_key_hex() -> str:
    """
    Generate a random encryption key as hex string.
    
    Returns:
        Hex string of 32-byte key
    """
    return os.urandom(32).hex()

