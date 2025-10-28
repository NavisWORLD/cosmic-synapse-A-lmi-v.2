"""
Key Management Service Client

Handles secure key storage, rotation, and retrieval.
Supports both local filesystem and remote KMS backends.
"""

import os
import json
import logging
from typing import Optional
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from .encryption import AESCipher, derive_key


class KeyManager:
    """
    Key Management Service for secure key storage and rotation.
    
    Supports:
    - Local filesystem storage
    - Key rotation policies
    - Master key encryption
    - Access control logging
    """
    
    def __init__(self, storage_path: str = "keys/", master_password: Optional[str] = None):
        """
        Initialize key manager.
        
        Args:
            storage_path: Directory for key storage
            master_password: Master password for encrypting keys
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.master_password = master_password
        
        # Initialize master key if needed
        self._initialize_master_key()
        
        self.logger.info(f"Key Manager initialized at {self.storage_path}")
    
    def _initialize_master_key(self):
        """Initialize or load master key."""
        master_key_file = self.storage_path / "master.key"
        
        if master_key_file.exists():
            # Load existing master key
            if self.master_password:
                master_key_data = master_key_file.read_bytes()
                # Decrypt with password
                _, salt, encrypted_key = master_key_data.split(b':')
                salt = bytes.fromhex(salt.decode())
                encrypted_key = bytes.fromhex(encrypted_key.decode())
                
                key, _ = derive_key(self.master_password.encode(), salt)
                cipher = AESCipher(key)
                self.master_cipher = AESCipher(cipher.decrypt_from_string(encrypted_key.decode()))
            else:
                raise ValueError("Master password required to unlock key manager")
        else:
            # Create new master key
            if self.master_password is None:
                # Generate random master key
                master_key = os.urandom(32)
                master_key_file.write_bytes(master_key)
            else:
                # Derive master key from password
                master_key, salt = derive_key(self.master_password.encode())
                # Store encrypted
                cipher = AESCipher(master_key)
                key, _ = derive_key(self.master_password.encode())
                encrypted = cipher.encrypt_to_string(master_key.hex())
                master_key_file.write_bytes(f"{salt.hex()}:{encrypted}".encode())
            
            self.master_cipher = AESCipher(master_key)
    
    def store_key(self, key_name: str, key: bytes, metadata: Optional[dict] = None) -> bool:
        """
        Store an encryption key securely.
        
        Args:
            key_name: Name identifier for the key
            key: Key bytes to store
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        try:
            # Encrypt key with master cipher
            encrypted_key = self.master_cipher.encrypt_to_string(key)
            
            # Create key file
            key_file = self.storage_path / f"{key_name}.key"
            
            # Store metadata and key
            key_data = {
                'key': encrypted_key,
                'metadata': metadata or {}
            }
            
            key_file.write_text(json.dumps(key_data))
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            self.logger.info(f"Stored key: {key_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing key {key_name}: {e}")
            return False
    
    def retrieve_key(self, key_name: str) -> Optional[bytes]:
        """
        Retrieve an encryption key.
        
        Args:
            key_name: Name identifier for the key
            
        Returns:
            Key bytes or None if not found
        """
        try:
            key_file = self.storage_path / f"{key_name}.key"
            
            if not key_file.exists():
                self.logger.warning(f"Key not found: {key_name}")
                return None
            
            # Load key data
            key_data = json.loads(key_file.read_text())
            
            # Decrypt key
            key = self.master_cipher.decrypt_from_string(key_data['key'])
            
            self.logger.info(f"Retrieved key: {key_name}")
            return key
            
        except Exception as e:
            self.logger.error(f"Error retrieving key {key_name}: {e}")
            return None
    
    def delete_key(self, key_name: str) -> bool:
        """
        Delete a stored key.
        
        Args:
            key_name: Name identifier for the key
            
        Returns:
            Success status
        """
        try:
            key_file = self.storage_path / f"{key_name}.key"
            if key_file.exists():
                key_file.unlink()
                self.logger.info(f"Deleted key: {key_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting key {key_name}: {e}")
            return False
    
    def list_keys(self) -> list:
        """
        List all stored keys.
        
        Returns:
            List of key names
        """
        keys = []
        for key_file in self.storage_path.glob("*.key"):
            if key_file.name != "master.key":
                keys.append(key_file.stem)
        return keys
    
    def rotate_key(self, key_name: str) -> bool:
        """
        Rotate (regenerate) a key.
        
        Args:
            key_name: Name identifier for the key
            
        Returns:
            Success status
        """
        try:
            # Get old key info
            key_file = self.storage_path / f"{key_name}.key"
            if not key_file.exists():
                self.logger.warning(f"Key not found for rotation: {key_name}")
                return False
            
            old_key_data = json.loads(key_file.read_text())
            
            # Generate new key
            new_key = os.urandom(32)
            
            # Store new key
            success = self.store_key(key_name, new_key, old_key_data.get('metadata'))
            
            if success:
                self.logger.info(f"Rotated key: {key_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error rotating key {key_name}: {e}")
            return False

