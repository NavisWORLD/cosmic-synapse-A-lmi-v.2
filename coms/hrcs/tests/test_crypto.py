"""
Tests for cryptographic functionality
"""

import pytest
from hrcs.core.crypto import HRCSSecurity


class TestHRCSSecurity:
    """Test security implementation"""
    
    def test_initialization_no_key(self):
        """Test initialization without key"""
        security = HRCSSecurity()
        assert security.key is not None
        assert len(security.key) == 32
    
    def test_initialization_with_key(self):
        """Test initialization with pre-shared key"""
        key = "test-password-123"
        security = HRCSSecurity(pre_shared_key=key)
        assert security.key is not None
        assert len(security.key) == 32
    
    def test_encrypt_decrypt(self):
        """Test encrypt/decrypt round trip"""
        security = HRCSSecurity()
        plaintext = b"Hello, secret message!"
        
        # Encrypt
        encrypted = security.encrypt_packet(plaintext)
        assert encrypted != plaintext
        assert len(encrypted) > len(plaintext)
        
        # Decrypt
        decrypted = security.decrypt_packet(encrypted)
        assert decrypted == plaintext
    
    def test_different_keys(self):
        """Test different keys produce different ciphertext"""
        security1 = HRCSSecurity(pre_shared_key="key1")
        security2 = HRCSSecurity(pre_shared_key="key2")
        
        plaintext = b"test message"
        
        enc1 = security1.encrypt_packet(plaintext)
        enc2 = security2.encrypt_packet(plaintext)
        
        assert enc1 != enc2
    
    def test_key_derivation_consistency(self):
        """Test key derivation is consistent"""
        key1 = HRCSSecurity.derive_key("password")
        key2 = HRCSSecurity.derive_key("password")
        
        assert key1 == key2
    
    def test_lorenz_seed_generation(self):
        """Test Lorenz seed generation from key"""
        security = HRCSSecurity(pre_shared_key="test")
        x0, y0, z0 = security.generate_lorenz_seed()
        
        # Should return three floats
        assert isinstance(x0, float)
        assert isinstance(y0, float)
        assert isinstance(z0, float)
        
        # Should be deterministic
        x1, y1, z1 = security.generate_lorenz_seed()
        assert x0 == x1
        assert y0 == y1
        assert z0 == z1

