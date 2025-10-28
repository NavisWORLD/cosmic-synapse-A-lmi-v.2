"""
Security Module

Complete security implementation including encryption, homomorphic encryption,
secure multi-party computation, and federated learning.
"""

from .encryption import AESCipher, encrypt_data, decrypt_data
from .key_manager import KeyManager
from .homomorphic import HomomorphicEncryption
from .smpc import SecureMultiPartyComputation
from .federated import FederatedLearning

__all__ = [
    'AESCipher',
    'encrypt_data',
    'decrypt_data',
    'KeyManager',
    'HomomorphicEncryption',
    'SecureMultiPartyComputation',
    'FederatedLearning'
]

