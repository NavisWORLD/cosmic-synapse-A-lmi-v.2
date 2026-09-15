"""A-LMI security public API with optional mechanisms loaded lazily.

AES-GCM helpers belong to the dependency-light core. Experimental federated,
homomorphic, and multi-party modules may require heavyweight optional
packages and are imported only when explicitly requested.
"""

from .encryption import (
    AESCipher,
    decrypt_data,
    decrypt_with_password,
    encrypt_data,
    encrypt_with_password,
    generate_key,
    generate_key_hex,
)

__all__ = [
    "AESCipher",
    "encrypt_data",
    "decrypt_data",
    "encrypt_with_password",
    "decrypt_with_password",
    "generate_key",
    "generate_key_hex",
    "KeyManager",
    "HomomorphicEncryption",
    "SecureMultiPartyComputation",
    "FederatedLearning",
]


def __getattr__(name):
    if name == "KeyManager":
        from .key_manager import KeyManager

        return KeyManager
    if name == "HomomorphicEncryption":
        from .homomorphic import HomomorphicEncryption

        return HomomorphicEncryption
    if name == "SecureMultiPartyComputation":
        from .smpc import SecureMultiPartyComputation

        return SecureMultiPartyComputation
    if name == "FederatedLearning":
        from .federated import FederatedLearning

        return FederatedLearning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
