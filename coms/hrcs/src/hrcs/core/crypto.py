"""HRCS authenticated-encryption helpers.

The active protocol uses ChaCha20-Poly1305 with a static/pre-shared derived
key. This provides confidentiality, integrity, and authenticity for holders of
the key. It does **not** provide forward secrecy because no ephemeral key
agreement or ratchet is implemented.
"""

from __future__ import annotations

import secrets
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class HRCSSecurity:
    def __init__(self, pre_shared_key: Optional[str] = None):
        self.key = self.derive_key(pre_shared_key) if pre_shared_key else secrets.token_bytes(32)
        self.cipher = ChaCha20Poly1305(self.key)

    @staticmethod
    def derive_key(password: str, salt: bytes = b"HRCS_NETWORK") -> bytes:
        if not password:
            raise ValueError("HRCS pre-shared key must not be empty")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=200_000,
        )
        return kdf.derive(password.encode("utf-8"))

    def encrypt_packet(self, packet_bytes: bytes) -> bytes:
        nonce = secrets.token_bytes(12)
        return nonce + self.cipher.encrypt(nonce, packet_bytes, None)

    def decrypt_packet(self, encrypted_data: bytes) -> bytes:
        if len(encrypted_data) < 12 + 16:
            raise ValueError("Encrypted data too short")
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        try:
            return self.cipher.decrypt(nonce, ciphertext, None)
        except Exception as exc:
            raise ValueError("HRCS packet authentication failed") from exc

    def generate_lorenz_seed(self) -> Tuple[float, float, float]:
        digest = hashes.Hash(hashes.SHA256())
        digest.update(self.key)
        seed_bytes = digest.finalize()

        def map_component(chunk: bytes) -> float:
            fraction = int.from_bytes(chunk, "big") / float(2**32 - 1)
            return fraction * 30.0 - 15.0

        return (
            map_component(seed_bytes[0:4]),
            map_component(seed_bytes[4:8]),
            map_component(seed_bytes[8:12]),
        )
