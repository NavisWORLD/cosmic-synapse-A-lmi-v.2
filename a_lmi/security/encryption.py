"""Authenticated AES-256-GCM helpers for active A-LMI storage.

Password-based encryption uses a versioned self-contained envelope carrying
its PBKDF2 salt and parameters. Historical callers that use ``encrypt_data``
retain the two-field return shape; password mode prefixes the ciphertext with
its KDF salt so decryption can reproduce the same key.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Tuple, Union

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

PASSWORD_ENVELOPE_VERSION = 1
DEFAULT_PBKDF2_ITERATIONS = 600_000


class AESCipher:
    """AES-256-GCM authenticated encryption using a caller-owned 32-byte key."""

    def __init__(self, key: bytes | None = None):
        self.logger = logging.getLogger(__name__)
        key = os.urandom(32) if key is None else key
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self.key = key
        self.aesgcm = AESGCM(self.key)

    def encrypt(self, plaintext: Union[str, bytes]) -> Tuple[str, str]:
        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, None)
        return (
            base64.b64encode(ciphertext).decode("ascii"),
            base64.b64encode(nonce).decode("ascii"),
        )

    def decrypt(self, ciphertext_b64: str, nonce_b64: str) -> bytes:
        ciphertext = base64.b64decode(ciphertext_b64, validate=True)
        nonce = base64.b64decode(nonce_b64, validate=True)
        if len(nonce) != 12:
            raise ValueError("AES-GCM nonce must be 12 bytes")
        return self.aesgcm.decrypt(nonce, ciphertext, None)

    def encrypt_to_string(self, plaintext: Union[str, bytes]) -> str:
        ciphertext, nonce = self.encrypt(plaintext)
        return f"{nonce}:{ciphertext}"

    def decrypt_from_string(self, encrypted_data: str) -> bytes:
        parts = encrypted_data.split(":", 1)
        if len(parts) != 2:
            raise ValueError("Invalid encrypted data format")
        nonce_b64, ciphertext_b64 = parts
        return self.decrypt(ciphertext_b64, nonce_b64)


def derive_key(
    password: Union[str, bytes],
    salt: bytes | None = None,
    iterations: int = DEFAULT_PBKDF2_ITERATIONS,
) -> Tuple[bytes, bytes]:
    """Derive an AES-256 key with PBKDF2-HMAC-SHA256 and return its salt."""

    if iterations < 1:
        raise ValueError("PBKDF2 iterations must be positive")
    password_bytes = password.encode("utf-8") if isinstance(password, str) else password
    if not isinstance(password_bytes, bytes) or not password_bytes:
        raise ValueError("Password must not be empty")
    actual_salt = os.urandom(16) if salt is None else salt
    if len(actual_salt) < 16:
        raise ValueError("PBKDF2 salt must be at least 16 bytes")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=actual_salt,
        iterations=iterations,
    )
    return kdf.derive(password_bytes), actual_salt


def encrypt_with_password(
    data: Union[str, bytes],
    password: Union[str, bytes],
    *,
    iterations: int = DEFAULT_PBKDF2_ITERATIONS,
) -> str:
    """Return a portable JSON password-encryption envelope."""

    key, salt = derive_key(password, iterations=iterations)
    ciphertext, nonce = AESCipher(key).encrypt(data)
    envelope = {
        "version": PASSWORD_ENVELOPE_VERSION,
        "cipher": "AES-256-GCM",
        "kdf": "PBKDF2-HMAC-SHA256",
        "iterations": iterations,
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce": nonce,
        "ciphertext": ciphertext,
    }
    return json.dumps(envelope, separators=(",", ":"), sort_keys=True)


def decrypt_with_password(
    envelope_json: str, password: Union[str, bytes]
) -> bytes:
    """Decrypt a version-1 password envelope using its stored KDF material."""

    try:
        envelope = json.loads(envelope_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid password-encryption envelope") from exc

    if envelope.get("version") != PASSWORD_ENVELOPE_VERSION:
        raise ValueError(f"Unsupported encryption envelope version: {envelope.get('version')}")
    if envelope.get("cipher") != "AES-256-GCM":
        raise ValueError("Unsupported encryption cipher")
    if envelope.get("kdf") != "PBKDF2-HMAC-SHA256":
        raise ValueError("Unsupported encryption KDF")

    try:
        salt = base64.b64decode(envelope["salt"], validate=True)
        iterations = int(envelope["iterations"])
        nonce = envelope["nonce"]
        ciphertext = envelope["ciphertext"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Incomplete password-encryption envelope") from exc

    key, _ = derive_key(password, salt=salt, iterations=iterations)
    return AESCipher(key).decrypt(ciphertext, nonce)


def encrypt_data(
    data: Union[str, bytes],
    key: bytes | None = None,
    password: Union[str, bytes, None] = None,
) -> Tuple[str, str]:
    """Compatibility API for key- or password-based encryption.

    Password mode preserves the historical two-value return shape while
    embedding the salt into the first value as ``pbe1:<salt>:<ciphertext>``.
    """

    if key is not None and password is not None:
        raise ValueError("Provide either key or password, not both")
    if key is None and password is None:
        key = os.urandom(32)

    if password is not None:
        key, salt = derive_key(password)
        ciphertext, nonce = AESCipher(key).encrypt(data)
        salt_b64 = base64.b64encode(salt).decode("ascii")
        return f"pbe1:{salt_b64}:{ciphertext}", nonce

    return AESCipher(key).encrypt(data)


def decrypt_data(
    encrypted_data: str,
    nonce: str,
    key: bytes | None = None,
    password: Union[str, bytes, None] = None,
) -> bytes:
    """Compatibility decryption API with repaired password salt handling."""

    if key is not None and password is not None:
        raise ValueError("Provide either key or password, not both")

    if password is not None:
        parts = encrypted_data.split(":", 2)
        if len(parts) != 3 or parts[0] != "pbe1":
            raise ValueError(
                "Password ciphertext is missing its stored PBKDF2 salt; "
                "legacy broken password ciphertext cannot be reconstructed"
            )
        salt = base64.b64decode(parts[1], validate=True)
        key, _ = derive_key(password, salt=salt)
        encrypted_data = parts[2]
    elif key is None:
        raise ValueError("Either key or password must be provided")

    return AESCipher(key).decrypt(encrypted_data, nonce)


def generate_key() -> bytes:
    return os.urandom(32)


def generate_key_hex() -> str:
    return os.urandom(32).hex()
