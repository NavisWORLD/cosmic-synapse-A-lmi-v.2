import json

from a_lmi.security.encryption import decrypt_with_password, encrypt_with_password


def test_password_encryption_round_trip_persists_kdf_material():
    envelope = encrypt_with_password(b"cosmos-memory", "correct horse battery staple")
    parsed = json.loads(envelope)
    assert parsed["version"] == 1
    assert parsed["cipher"] == "AES-256-GCM"
    assert parsed["kdf"] == "PBKDF2-HMAC-SHA256"
    assert parsed["salt"]
    assert parsed["nonce"]
    assert decrypt_with_password(envelope, "correct horse battery staple") == b"cosmos-memory"
