import numpy as np

from a_lmi.core.light_token import LightToken


def make_token():
    return LightToken("memory://fixture", "text", "memory://raw", "hello cosmos")


def test_embedding_spectrum_is_explicit_one_sided_real_transform():
    token = make_token()
    token.set_embedding(np.arange(1536, dtype=np.float32))
    power = token.get_spectral_power()
    assert power.dtype.kind == "f"
    assert power.shape == (769,)
    assert token.metadata["spectral_transform"] == "embedding_rfft"


def test_lighttoken_round_trip_preserves_one_sided_spectrum():
    token = make_token()
    token.set_embedding(np.linspace(-1.0, 1.0, 1536, dtype=np.float32))
    restored = LightToken.from_json(token.to_json())
    np.testing.assert_allclose(restored.joint_embedding, token.joint_embedding)
    np.testing.assert_allclose(restored.spectral_signature, token.spectral_signature)
    assert restored.get_spectral_power().shape == (769,)
