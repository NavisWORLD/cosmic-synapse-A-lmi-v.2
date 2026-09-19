import numpy as np

from a_lmi.core.light_token import LightToken, resonance_match, spectral_similarity


def make_token():
    return LightToken("memory://fixture", "text", "memory://raw", "hello cosmos")


def token_with_embedding(values):
    token = make_token()
    token.set_embedding(np.asarray(values, dtype=np.float32))
    return token


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


def test_similarity_degenerate_contract_for_identical_zero_spectra():
    zeros = np.zeros(1536, dtype=np.float32)
    a = token_with_embedding(zeros)
    b = token_with_embedding(zeros)
    assert spectral_similarity(a, b, "power_correlation") == 1.0
    assert spectral_similarity(a, b, "cosine") == 1.0
    assert spectral_similarity(a, b, "euclidean") == 1.0


def test_similarity_methods_are_symmetric_for_finite_vectors():
    a = token_with_embedding(np.linspace(-1.0, 1.0, 1536, dtype=np.float32))
    b = token_with_embedding(np.sin(np.arange(1536, dtype=np.float32) / 13.0))
    for method in ("power_correlation", "cosine", "euclidean"):
        ab = spectral_similarity(a, b, method)
        ba = spectral_similarity(b, a, method)
        assert np.isfinite(ab)
        assert np.isclose(ab, ba, rtol=1e-6, atol=1e-6)


def test_resonance_match_orders_scores_descending():
    query = token_with_embedding(np.linspace(-1.0, 1.0, 1536, dtype=np.float32))
    same = token_with_embedding(np.linspace(-1.0, 1.0, 1536, dtype=np.float32))
    shifted = token_with_embedding(np.linspace(-0.8, 1.2, 1536, dtype=np.float32))
    matches = resonance_match(query, [shifted, same], threshold=-1.0)
    assert matches[0][0] is same
    assert matches[0][1] >= matches[1][1]


def test_unknown_similarity_method_fails_closed():
    a = token_with_embedding(np.zeros(1536, dtype=np.float32))
    b = token_with_embedding(np.zeros(1536, dtype=np.float32))
    try:
        spectral_similarity(a, b, "not-a-method")
    except ValueError as exc:
        assert "Unknown similarity method" in str(exc)
    else:
        raise AssertionError("unknown similarity method must raise ValueError")
