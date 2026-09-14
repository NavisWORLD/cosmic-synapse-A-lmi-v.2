import numpy as np

from a_lmi.services.multimodal_encoder import adapt_embedding_dimension, embedding_space_for


def test_dimension_adapter_is_deterministic_and_preserves_norm():
    source = np.linspace(-1.0, 1.0, 768, dtype=np.float32)
    first = adapt_embedding_dimension(source, 1536)
    second = adapt_embedding_dimension(source, 1536)
    assert first.shape == (1536,)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first), np.linalg.norm(source), rtol=1e-6)


def test_embedding_spaces_do_not_pretend_wavlm_is_clip_aligned():
    assert embedding_space_for("text") == embedding_space_for("image")
    assert embedding_space_for("audio") != embedding_space_for("text")
    assert embedding_space_for("speech") == embedding_space_for("audio")
