import numpy as np

from a_lmi.services.multimodal_encoder import (
    CLIP_MODEL_ID,
    CLIP_MODEL_REVISION,
    WAVLM_MODEL_ID,
    WAVLM_MODEL_REVISION,
    MultimodalEncoder,
)


def test_encoder_imports_without_loading_optional_ml_models_and_names_spaces():
    clip_space = f"clip:{CLIP_MODEL_ID}@{CLIP_MODEL_REVISION}"
    wavlm_space = f"wavlm:{WAVLM_MODEL_ID}@{WAVLM_MODEL_REVISION}"
    assert MultimodalEncoder.embedding_space_for("text") == clip_space
    assert MultimodalEncoder.embedding_space_for("image") == clip_space
    assert MultimodalEncoder.embedding_space_for("audio") == wavlm_space
    assert MultimodalEncoder.embedding_spaces_aligned("text", "image") is True
    assert MultimodalEncoder.embedding_spaces_aligned("text", "audio") is False


def test_native_embedding_is_packed_deterministically_without_random_projection():
    native = np.arange(1, 769, dtype=np.float32)
    packed = MultimodalEncoder.pack_native_embedding(native)

    assert packed.shape == (1536,)
    assert packed.dtype == np.float32
    np.testing.assert_array_equal(packed[768:], np.zeros(768, dtype=np.float32))
    assert np.isclose(np.linalg.norm(packed), 1.0, atol=1e-6)
