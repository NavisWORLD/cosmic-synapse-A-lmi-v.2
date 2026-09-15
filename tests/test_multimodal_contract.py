import numpy as np
import pytest

from a_lmi.services.multimodal_encoder import (
    CLIP_MODEL_ID,
    CLIP_MODEL_REVISION,
    WAVLM_MODEL_ID,
    WAVLM_MODEL_REVISION,
    MultimodalEncoder,
    adapt_embedding_dimension,
    embedding_space_for,
)


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


def test_embedding_space_provenance_pins_exact_hub_revisions():
    assert len(CLIP_MODEL_REVISION) == 40
    assert len(WAVLM_MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in CLIP_MODEL_REVISION)
    assert all(character in "0123456789abcdef" for character in WAVLM_MODEL_REVISION)
    assert embedding_space_for("text") == f"clip:{CLIP_MODEL_ID}@{CLIP_MODEL_REVISION}"
    assert embedding_space_for("audio") == f"wavlm:{WAVLM_MODEL_ID}@{WAVLM_MODEL_REVISION}"


def test_feature_tensor_accepts_transformers_structured_pooling_output():
    class FakeTensor:
        pass

    class StructuredOutput:
        def __init__(self):
            self.pooler_output = FakeTensor()

    output = StructuredOutput()
    assert MultimodalEncoder._feature_tensor(output) is output.pooler_output


def test_feature_tensor_rejects_output_without_tensor_candidate():
    class EmptyOutput:
        pooler_output = None
        last_hidden_state = None

    with pytest.raises(RuntimeError, match="tensor"):
        MultimodalEncoder._feature_tensor(EmptyOutput())
