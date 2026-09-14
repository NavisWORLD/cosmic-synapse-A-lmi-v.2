import numpy as np
import pytest

from a_lmi.services.audio_processor import AudioProcessor


def config():
    return {"a_lmi": {"perception": {"audio_processor": {"sample_rate": 16000, "chunk_size": 1024, "format": "paInt16", "channels": 1}}}}


def test_audio_import_and_construction_do_not_require_pyaudio():
    processor = AudioProcessor(config(), audio_backend=None)
    assert processor.audio_available is False
    assert processor._classify_environment(np.zeros(16000, dtype=np.int16)) is None
    assert processor._transcribe_speech(np.zeros(16000, dtype=np.int16)) is None
    with pytest.raises(RuntimeError, match="audio backend"):
        processor.start_recording()
