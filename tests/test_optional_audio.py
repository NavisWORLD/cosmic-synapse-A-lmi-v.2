from a_lmi.services.audio_processor import AudioProcessor


def _config():
    return {
        "a_lmi": {
            "perception": {
                "audio_processor": {
                    "sample_rate": 44100,
                    "chunk_size": 4096,
                    "format": "paInt16",
                    "channels": 1,
                }
            }
        }
    }


def test_audio_processor_can_be_constructed_without_opening_optional_hardware():
    processor = AudioProcessor(_config())
    assert isinstance(processor.is_available(), bool)
    assert processor.audio is None
    assert processor.stream is None
    assert processor.is_recording is False
