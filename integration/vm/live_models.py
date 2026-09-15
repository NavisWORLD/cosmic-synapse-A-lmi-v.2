"""Real CPU model execution probes for VM integration evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from a_lmi.services.multimodal_encoder import (
    CLIP_MODEL_ID,
    CLIP_MODEL_REVISION,
    WAVLM_MODEL_ID,
    WAVLM_MODEL_REVISION,
    MultimodalEncoder,
)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True))


def run_clip() -> None:
    from PIL import Image

    encoder = MultimodalEncoder(device="cpu")
    text_vector = encoder.encode_text("COSMIC SYNAPSE VM integration probe")
    image = Image.new("RGB", (224, 224), color=(32, 64, 128))
    image_vector = encoder.encode_image(image)

    assert text_vector.shape == (1536,)
    assert image_vector.shape == (1536,)
    assert np.isfinite(text_vector).all()
    assert np.isfinite(image_vector).all()
    assert np.linalg.norm(text_vector) > 0
    assert np.linalg.norm(image_vector) > 0
    _emit(
        {
            "gate": "clip-live-cpu",
            "model": CLIP_MODEL_ID,
            "revision": CLIP_MODEL_REVISION,
            "text_shape": list(text_vector.shape),
            "image_shape": list(image_vector.shape),
            "device": "cpu",
        }
    )


def run_wavlm() -> None:
    encoder = MultimodalEncoder(device="cpu")
    sample_rate = 16000
    seconds = 1.0
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    vector = encoder.encode_audio(waveform, sample_rate=sample_rate)

    assert vector.shape == (1536,)
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) > 0
    _emit(
        {
            "gate": "wavlm-live-cpu",
            "model": WAVLM_MODEL_ID,
            "revision": WAVLM_MODEL_REVISION,
            "shape": list(vector.shape),
            "samples": int(waveform.size),
            "sample_rate": sample_rate,
            "device": "cpu",
        }
    )


def run_vosk() -> None:
    from vosk import KaldiRecognizer, Model, SetLogLevel

    from a_lmi.services.audio_processor import AudioProcessor

    model_name = "vosk-model-small-en-us-0.15"
    model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    SetLogLevel(-1)

    with tempfile.TemporaryDirectory(prefix="cosmic-vosk-") as tmp:
        root = Path(tmp)
        archive = root / f"{model_name}.zip"
        urllib.request.urlretrieve(model_url, archive)
        with zipfile.ZipFile(archive, "r") as handle:
            handle.extractall(root)
        model_path = root / model_name
        assert model_path.is_dir()

        recognizer = KaldiRecognizer(Model(str(model_path)), 16000)
        config = {
            "a_lmi": {
                "perception": {
                    "audio_processor": {
                        "sample_rate": 16000,
                        "chunk_size": 4096,
                        "format": "paInt16",
                        "channels": 1,
                    },
                    "speech_to_text": {
                        "model_path": str(model_path),
                        "sample_rate": 16000,
                    },
                }
            }
        }
        processor = AudioProcessor(
            config,
            audio_backend=None,
            speech_recognizer=recognizer,
        )
        # Three seconds of deterministic PCM is sufficient to exercise the real
        # recognizer path even when silence legitimately produces no transcript.
        pcm = np.zeros(16000 * 3, dtype=np.int16)
        result = processor._transcribe_speech(pcm)
        final = json.loads(recognizer.FinalResult())
        assert result is None or isinstance(result, str)
        assert isinstance(final, dict) and "text" in final
        _emit(
            {
                "gate": "vosk-live-cpu",
                "model": model_name,
                "model_url": model_url,
                "sample_rate": 16000,
                "samples": int(pcm.size),
                "transcript_present": bool((result or final.get("text", "")).strip()),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["clip", "wavlm", "vosk"])
    args = parser.parse_args()
    if args.command == "clip":
        run_clip()
    elif args.command == "wavlm":
        run_wavlm()
    elif args.command == "vosk":
        run_vosk()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
