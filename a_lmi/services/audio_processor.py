"""Optional real-time audio capture and provider-based audio analysis.

PyAudio, Vosk, and learned environmental classifiers are optional. Importing or
constructing the service does not open hardware. Missing providers are reported
as unavailable rather than replaced with fabricated classifications or text.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

_AUTO = object()


class AudioProcessor:
    """Real-time microphone service with explicit optional providers."""

    def __init__(
        self,
        config: dict,
        *,
        audio_backend: Any = _AUTO,
        esc_classifier: Any = None,
        speech_recognizer: Any = None,
        artifact_store: Any = None,
    ):
        self.root_config = config
        perception = config["a_lmi"]["perception"]
        self.config = perception["audio_processor"]
        self.speech_config = perception.get("speech_to_text", {})
        self.logger = logging.getLogger(__name__)

        if audio_backend is _AUTO:
            try:
                import pyaudio as imported_backend
            except ImportError:
                imported_backend = None
            audio_backend = imported_backend
        self.audio_backend = audio_backend
        self.audio_available = audio_backend is not None

        self.sample_rate = int(self.config["sample_rate"])
        self.chunk_size = int(self.config["chunk_size"])
        self.channels = int(self.config["channels"])
        self.format_name = str(self.config.get("format", "paInt16"))
        self.format = self._parse_format(self.format_name)

        self.audio = None
        self.stream = None
        self.is_recording = False
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None

        self.esc_classifier = esc_classifier
        self.speech_recognizer = speech_recognizer
        self.artifact_store = artifact_store
        self._vosk_load_attempted = speech_recognizer is not None
        self.on_audio_processed: Optional[Callable[[dict], None]] = None

    def is_available(self) -> bool:
        return bool(self.audio_available)

    def _parse_format(self, format_str: str):
        if self.audio_backend is None:
            return None
        format_map = {
            "paInt16": getattr(self.audio_backend, "paInt16", None),
            "paInt32": getattr(self.audio_backend, "paInt32", None),
            "paFloat32": getattr(self.audio_backend, "paFloat32", None),
        }
        value = format_map.get(format_str)
        if value is None:
            raise ValueError(f"Unsupported audio format: {format_str}")
        return value

    def start_recording(self) -> None:
        if not self.audio_available:
            raise RuntimeError(
                "audio backend is unavailable; install the audio extra and configure microphone access"
            )
        if self.is_recording:
            return

        self.audio = self.audio_backend.PyAudio()
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            input=True,
            stream_callback=self._audio_callback,
        )
        self.is_recording = True
        self.stream.start_stream()
        self.processing_thread = threading.Thread(
            target=self._process_audio_loop, daemon=True, name="a-lmi-audio"
        )
        self.processing_thread.start()

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        self.is_recording = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording and in_data:
            self.audio_queue.put(bytes(in_data))
        pa_continue = (
            getattr(self.audio_backend, "paContinue", 0)
            if self.audio_backend is not None
            else 0
        )
        return (None, pa_continue)

    def _process_audio_loop(self) -> None:
        buffer: list[int] = []
        buffer_size = int(self.sample_rate * 3.0)
        while self.is_recording:
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                buffer.extend(np.frombuffer(audio_data, dtype=np.int16))
                while len(buffer) >= buffer_size:
                    audio_chunk = np.asarray(buffer[:buffer_size], dtype=np.int16)
                    del buffer[:buffer_size]
                    timestamp = datetime.now(timezone.utc).isoformat()
                    audio_ref = None
                    raw_sha256 = None
                    if self.artifact_store is not None:
                        artifact = self.artifact_store.store_bytes(
                            audio_chunk.tobytes(),
                            f"audio/{timestamp.replace(':', '-')}.pcm",
                            "audio/L16",
                        )
                        audio_ref = artifact.uri
                        raw_sha256 = artifact.sha256
                    event = {
                        "timestamp": timestamp,
                        "type": "audio",
                        "sample_rate": self.sample_rate,
                        "esc_class": self._classify_environment(audio_chunk),
                        "transcription": self._transcribe_speech(audio_chunk),
                        "audio_ref": audio_ref,
                        "raw_sha256": raw_sha256,
                        "stream_id": "microphone",
                    }
                    if self.on_audio_processed:
                        self.on_audio_processed(event)
            except Exception as exc:
                self.logger.error("Audio processing error: %s", exc, exc_info=True)

    def _classify_environment(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Return a provider result or None; never invent a placeholder class."""

        if self.esc_classifier is None:
            return None
        result = (
            self.esc_classifier.classify(audio_chunk)
            if hasattr(self.esc_classifier, "classify")
            else self.esc_classifier(audio_chunk)
        )
        if isinstance(result, tuple):
            return str(result[0])
        return None if result is None else str(result)

    def _ensure_vosk_recognizer(self) -> None:
        if self.speech_recognizer is not None or self._vosk_load_attempted:
            return
        self._vosk_load_attempted = True
        model_path = self.speech_config.get("model_path")
        if not model_path or not Path(model_path).is_dir():
            return
        try:
            from vosk import KaldiRecognizer, Model
        except ImportError:
            self.logger.info("Vosk not installed; speech transcription disabled")
            return
        expected_rate = int(self.speech_config.get("sample_rate", self.sample_rate))
        if expected_rate != self.sample_rate:
            self.logger.warning(
                "Vosk model sample rate %s does not match capture rate %s; transcription disabled",
                expected_rate,
                self.sample_rate,
            )
            return
        self.speech_recognizer = KaldiRecognizer(Model(model_path), self.sample_rate)

    def _transcribe_speech(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Transcribe with an injected/Vosk recognizer when available."""

        self._ensure_vosk_recognizer()
        recognizer = self.speech_recognizer
        if recognizer is None:
            return None
        payload = np.asarray(audio_chunk, dtype=np.int16).tobytes()
        if hasattr(recognizer, "AcceptWaveform"):
            if not recognizer.AcceptWaveform(payload):
                return None
            result = json.loads(recognizer.Result())
            text = str(result.get("text", "")).strip()
            return text or None
        if callable(recognizer):
            result = recognizer(audio_chunk, self.sample_rate)
            return None if result is None else str(result)
        raise TypeError("Unsupported speech recognizer interface")


class EnvironmentalSoundClassifier:
    """Adapter for a real environmental classifier supplied by a deployment."""

    def __init__(self, model: Any = None):
        self.model = model

    def classify(self, audio_chunk: np.ndarray) -> tuple[str, float]:
        if self.model is None:
            raise RuntimeError("No environmental sound model is configured")
        result = self.model(audio_chunk)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Environmental classifier must return (label, confidence)")
        return str(result[0]), float(result[1])
