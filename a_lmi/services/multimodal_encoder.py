"""Optional multimodal encoders with explicit embedding-space provenance.

Text and images use the same pretrained CLIP space. Audio/speech use WavLM
and are intentionally labelled as a different space. The active 1536-value
LightToken carrier is produced by deterministic dimension adaptation (padding
or truncation + norm preservation), not an untrained random projection.

Heavy ML dependencies and model weights are loaded only when an encode method
is actually called.
"""

from __future__ import annotations

import io
import logging
from typing import List, Union

import numpy as np

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"
LIGHTTOKEN_DIMENSION = 1536


def embedding_space_for(modality: str) -> str:
    normalized = modality.lower()
    if normalized in {"text", "image"}:
        return f"clip:{CLIP_MODEL_ID}"
    if normalized in {"audio", "speech"}:
        return f"wavlm:{WAVLM_MODEL_ID}"
    raise ValueError(f"Unknown modality: {modality}")


def embedding_spaces_aligned(first: str, second: str) -> bool:
    """Return whether modalities share a pretrained semantic embedding space."""

    return embedding_space_for(first) == embedding_space_for(second)


def adapt_embedding_dimension(
    embedding: np.ndarray, target_dim: int = LIGHTTOKEN_DIMENSION
) -> np.ndarray:
    """Deterministically adapt the final dimension while preserving L2 norm.

    Padding with zeros is lossless when expanding the current 768-dimensional
    CLIP/WavLM outputs to the 1536-value LightToken carrier. Shrinking is a
    compatibility operation and therefore truncates before rescaling to the
    original norm; no semantic-alignment claim is made for that case.
    """

    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim not in {1, 2}:
        raise ValueError("embedding must be a 1D vector or 2D batch")
    if target_dim <= 0:
        raise ValueError("target_dim must be positive")

    source_dim = array.shape[-1]
    if source_dim == target_dim:
        return array.copy()

    if source_dim < target_dim:
        pad_width = [(0, 0)] * array.ndim
        pad_width[-1] = (0, target_dim - source_dim)
        return np.pad(array, pad_width, mode="constant").astype(np.float32)

    truncated = array[..., :target_dim].astype(np.float32, copy=True)
    source_norm = np.linalg.norm(array, axis=-1, keepdims=True)
    target_norm = np.linalg.norm(truncated, axis=-1, keepdims=True)
    scale = np.divide(
        source_norm,
        target_norm,
        out=np.ones_like(source_norm, dtype=np.float32),
        where=target_norm != 0,
    )
    return (truncated * scale).astype(np.float32)


class MultimodalEncoder:
    """CLIP + WavLM encoder with lazy optional dependencies."""

    def __init__(self, device: str = "auto"):
        self.logger = logging.getLogger(__name__)
        self.requested_device = device
        self._torch = None
        self.device = None
        self.clip_model = None
        self.clip_processor = None
        self.wavlm_model = None
        self.wavlm_processor = None

    @staticmethod
    def embedding_space_for(modality: str) -> str:
        return embedding_space_for(modality)

    @staticmethod
    def embedding_spaces_aligned(first: str, second: str) -> bool:
        return embedding_spaces_aligned(first, second)

    @staticmethod
    def pack_native_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        return adapt_embedding_dimension(vector, LIGHTTOKEN_DIMENSION)

    def _load_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Multimodal encoding requires the optional ML dependencies; install .[ml]"
            ) from exc
        self._torch = torch
        if self.requested_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.requested_device)
        return torch

    def _ensure_clip(self) -> None:
        if self.clip_model is not None:
            return
        torch = self._load_torch()
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP encoding requires transformers; install the optional ML extra"
            ) from exc
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        self.logger.info("Loaded CLIP model %s on %s", CLIP_MODEL_ID, self.device)

    def _ensure_wavlm(self) -> None:
        if self.wavlm_model is not None:
            return
        self._load_torch()
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel
        except ImportError as exc:
            raise RuntimeError(
                "WavLM encoding requires transformers; install the optional ML extra"
            ) from exc
        self.wavlm_model = WavLMModel.from_pretrained(WAVLM_MODEL_ID)
        self.wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
        self.wavlm_model.to(self.device)
        self.wavlm_model.eval()
        for parameter in self.wavlm_model.parameters():
            parameter.requires_grad = False
        self.logger.info("Loaded WavLM model %s on %s", WAVLM_MODEL_ID, self.device)

    @staticmethod
    def _normalize_and_pack(features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        normalized = np.divide(
            values,
            norms,
            out=np.zeros_like(values, dtype=np.float32),
            where=norms != 0,
        )
        return adapt_embedding_dimension(normalized, LIGHTTOKEN_DIMENSION)

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        self._ensure_clip()
        torch = self._torch
        is_single = isinstance(text, str)
        texts = [text] if is_single else list(text)
        inputs = self.clip_processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        packed = self._normalize_and_pack(features.detach().cpu().numpy())
        return packed[0] if is_single else packed

    def encode_image(self, image) -> np.ndarray:
        self._ensure_clip()
        torch = self._torch
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Image encoding requires Pillow; install the ML extra") from exc

        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        return self._normalize_and_pack(features.detach().cpu().numpy())[0]

    def encode_audio(
        self,
        audio: Union[np.ndarray, bytes, str],
        sample_rate: int = 16000,
    ) -> np.ndarray:
        self._ensure_wavlm()
        torch = self._torch

        if isinstance(audio, (bytes, str)):
            try:
                import soundfile as sf
            except ImportError as exc:
                raise RuntimeError("Audio decoding requires soundfile; install .[audio]") from exc
            audio, sample_rate = sf.read(io.BytesIO(audio) if isinstance(audio, bytes) else audio)

        values = np.asarray(audio, dtype=np.float32)
        if values.ndim > 1:
            values = values.mean(axis=1)
        if sample_rate != 16000:
            try:
                import librosa
            except ImportError as exc:
                raise RuntimeError("Audio resampling requires librosa; install .[audio]") from exc
            values = librosa.resample(values, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        inputs = self.wavlm_processor(
            values, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.wavlm_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        return self._normalize_and_pack(features.detach().cpu().numpy())[0]

    def encode(self, data, modality: str = "text") -> np.ndarray:
        normalized = modality.lower()
        if normalized == "text":
            return self.encode_text(data)
        if normalized == "image":
            return self.encode_image(data)
        if normalized in {"audio", "speech"}:
            return self.encode_audio(data)
        raise ValueError(f"Unknown modality: {modality}")


def load_encoder(device: str = "auto") -> MultimodalEncoder:
    return MultimodalEncoder(device=device)
