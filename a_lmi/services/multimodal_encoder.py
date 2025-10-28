"""
Multimodal Encoder with CLIP and Audio Support

Implements proper multimodal embeddings using OpenAI CLIP for text/images
and WavLM for audio, with projection to unified 1536-dimensional space.
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, WavLMModel, Wav2Vec2FeatureExtractor
from PIL import Image
import numpy as np
import logging
from typing import Union, List, Tuple
import io


class MultimodalEncoder(nn.Module):
    """
    Unified multimodal encoder for text, images, and audio.
    
    Uses:
    - CLIP (ViT-Large) for text and images
    - WavLM-Base for audio
    - Projection layers to 1536-dimensional unified space
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize multimodal encoder.
        
        Args:
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load CLIP model for text and images
        self.logger.info("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        # Load WavLM for audio
        self.logger.info("Loading WavLM model...")
        self.wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm_model.to(self.device)
        self.wavlm_model.eval()
        
        # Projection layers guide to 1536 dimensions
        clip_dim = self.clip_model.config.projection_dim  # 768 for ViT-Large
        wavlm_dim = self.wavlm_model.config.hidden_size   # 768 for WavLM-Base
        
        target_dim = 1536
        
        # CLIP projection
        self.clip_projection = nn.Linear(clip_dim, target_dim)
        
        # Audio projection
        self.audio_projection = nn.Linear(wavlm_dim, target_dim)
        
        # Move projections to device
        self.clip_projection.to(self.device)
        self.audio_projection.to(self.device)
        
        # Freeze encoder models (only train projections if needed)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.wavlm_model.parameters():
            param.requires_grad = False
        
        self.logger.info("Multimodal encoder initialized")
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text using CLIP.
        
        Args:
            text: Text string or list of strings
            
        Returns:
            1536-dimensional embedding(s)
        """
        if isinstance(text, str):
            text = [text]
        
        # Process with CLIP
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get CLIP text embeddings
            text_features = self.clip_model.get_text_features(**inputs)
            # Project to 1536 dimensions
            projected = self.clip_projection(text_features)
            # L2 normalize
            projected = projected / projected.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embeddings = projected.cpu().numpy().astype(np.float32)
        
        # Return single embedding if single input
        if len(text) == 1:
            return embeddings[0]
        
        return embeddings
    
    def encode_image(self, image: Union[Image.Image, np.ndarray, bytes]) -> np.ndarray:
        """
        Encode image using CLIP.
        
        Args:
            image: PIL Image, numpy array, or bytes
            
        Returns:
            1536-dimensional embedding
        """
        # Convert to PIL Image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Process with CLIP
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get CLIP image embeddings
            image_features = self.clip_model.get_image_features(**inputs)
            # Project to 1536 dimensions
            projected = self.clip_projection(image_features)
            # L2 normalize
            projected = projected / projected.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embedding = projected.cpu().numpy().astype(np.float32)
        
        return embedding[0]  # Return single embedding
    
    def encode_audio(
        self,
        audio: Union[np.ndarray, bytes, str],
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Encode audio using WavLM.
        
        Args:
            audio: Audio array (1D), bytes, or file path
            sample_rate: Sample rate of audio
            
        Returns:
            1536-dimensional embedding
        """
        # Convert audio to numpy array if needed
        if isinstance(audio, bytes):
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(audio))
            sample_rate = sr
        elif isinstance(audio, str):
            import soundfile as sf
            audio, sample_rate = sf.read(audio)
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Ensure correct sample rate (WavLM expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Process with WavLM
        inputs = self.wavlm_processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get WavLM embeddings
            outputs = self.wavlm_model(**inputs)
            # Use mean pooling over time dimension
            audio_features = outputs.last_hidden_state.mean(dim=1)
            # Project to 1536 dimensions
            projected = self.audio_projection(audio_features)
            # L2 normalize
            projected = projected / projected.norm(dim=-1, keepdim=True)
        
        # Convert to numpy
        embedding = projected.cpu().numpy().astype(np.float32)
        
        return embedding[0]
    
    def encode(
        self,
        data: Union[str, Image.Image, np.ndarray, bytes],
        modality: str = 'text'
    ) -> np.ndarray:
        """
        Generic encode method that dispatches based on modality.
        
        Args:
            data: Input data (text, image, or audio)
            modality: Type of data ('text', 'image', or 'audio')
            
        Returns:
            1536-dimensional embedding
        """
        if modality == 'text':
            return self.encode_text(data)
        elif modality == 'image':
            return self.encode_image(data)
        elif modality == 'audio':
            return self.encode_audio(data)
        else:
            raise ValueError(f"Unknown modality: {modality}")


def load_encoder(device: str = 'auto') -> MultimodalEncoder:
    """
    Load and return a MultimodalEncoder instance.
    
    Args:
        device: Device to use
        
    Returns:
        Initialized MultimodalEncoder
    """
    return MultimodalEncoder(device=device)

