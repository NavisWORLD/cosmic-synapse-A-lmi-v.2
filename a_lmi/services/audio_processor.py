"""
Audio Processing Service

Handles real-time audio input, speech-to-text transcription, and 
Environmental Sound Classification (ESC) for acoustic context tracking.
"""

import pyaudio
import wave
import numpy as np
import logging
from typing import Optional, Callable
import threading
import queue
import time
from datetime import datetime, timezone

# Note: Vosk requires separate installation
# from vosk import Model, KaldiRecognizer
import json


class AudioProcessor:
    """
    Real-time audio processor with ESC and speech recognition.
    
    Implements:
    - Continuous microphone streaming
    - Environmental Sound Classification
    - Speech-to-text transcription (Vosk)
    - Acoustic context metadata capture
    """
    
    def __init__(self, config: dict):
        """
        Initialize audio processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['a_lmi']['perception']['audio_processor']
        self.logger = logging.getLogger(__name__)
        
        # Audio parameters
        self.sample_rate = self.config['sample_rate']
        self.chunk_size = self.config['chunk_size']
        self.format = self._parse_format(self.config['format'])
        self.channels = self.config['channels']
        
        # Audio streaming
        self.audio = None
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # ESC model (would be loaded from trained PyTorch model)
        self.esc_model = None
        self.esc_labels = [
            'silence', 'indoor', 'outdoor', 'transport', 'nature',
            'office', 'cafe', 'library', 'street', 'park',
            'residential', 'commercial', 'industrial'
        ]
        
        # Speech recognition (Vosk)
        # self.vosk_model = None
        # self.recognizer = None
        
        # Callback for processed audio data
        self.on_audio_processed: Optional[Callable] = None
    
    def _parse_format(self, format_str: str) -> int:
        """Parse format string to pyaudio format constant."""
        format_map = {
            'paInt16': pyaudio.paInt16,
            'paInt32': pyaudio.paInt32,
            'paFloat32': pyaudio.paFloat32
        }
        return format_map.get(format_str, pyaudio.paInt16)
    
    def start_recording(self):
        """Start continuous audio recording."""
        if self.is_recording:
            self.logger.warning("Already recording")
            return
        
        self.audio = pyaudio.PyAudio()
        
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            frames_per_buffer=self.chunk_size,
            input=True,
            stream_callback=self._audio_callback
        )
        
        self.is_recording = True
        self.stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Started audio recording")
    
    def stop_recording(self):
        """Stop audio recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        self.logger.info("Stopped audio recording")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback for audio stream data.
        
        This is called by pyaudio whenever new audio data is available.
        """
        if self.is_recording:
            self.audio_queue.put(in_data)
        
        return (None, pyaudio.paContinue)
    
    def _process_audio_loop(self):
        """
        Main processing loop (runs in separate thread).
        
        Processes audio chunks from queue:
        1. Classify environmental sound
        2. Perform speech-to-text
        3. Emit processed data
        """
        buffer = []
        buffer_duration = 3.0  # seconds of audio to buffer
        buffer_size = int(self.sample_rate * buffer_duration)
        
        while self.is_recording:
            try:
                # Get audio chunk from queue
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                buffer.extend(audio_array)
                
                # Process when we have enough data
                if len(buffer) >= buffer_size:
                    # Prepare audio chunk
                    audio_chunk = np.array(buffer[:buffer_size], dtype=np.int16)
                    buffer = buffer[buffer_size:]
                    
                    # Classify environment
                    esc_class = self._classify_environment(audio_chunk)
                    
                    # Transcribe speech
                    transcription = self._transcribe_speech(audio_chunk)
                    
                    # Create processed audio event
                    event = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'type': 'audio',
                        'sample_rate': self.sample_rate,
                        'esc_class': esc_class,
                        'transcription': transcription,
                        'audio_ref': None,  # Would store in object storage
                        'stream_id': 'microphone'
                    }
                    
                    # Call callback if set
                    if self.on_audio_processed:
                        self.on_audio_processed(event)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in audio processing: {e}", exc_info=True)
    
    def _classify_environment(self, audio_chunk: np.ndarray) -> str:
        """
        Classify environmental sound using ESC model.
        
        In production, this would use a trained CNN model.
        For now, returns a placeholder classification.
        
        Args:
            audio_chunk: Audio signal
            
        Returns:
            ESC class label
        """
        # TODO: Implement actual ESC model inference
        # For now, return placeholder
        return 'indoor'
    
    def _transcribe_speech(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Transcribe speech using Vosk.
        
        Args:
            audio_chunk: Audio signal
            
        Returns:
            Transcription text or None
        """
        # TODO: Implement Vosk transcription
        # For now, return None
        return None


class EnvironmentalSoundClassifier:
    """
    Environmental Sound Classification using deep learning.
    
    This classifies ambient acoustic conditions for the vibrational
    information theory - tracking how environmental frequencies affect processing.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize ESC classifier.
        
        Args:
            model_path: Path to trained PyTorch model
        """
        self.model_path = model_path
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Load model (would use torch.load in production)
        # self.model = torch.load(model_path)
    
    def classify(self, audio_chunk: np.ndarray) -> tuple[str, float]:
        """
        Classify environmental sound.
        
        Args:
            audio_chunk: Audio signal
            
        Returns:
            (class_label, confidence) tuple
        """
        # TODO: Implement actual classification
        return ('indoor', 0.5)


def main():
    """Test audio processor."""
    import yaml
    
    # Load config
    with open('infrastructure/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create processor
    processor = AudioProcessor(config)
    
    # Define callback
    def on_audio(audio_data):
        print(f"Audio event: {audio_data['esc_class']} - {audio_data.get('transcription')}")
    
    processor.on_audio_processed = on_audio
    
    # Start recording
    try:
        processor.start_recording()
        print("Recording... Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        processor.stop_recording()


if __name__ == "__main__":
    main()

