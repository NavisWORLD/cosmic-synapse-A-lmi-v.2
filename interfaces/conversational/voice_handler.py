"""
Voice Handler for Speech-to-Text and Text-to-Speech

Real-time voice interaction with the A-LMI system using Vosk.
"""

import pyaudio
import logging
from typing import Optional, Callable
import threading
import queue
import json


class VoiceHandler:
    """
    Voice input/output handler with Vosk STT.
    
    Features:
    - Real-time speech recognition
    - Continuous audio capture
    - Callback for recognized text
    - Optional text-to-speech
    """
    
    def __init__(self, model_path: str = "models/vosk-model-en-us-0.22", sample_rate: int = 16000):
        """
        Initialize voice handler.
        
        Args:
            model_path: Path to Vosk model
            sample_rate: Audio sample rate
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.model = None
        self.recognizer = None
        self.audio = None
        self.stream = None
        self.is_listening = False
        
        # Callback for recognized text
        self.on_text_recognized: Optional[Callable[[str], None]] = None
        
        # Try to load Vosk model
        try:
            from vosk import Model, KaldiRecognizer
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, sample_rate)
            self.recognizer.SetWords(True)
            self.logger.info("Vosk model loaded successfully")
        except ImportError:
            self.logger.warning("Vosk not installed. Install with: pip install vosk")
        except Exception as e:
            self.logger.warning(f"Could not load Vosk model: {e}")
    
    def start_listening(self):
        """Start continuous speech recognition."""
        if self.is_listening:
            self.logger.warning("Already listening")
            return
        
        if self.recognizer is None:
            self.logger.error("Vosk recognizer not initialized")
            return
        
        try:
            self.audio = pyaudio.PyAudio()
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=4000
            )
            
            self.is_listening = True
            
            # Start recognition thread
            self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
            self.recognition_thread.start()
            
            self.logger.info("Voice recognition started")
            
        except Exception as e:
            self.logger.error(f"Error starting voice recognition: {e}")
    
    def stop_listening(self):
        """Stop speech recognition."""
        if not self.is_listening:
            return
        
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        self.logger.info("Voice recognition stopped")
    
    def _recognition_loop(self):
        """Main recognition loop running in separate thread."""
        while self.is_listening:
            try:
                data = self.stream.read(4000)
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '')
                    
                    if text and self.on_text_recognized:
                        self.on_text_recognized(text)
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    # Handle partial results if needed
                    
            except Exception as e:
                self.logger.error(f"Error in recognition loop: {e}")
    
    def speak(self, text: str):
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
        """
        # TODO: Implement text-to-speech
        self.logger.info(f"Would speak: {text}")


def main():
    """Test voice handler."""
    handler = VoiceHandler()
    
    def on_text(text):
        print(f"You said: {text}")
    
    handler.on_text_recognized = on_text
    
    try:
        handler.start_listening()
        print("Listening... Press Ctrl+C to stop")
        
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        handler.stop_listening()


if __name__ == "__main__":
    main()

