# matutor/tts_service.py
import os
import uuid
import threading
from TTS.api import TTS
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "media")  # same as MEDIA_ROOT
os.makedirs(AUDIO_DIR, exist_ok=True)

class TTSService:
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
        self._tts = None

    def _ensure_loaded(self):
        if self._tts is None:
            with self._lock:
                if self._tts is None:
                    print(f"[TTSService] Loading {self.model_name} on {self.device} (may download weights)...")
                    self._tts = TTS(model_name=self.model_name, progress_bar=False, gpu=(self.device=="cuda"))
                    print("[TTSService] Model loaded.")

    def generate(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Text empty")
        self._ensure_loaded()
        filename = f"tts_{uuid.uuid4().hex}.wav"
        out_path = os.path.join(AUDIO_DIR, filename)
        # synchronous call; for higher QPS use a job queue / workers
        self._tts.tts_to_file(text=text, file_path=out_path)
        return out_path
