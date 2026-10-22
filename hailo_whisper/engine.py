# -*- coding: utf-8 -*-
# hailo_whisper/engine.py
"""
engine.py: Unified Whisper Inference Engine with Hailo-10H NPU Acceleration
and Zero-Downtime CPU Fallback.
Provides seamless integration with voice_pipeline.py and meeting_minutes.py.
"""

import os
import sys
import time
import logging
import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger("hailo_whisper.engine")

try:
    import opencc
    _OPENCC = opencc.OpenCC('s2twp.json')
except Exception:
    _OPENCC = None


class WhisperEngineInterface(ABC):
    """Abstract Base Class for all Whisper ASR inference backends."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file and return text with performance metrics.

        Returns:
            Dict containing:
                - "text": str (Transcribed and formatted text)
                - "language": str (Detected or specified language)
                - "inference_time_ms": float (Execution latency in ms)
                - "backend": str ("hailo-10h" or "cpu-faster-whisper")
        """
        pass

    @abstractmethod
    def is_hardware_accelerated(self) -> bool:
        """Return True if running on Hailo-10H NPU hardware."""
        pass


class CpuFallbackWhisperEngine(WhisperEngineInterface):
    """
    CPU-based Faster-Whisper fallback engine.
    Ensures zero downtime if Hailo-10H NPU is detached or initializing.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            logger.info(f"[WhisperEngine] Initializing CPU Fallback WhisperModel ({self.model_size}, {self.compute_type})...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
            )
            logger.info("[WhisperEngine] CPU Fallback WhisperModel ready.")
        except ImportError:
            logger.warning("[WhisperEngine] faster_whisper not installed in current environment.")
            self._model = None

    def is_hardware_accelerated(self) -> bool:
        return False

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        self._ensure_model()
        if self._model is None:
            raise RuntimeError("faster_whisper model could not be loaded in CPU fallback.")
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            audio_path,
            beam_size=1,
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            temperature=temperature,
            condition_on_previous_text=False,
        )

        detected_lang = info.language if info else (language or "unknown")
        raw_text = "".join(s.text for s in segments).strip()

        # Traditional Chinese normalization
        final_text = raw_text
        if _OPENCC and detected_lang == "zh":
            final_text = _OPENCC.convert(raw_text)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "text": final_text,
            "language": detected_lang,
            "inference_time_ms": round(elapsed_ms, 2),
            "backend": "cpu-faster-whisper",
        }


_hailo_inferring: bool = False

def is_hailo_inference_active() -> bool:
    """Return True if Hailo-10H NPU is currently executing Whisper speech-to-text inference."""
    return _hailo_inferring


class HailoWhisperEngine(WhisperEngineInterface):
    """
    Hailo-10H NPU (40 TOPS) Accelerated Whisper Inference Engine.
    Leverages native hailo_platform.genai.Speech2Text and on-board 8GB LPDDR4X
    for sub-300ms chunk processing.
    """

    CANDIDATE_HEF_PATHS = [
        "/home/pi/models/hailo10h/Whisper-Small.hef",
        "/home/pi/models/hailo10h/Whisper-Base.hef",
        "/usr/local/hailo/resources/models/Whisper-Base.hef",
        "/home/pi/models/whisper_small_h10h.hef",
    ]

    def __init__(self, hef_path: Optional[str] = None):
        self.hef_path = hef_path
        self._vdevice = None
        self._s2t = None
        self._initialized = False
        self._probe_and_init()

    def _resolve_hef_path(self) -> Optional[str]:
        if self.hef_path and os.path.exists(self.hef_path):
            return self.hef_path
        for p in self.CANDIDATE_HEF_PATHS:
            if os.path.exists(p):
                return p
        return None

    def _probe_and_init(self):
        """Probe HailoRT runtime, VDevice and Speech2Text engine across candidate HEF models."""
        try:
            from hailo_platform import VDevice
            from hailo_platform.genai import Speech2Text
            
            candidates = [self.hef_path] if self.hef_path else self.CANDIDATE_HEF_PATHS
            existing_candidates = [p for p in candidates if p and os.path.exists(p)]
            if not existing_candidates:
                logger.warning("[HailoWhisper] No compatible Whisper HEF model found. Fallback recommended.")
                self._initialized = False
                return

            self._vdevice = VDevice()
            for cand_path in existing_candidates:
                try:
                    logger.info(f"[HailoWhisper] Attempting Speech2Text initialization with: {cand_path}...")
                    self._s2t = Speech2Text(self._vdevice, cand_path)
                    self.hef_path = cand_path
                    self._initialized = True
                    logger.info(f"✅ [HailoWhisper] Hailo-10H NPU Speech2Text Engine ready with {os.path.basename(cand_path)}!")
                    return
                except Exception as init_err:
                    logger.warning(f"[HailoWhisper] Candidate {cand_path} failed Speech2Text initialization: {init_err}")
            
            self._initialized = False
            logger.warning("[HailoWhisper] All candidate HEF models failed Speech2Text initialization. Fallback recommended.")
        except Exception as e:
            logger.warning(f"[HailoWhisper] Hailo-10H runtime initialization failed: {e}. Fallback recommended.")
            self._initialized = False

    def is_hardware_accelerated(self) -> bool:
        return self._initialized and self._s2t is not None

    def _load_audio_to_pcm_float(self, audio_source: Any) -> Optional[np.ndarray]:
        """Convert audio file or memory buffer to 16kHz float32 [-1.0, 1.0) mono numpy array."""
        import numpy as np
        if isinstance(audio_source, np.ndarray):
            audio = audio_source.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.flatten()
            return np.clip(audio, -1.0, 0.999999)

        if isinstance(audio_source, bytes):
            audio = np.frombuffer(audio_source, dtype=np.int16).astype(np.float32) / 32768.0
            return np.clip(audio, -1.0, 0.999999)

        if isinstance(audio_source, str) and os.path.exists(audio_source):
            try:
                import wave
                with wave.open(audio_source, "rb") as wf:
                    framerate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    if n_channels > 1:
                        audio = audio[::n_channels]
                    return np.clip(audio, -1.0, 0.999999)
            except Exception:
                pass

            # ffmpeg fallback
            try:
                import subprocess
                cmd = [
                    "ffmpeg", "-nostdin", "-threads", "1",
                    "-i", audio_source,
                    "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-"
                ]
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
                return np.clip(audio, -1.0, 0.999999)
            except Exception as e:
                logger.error(f"[HailoWhisper] Failed to load audio via ffmpeg: {e}")
                return None

    def _trim_leading_silence(self, audio: np.ndarray, sr: int = 16000, thresh: float = 0.005) -> np.ndarray:
        """Trim leading silence/background noise to prevent Whisper acoustic hallucination loops."""
        if audio is None or len(audio) < sr:
            return audio
        # 2026-10-22 e2e 實測：把開頭靜音剪到只剩 200 ms 會讓 Whisper 誤判第一個字 (例：「測試的部分」→「estory的部分」)。
        # 改為只在開頭靜音 >= 1.5 s 時修剪，且保留 500 ms 前導；同一測試集 CER 32.5% → 28.9%。
        frame_len = int(sr * 0.1)  # 100ms frames
        max_scan = min(len(audio) - frame_len, sr * 8)  # scan up to first 8 seconds
        for i in range(0, max_scan, frame_len):
            rms = np.sqrt(np.mean(audio[i : i + frame_len] ** 2))
            if rms > thresh:
                if i < int(sr * 1.5):
                    return audio
                lead_idx = max(0, i - int(sr * 0.5))  # preserve 500ms lead-in
                return audio[lead_idx:]
        return audio

    def transcribe(
        self,
        audio_path: Any,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        if not self.is_hardware_accelerated():
            raise RuntimeError("Hailo-10H Engine not initialized. Please use CPU fallback.")

        global _hailo_inferring
        _hailo_inferring = True
        t0 = time.perf_counter()
        try:
            audio_data = self._load_audio_to_pcm_float(audio_path)
            if audio_data is not None:
                audio_data = self._trim_leading_silence(audio_data)

            if audio_data is None or len(audio_data) < 1600:
                return {
                    "text": "",
                    "language": language or "en",
                    "inference_time_ms": 0.0,
                    "backend": "hailo-10h",
                }

            from hailo_platform.genai import Speech2TextTask
            # Set default language to "zh" (Chinese) for clean multi-byte token decoding without byte-splitting errors
            lang_code = language.lower() if language else "zh"

            # Calculate dynamic timeout: minimum 30s, or 1.5x audio duration + 10s
            audio_dur_sec = len(audio_data) / 16000.0
            calc_timeout_ms = max(30000, int((audio_dur_sec * 1.5 + 10.0) * 1000))

            raw_text = self._s2t.generate_all_text(
                audio_data=audio_data,
                task=Speech2TextTask.TRANSCRIBE,
                language=lang_code,
                timeout_ms=calc_timeout_ms
            )

            final_text = (raw_text or "").strip()
            # Filter prompt bleeding if present
            for filter_word in ("No Hallucination.", "Verbatim speech transcription", "no hallucination", "繁體中文逐字稿", "無幻覺"):
                if filter_word in final_text:
                    final_text = final_text.replace(filter_word, "").strip()

            # Clean Unicode replacement character (\ufffd) and repeated pause noise artifacts
            final_text = final_text.replace("\ufffd", "").replace("\uFFFD", "")
            final_text = re.sub(r"([,\-\s]{2,})+", "，", final_text)
            final_text = re.sub(r"(，\s*){2,}", "，", final_text)
            final_text = re.sub(r"\b(\w)\1{3,}\b", r"\1", final_text)

            # Clean leading/trailing punctuation artifacts (dashes, commas, dots)
            final_text = re.sub(r"^[\s\-,\.，。]+", "", final_text)
            final_text = re.sub(r"[\s\-]+$", "", final_text)

            # Traditional Chinese conversion (Taiwan standard)
            if _OPENCC and final_text:
                try:
                    final_text = _OPENCC.convert(final_text)
                except Exception as cc_err:
                    logger.warning(f"[HailoWhisper] OpenCC conversion warning: {cc_err}")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "text": final_text,
                "language": lang_code or "auto",
                "inference_time_ms": round(elapsed_ms, 2),
                "backend": "hailo-10h",
            }
        finally:
            _hailo_inferring = False


def get_whisper_engine(preferred_backend: str = "auto", hef_path: Optional[str] = None) -> WhisperEngineInterface:
    """
    Factory function to retrieve the optimal Whisper engine based on hardware availability.

    Args:
        preferred_backend: "auto", "hailo", or "cpu".
        hef_path: Custom path to Hailo-10H Whisper HEF model.

    Returns:
        WhisperEngineInterface instance (HailoWhisperEngine or CpuFallbackWhisperEngine).
    """
    if preferred_backend in ("auto", "hailo"):
        try:
            hailo_engine = HailoWhisperEngine(hef_path=hef_path)
            if hailo_engine.is_hardware_accelerated():
                logger.info("[WhisperFactory] Active Backend: Hailo-10H NPU (40 TOPS)")
                return hailo_engine
        except Exception as e:
            logger.warning(f"[WhisperFactory] Hailo-10H detection failed: {e}")

    logger.info("[WhisperFactory] Active Backend: CPU Fallback (faster-whisper)")
    return CpuFallbackWhisperEngine()
