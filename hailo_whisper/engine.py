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

# Whisper 解碼參數 (HailoRT >= 5.2.0 的 Speech2Text 支援 repetition_penalty 與自動語言偵測)。
#   PI5_WHISPER_LANGUAGE            預設語言 (ISO-639-1，例如 zh、en)；"auto" = 由模型自動偵測
#   PI5_WHISPER_REPETITION_PENALTY  重複懲罰，1.0 = 不懲罰
# 2026-10-29 raspi543 A/B (Whisper-Small HEF，8 s 切片，1/3/5/10 分鐘測試會議＋中英夾雜／英文短句＋底噪)：
#   - repetition_penalty 1.1：平均 CER 0.401 → 0.363，10 分鐘會議的重複迴圈 (同一片段連續 10 次) 消失；1.05／1.15／1.2 以上都較差。
#   - auto：中文會議與強制 zh 結果相同；英文句不再被翻成中文 (CER 0.54 → 0)；底噪片段的幻覺 8/10 → 1–2/10。
WHISPER_LANGUAGE = os.environ.get("PI5_WHISPER_LANGUAGE", "auto").strip().lower() or "auto"
WHISPER_REPETITION_PENALTY = float(os.environ.get("PI5_WHISPER_REPETITION_PENALTY", "1.1"))
# 近乎靜音 (RMS < -55 dBFS) 不送 NPU：
#   - 合成測試：數位靜音被解碼成「有 有，有」。
#   - 228 真人藍牙錄音 (10-29)：說話段 RMS 約 -36～-34 dBFS；錄音尾端無人說話段 -75～-60 dBFS，
#     auto 會在其中幻覺出「I'm going to get this.」「yes The the the」。-55 dBFS 與說話段仍有約 20 dB 餘裕。
SILENCE_RMS = 10 ** (-55 / 20)
# 自動偵測偶爾把中文誤判成日／韓文；輸出含假名或諺文就改用 zh 重跑一次
_UNEXPECTED_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af\u1100-\u11ff]")


def is_near_silent(audio: "np.ndarray") -> bool:
    return audio.size == 0 or float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < SILENCE_RMS


def needs_zh_retry(text: str) -> bool:
    return bool(text) and _UNEXPECTED_SCRIPT_RE.search(text) is not None


# 語音偵測 (VAD)：用 faster-whisper 內建的 Silero VAD (CPU 備援路徑本來就以 vad_filter 使用)，找不到語音的切片不送 NPU。
#   PI5_WHISPER_VAD=0 關閉；faster-whisper 未安裝時只用上面的 RMS 門檻。
# 2026-10-29 228 喇叭→藍牙實錄：PipeWire 1.6.9 後錄音電平約高 10 dB，無人說話時底噪約 -49 dBFS，
#   高於 -55 dBFS 門檻，auto 會幻覺出「I'm going to kill you.」之類英文句子並寫進紀要。
#   Silero (預設 threshold 0.5、最短語音 250 ms)：純底噪 13/13 段擋下；真人錄音只擋結尾無人說話段；
#   測試會議壓到 -50 dBFS 再疊真實底噪 (SNR 約 0 dB) 仍 0 段誤擋；每 8 s 切片約 30 ms CPU (單執行緒)。
WHISPER_VAD = os.environ.get("PI5_WHISPER_VAD", "1").strip() != "0"
_VAD_OPTIONS = None


def speech_seconds(audio: "np.ndarray", sr: int = 16000) -> Optional[float]:
    """Silero VAD 偵測到的語音總秒數；VAD 關閉、未安裝或出錯時回傳 None (呼叫端照常送 NPU)。"""
    global _VAD_OPTIONS
    if not WHISPER_VAD:
        return None
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception:
        return None
    try:
        if _VAD_OPTIONS is None:
            _VAD_OPTIONS = VadOptions(min_speech_duration_ms=250)
        spans = get_speech_timestamps(np.asarray(audio, dtype=np.float32), vad_options=_VAD_OPTIONS, sampling_rate=sr)
    except Exception as e:
        logger.warning(f"[HailoWhisper] VAD failed, sending the slice anyway: {e}")
        return None
    return sum(s["end"] - s["start"] for s in spans) / float(sr)

# 會議結構詞的系統性同音錯字 (2026-10-30 統計 228 的 51 份逐字稿，約 25.5 萬字)：
#   「X方面，決議是…」：決意 72、捷一 8、決一 3，正確的「決議」只有 4 次。
#   「待辦事項是…」：待辦事像 14、代辦事像 10、代版事像 7、代辦事向 6…，正確的只有 9 次。
# 長會議紀要以「決議」等關鍵字挑出決議條目，錯字會讓決議段落變成「無」(30 分鐘喇叭自測)。
# 只修正後面接「是」的決議，以及「待辦」系列後接「事X」的待辦事項，不動其他用法。
# 10-30 228 再查 61 份逐字稿：還有「待辦事是」(漏了「項」) 6 次、「戴版是向／待辦是向」4 次、「待辦事想」3 次、
# 「待辦職像／待半時像」各 1 次，一併收進來。「事X」照舊直接改；「是X／職X／時X」和單獨的「事」
# 必須後面接著「是」(「待辦事項是…」的句型) 才改，避免「代辦是向國稅局申請」這類正常用法被改掉。
_DECISION_FIX_RE = re.compile(r"(?:決意|捷一|決一|絕一|覺一)(?=\s*是)")
_TODO_FIX_RE = re.compile(r"(?:待辦|代辦|代版|戴板|戴版|待半|大辦|帶辦)\s*"
                          r"(?:事\s*[項像相象向下巷想]|(?:[是職時]\s*[項像相象向下巷想]|事)(?=\s*是))")


def fix_meeting_terms(text: str) -> str:
    if not text:
        return text or ""
    return _TODO_FIX_RE.sub("待辦事項", _DECISION_FIX_RE.sub("決議", text))


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

    def __init__(self, hef_path: Optional[str] = None, group_id: Optional[str] = None):
        self.hef_path = hef_path
        # group_id：與 hailo-ollama (HAILO_OLLAMA_VDEVICE_GROUP_ID) 相同時兩個行程可共用 Hailo-10H；None = 獨占 VDevice
        self.group_id = group_id or None
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

            if self.group_id:
                params = VDevice.create_params()
                params.group_id = self.group_id
                self._vdevice = VDevice(params)
            else:
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
        repetition_penalty: Optional[float] = None,
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

            if (audio_data is None or len(audio_data) < 1600 or is_near_silent(audio_data)
                    or speech_seconds(audio_data) == 0.0):
                return {
                    "text": "",
                    "language": language or "en",
                    "inference_time_ms": 0.0,
                    "backend": "hailo-10h",
                }

            from hailo_platform.genai import Speech2TextTask
            # 預設 "zh" (中文多位元組 token 解碼較乾淨)；"auto" → None 交給模型自動偵測語言
            lang = (language or WHISPER_LANGUAGE).lower()
            lang_code = None if lang == "auto" else lang
            rp = WHISPER_REPETITION_PENALTY if repetition_penalty is None else float(repetition_penalty)

            # Calculate dynamic timeout: minimum 30s, or 1.5x audio duration + 10s
            audio_dur_sec = len(audio_data) / 16000.0
            calc_timeout_ms = max(30000, int((audio_dur_sec * 1.5 + 10.0) * 1000))

            gen_kwargs: Dict[str, Any] = {
                "audio_data": audio_data,
                "task": Speech2TextTask.TRANSCRIBE,
                "language": lang_code,
                "timeout_ms": calc_timeout_ms,
            }
            if rp != 1.0:   # HailoRT < 5.2.0 沒有這個參數；1.0 時不傳以維持相容
                gen_kwargs["repetition_penalty"] = rp
            raw_text = self._s2t.generate_all_text(**gen_kwargs)
            if lang_code is None and needs_zh_retry(raw_text or ""):
                logger.info(f"[HailoWhisper] auto language produced kana/hangul, retry with zh: {raw_text!r}")
                gen_kwargs["language"] = lang_code = "zh"
                raw_text = self._s2t.generate_all_text(**gen_kwargs)

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
            final_text = fix_meeting_terms(final_text)

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
