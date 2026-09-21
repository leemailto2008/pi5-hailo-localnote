# f:\12_prj_raspi5\telegram_agent\voice_pipeline.py
"""
Bi-directional Voice Pipeline for Raspberry Pi 5:
- Speech-To-Text (STT): faster-whisper (ARM NEON int8) + ffmpeg audio preprocessing
- Text-To-Speech (TTS): edge-tts (Microsoft Neural Taiwan Traditional Chinese: 曉臻 / 雲哲)
"""

import os
import asyncio
import subprocess
from typing import Optional

from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    DEFAULT_TTS_VOICE,
    DATA_DIR
)

# Global faster-whisper model cache (lazy-loaded to save RAM until first audio)
_whisper_model = None


def get_whisper_model():
    """Lazy load faster-whisper model into memory."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            print(f"[VoicePipeline] Initializing faster-whisper ({WHISPER_MODEL_SIZE}, {WHISPER_COMPUTE_TYPE})...")
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=4
            )
            print("[VoicePipeline] faster-whisper model loaded successfully.")
        except Exception as e:
            print(f"[VoicePipeline] Error loading faster-whisper: {e}")
            _whisper_model = False
    return _whisper_model if _whisper_model is not False else None


def convert_oga_to_wav(input_oga_path: str, output_wav_path: str) -> bool:
    """
    Convert Telegram .oga (Opus) audio to 16kHz mono PCM WAV using ffmpeg.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_oga_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return res.returncode == 0
    except Exception as e:
        print(f"[VoicePipeline] ffmpeg conversion failed: {e}")
        return False


def speech_to_text(audio_oga_path: str) -> Optional[str]:
    """
    Transcribe speech from a Telegram audio file to Traditional Chinese text.
    """
    wav_path = audio_oga_path.replace(".oga", ".wav").replace(".ogg", ".wav")
    if not convert_oga_to_wav(audio_oga_path, wav_path):
        return None

    try:
        model = get_whisper_model()
        if model is None:
            print("[VoicePipeline] Whisper model unavailable.")
            return None

        # Transcribe with language hint for Traditional Chinese
        segments, info = model.transcribe(
            wav_path,
            beam_size=3,
            language="zh",
            initial_prompt="以下是繁體中文語音輸入，請以繁體中文轉錄。"
        )
        text = "".join([segment.text for segment in segments]).strip()
        return text if text else None
    except Exception as e:
        print(f"[VoicePipeline] Transcription failed: {e}")
        return None
    finally:
        # Clean up intermediate WAV file
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass


async def text_to_speech_async(text: str, output_ogg_path: Optional[str] = None, voice: str = DEFAULT_TTS_VOICE) -> bool:
    """
    Synthesize text into natural Traditional Chinese neural voice using edge-tts.
    Outputs as OGG/Opus compatible with Telegram Voice Message.
    """
    if not text.strip():
        return False
    if output_ogg_path is None:
        import uuid
        output_ogg_path = os.path.join(TEMP_AUDIO_DIR, f"tts_{uuid.uuid4().hex[:8]}.ogg")
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_ogg_path)
        return os.path.exists(output_ogg_path) and os.path.getsize(output_ogg_path) > 0
    except Exception as e:
        print(f"[VoicePipeline] edge-tts synthesis failed: {e}")
        return False


def text_to_speech(text: str, output_ogg_path: str, voice: str = DEFAULT_TTS_VOICE) -> bool:
    """Synchronous wrapper for text_to_speech_async."""
    try:
        return asyncio.run(text_to_speech_async(text, output_ogg_path, voice))
    except Exception as e:
        print(f"[VoicePipeline] Synchronous TTS failed: {e}")
        return False
