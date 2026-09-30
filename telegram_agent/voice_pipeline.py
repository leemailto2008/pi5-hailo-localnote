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


try:
    from opencc import OpenCC
    _opencc_converter = OpenCC('s2twp')
except Exception:
    _opencc_converter = None


def to_taiwan_traditional(text: str) -> str:
    """Normalize Chinese text to Taiwan Traditional standard using OpenCC."""
    if not text:
        return ""
    if _opencc_converter:
        try:
            return _opencc_converter.convert(text)
        except Exception:
            pass
    return text


def is_silence_or_empty_audio(wav_path: str, rms_threshold: int = 80) -> bool:
    """
    Check if a WAV file contains only silence or background static.
    Uses struct to calculate RMS, compatible with Python 3.13 (no audioop dependency).
    """
    try:
        import wave
        import struct
        import math
        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return True
            frames = wf.readframes(n_frames)
            count = len(frames) // 2
            if count == 0:
                return True
            shorts = struct.unpack(f"<{count}h", frames)
            sum_sq = sum(s * s for s in shorts)
            rms = int(math.sqrt(sum_sq / count))
            if rms < rms_threshold:
                print(f"[VoicePipeline] Audio RMS too low ({rms} < {rms_threshold}), detected as silence.")
                return True
            return False
    except Exception as e:
        print(f"[VoicePipeline] Error in silence check: {e}")
        return False


def convert_oga_to_wav(input_oga_path: str, output_wav_path: str) -> bool:
    """
    Convert Telegram voice/audio to 16kHz mono PCM WAV with audio loudness normalization.
    Uses highpass filter + dynaudnorm to boost quiet speech and suppress low-frequency hum.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_oga_path,
            "-af", "highpass=f=70,lowpass=f=7600,dynaudnorm=f=75:g=15:m=10.0,volume=2.0",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
        if res.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 100:
            return True

        # Fallback attempt: Standard PCM conversion
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", input_oga_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav_path
        ]
        res_fb = subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return res_fb.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 100
    except Exception as e:
        print(f"[VoicePipeline] ffmpeg conversion failed: {e}")
        return False


def speech_to_text(audio_path: str) -> Optional[str]:
    """
    Transcribe speech from an audio file to Traditional Chinese text.
    Uses robust parameters compatible with faster-whisper on ARM NEON.
    """
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        print(f"[VoicePipeline] Audio file missing or empty: {audio_path}")
        return None

    base_name = os.path.splitext(audio_path)[0]
    wav_path = f"{base_name}_temp.wav"
    
    if not convert_oga_to_wav(audio_path, wav_path):
        print(f"[VoicePipeline] Audio conversion to WAV failed for: {audio_path}")
        return None

    try:
        # Pre-check: If audio is pure silence/static, discard
        if is_silence_or_empty_audio(wav_path):
            print(f"[VoicePipeline] Audio is silence, skipping Whisper transcription.")
            return None

        model = get_whisper_model()
        if model is None:
            print("[VoicePipeline] Whisper model unavailable.")
            return None

        clean_prompt = "以下是清晰的繁體中文日常語音對話。"

        text = ""
        # Pass 1: Silero VAD filter with relaxed settings
        try:
            segments, info = model.transcribe(
                wav_path,
                beam_size=5,
                language="zh",
                initial_prompt=clean_prompt,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(threshold=0.25, min_silence_duration_ms=400)
            )
            text = "".join([segment.text for segment in segments]).strip()
        except Exception as vad_err:
            print(f"[VoicePipeline] Pass 1 (VAD) exception: {vad_err}")

        # Pass 2: Fallback without VAD if empty
        if not text:
            print("[VoicePipeline] Pass 1 empty. Running Pass 2 (VAD disabled fallback)...")
            segments, info = model.transcribe(
                wav_path,
                beam_size=5,
                language="zh",
                initial_prompt=clean_prompt,
                condition_on_previous_text=False,
                vad_filter=False
            )
            text = "".join([segment.text for segment in segments]).strip()

        # Sanitize known Whisper hallucination substrings
        hallucinations = [
            "字幕by", "字幕提供", "請不吝點贊", "點贊訂閱", "謝謝大家", "轉載請註明",
            "感謝收看", "索兰娅", "點點欄目", "明鏡", "優質內容", "歡迎關注", "歡迎訂閱"
        ]
        for h in hallucinations:
            text = text.replace(h, "")
        text = text.strip()

        # Normalize to pure Taiwan Traditional Chinese
        if text:
            text = to_taiwan_traditional(text)
            print(f"[VoicePipeline] Transcribed text: '{text}' (Audio size: {os.path.getsize(audio_path)} bytes)")
            return text
        else:
            print(f"[VoicePipeline] Empty transcription after both passes. Audio size: {os.path.getsize(audio_path)} bytes")
            return None

    except Exception as e:
        print(f"[VoicePipeline] Transcription failed: {e}")
        return None
    finally:
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


async def record_audio_from_mic(duration_sec: int, output_path: str) -> bool:
    """
    Record audio from default microphone (PipeWire Bluetooth headset or onboard) using ffmpeg.
    Encodes as Opus audio in OGG container (native Telegram voice format).
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "pulse",
        "-i", "default",
        "-t", str(duration_sec),
        "-ar", "16000",
        "-ac", "1",
        "-af", "volume=2.2,highpass=f=75,lowpass=f=7500,dynaudnorm=f=75:g=15:m=8.0",
        "-c:a", "libopus",
        "-b:a", "64k",
        output_path
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.communicate()
        return proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 500
    except Exception as e:
        print(f"[VoicePipeline] Microphone recording failed: {e}")
        return False

