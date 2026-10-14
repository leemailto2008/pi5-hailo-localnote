# -*- coding: utf-8 -*-
"""
AI_HAT_plus/hailo_engine/__init__.py
Raspberry Pi AI HAT+ (Hailo-8 / Hailo-10H) 加速引擎套件
"""

from .device_manager import HailoDeviceManager, is_hailo_available
from .whisper_adapter import (
    BaseWhisperEngine,
    CPUWhisperEngine,
    HailoWhisperEngine,
    get_whisper_engine
)
from .audio_preprocessor import load_audio_pcm, log_mel_spectrogram

__all__ = [
    "HailoDeviceManager",
    "is_hailo_available",
    "BaseWhisperEngine",
    "CPUWhisperEngine",
    "HailoWhisperEngine",
    "get_whisper_engine",
    "load_audio_pcm",
    "log_mel_spectrogram"
]
