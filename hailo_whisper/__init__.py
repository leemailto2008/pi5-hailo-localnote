# -*- coding: utf-8 -*-
# hailo_whisper/__init__.py
"""
Hailo-10H NPU Whisper Acceleration Module.
Provides high-performance, low-latency, and zero-CPU-contention ASR inference
for Raspberry Pi 5 edge devices.
"""

from .engine import get_whisper_engine, WhisperEngineInterface
from .audio_processor import AudioProcessor
from .tokenizer import WhisperTokenizer

__all__ = ["get_whisper_engine", "WhisperEngineInterface", "AudioProcessor", "WhisperTokenizer"]
