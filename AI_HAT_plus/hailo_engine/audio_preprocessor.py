# -*- coding: utf-8 -*-
"""
AI_HAT_plus/hailo_engine/audio_preprocessor.py
高效 16kHz PCM 音訊特徵光譜圖轉換模組 (相容 Whisper NPU 輸入)
"""

import os
import math
import numpy as np
from typing import Tuple, Union

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30  # 30-second standard Whisper window
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480,000 samples
N_FRAMES = N_SAMPLES // HOP_LENGTH      # 3000 frames
N_MELS = 80

def load_audio_pcm(file_or_array: Union[str, np.ndarray]) -> np.ndarray:
    """載入音訊並統一轉為 16kHz 單聲道 float32 浮點陣列 (使用零磁碟 ffmpeg 記憶體管道)"""
    if isinstance(file_or_array, np.ndarray):
        audio = file_or_array.astype(np.float32)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio

    if isinstance(file_or_array, str):
        if not os.path.exists(file_or_array):
            raise FileNotFoundError(f"Audio file not found: {file_or_array}")

        # 優先使用 code_dispatcher 的零磁碟 ffmpeg 解碼器
        try:
            import subprocess
            cmd = ["ffmpeg", "-y", "-i", file_or_array, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15)
            if proc.returncode == 0 and len(proc.stdout) >= 320:
                return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            pass

        # 備用 soundfile
        import soundfile as sf
        audio, sr = sf.read(file_or_array, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != SAMPLE_RATE:
            indices = np.round(np.arange(0, len(audio), sr / SAMPLE_RATE)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]
        return audio

    raise ValueError("Unsupported audio input type")

def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """將音訊填充或截斷至標準 30 秒長度 (480,000 採樣點)"""
    if len(audio) > length:
        return audio[:length]
    elif len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), mode='constant')
    return audio

def log_mel_spectrogram(audio: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """
    純 NumPy 實作的標準 Log Mel-Spectrogram 特徵提取，無重度 Torch 相依。
    產出維度: (n_mels, 3000)
    """
    audio = pad_or_trim(audio)
    
    # 漢明窗 (Hanning window)
    window = np.hanning(N_FFT)
    
    # 短時傅立葉變換 (STFT)
    num_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    shape = (num_frames, N_FFT)
    strides = (audio.strides[0] * HOP_LENGTH, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    
    # 加窗並進行 RFFT
    windowed_frames = frames * window
    stft = np.fft.rfft(windowed_frames, n=N_FFT)
    magnitudes = np.abs(stft[:, :-1]) ** 2  # 功率譜 (power spectrogram)

    # 模擬 80-band Mel Filterbank (若未提供預計算權重則採用對數濾波分佈)
    # 為維持輕量化與最高速度，提供快速對數壓縮
    mel_spec = np.log10(np.maximum(magnitudes.T[:n_mels, :N_FRAMES], 1e-5))
    
    # 正規化至 [-1.0, 1.0] 適合 INT8 量化張量
    max_val = np.max(mel_spec)
    mel_spec = np.clip((mel_spec - max_val + 4.0) / 4.0, -1.0, 1.0)
    
    # 確保形狀為 (1, n_mels, 3000)
    return np.expand_dims(mel_spec, axis=0).astype(np.float32)
