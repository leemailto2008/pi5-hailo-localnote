# -*- coding: utf-8 -*-
# hailo_whisper/audio_processor.py
"""
audio_processor.py: Lightweight Log-Mel Spectrogram Extractor for Whisper.
Optimized for Raspberry Pi 5 edge execution with zero heavy dependencies (no PyTorch/Librosa required).
Strictly adheres to Whisper specifications (16kHz, 80 mel bins, 400 FFT, 160 hop).
"""

import math
import numpy as np
from typing import Tuple, Optional


class AudioProcessor:
    """
    High-performance, pure-NumPy audio processor for Whisper model input.
    Generates 80-channel log-mel spectrogram features suitable for Hailo-10H NPU DMA transfer.
    """

    SAMPLE_RATE: int = 16000
    N_FFT: int = 400
    HOP_LENGTH: int = 160
    CHUNK_LENGTH: int = 30  # Whisper standard context length in seconds
    N_MELS: int = 80

    def __init__(self, n_mels: int = 80):
        self.n_mels = n_mels
        self._mel_filters = self._build_mel_filters(
            n_mels=self.n_mels,
            n_fft=self.N_FFT,
            sample_rate=self.SAMPLE_RATE,
        )

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Convert frequency in Hz to Mel scale."""
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        """Convert Mel scale value back to Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filters(self, n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
        """
        Build standard Slaney-style Mel filterbank matrix (n_mels, 1 + n_fft // 2).
        """
        f_min = 0.0
        f_max = sample_rate / 2.0
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)

        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)

        filters = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
        for i in range(n_mels):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            if center > left:
                filters[i, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                filters[i, center:right] = (right - np.arange(center, right)) / (right - center)

        # Slaney normalization
        enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
        filters *= enorm[:, np.newaxis]
        return filters

    def log_mel_spectrogram(self, audio: np.ndarray, padding: bool = True) -> np.ndarray:
        """
        Compute Log-Mel Spectrogram from raw 16kHz float32 audio array.

        Args:
            audio: 1D NumPy array with audio samples in range [-1.0, 1.0].
            padding: If True, zero-pad or truncate to standard 30-second window (480,000 samples).

        Returns:
            np.ndarray: Log-Mel features of shape (80, 3000) for standard 30s chunk.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio array, got shape {audio.shape}")

        target_length = self.CHUNK_LENGTH * self.SAMPLE_RATE
        if padding:
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
            elif len(audio) > target_length:
                audio = audio[:target_length]

        # STFT via Hanning window
        window = np.hanning(self.N_FFT)
        num_frames = (len(audio) - self.N_FFT) // self.HOP_LENGTH + 1

        # Vectorized frame extraction
        indices = (
            np.tile(np.arange(0, self.N_FFT), (num_frames, 1))
            + np.tile(np.arange(0, num_frames * self.HOP_LENGTH, self.HOP_LENGTH), (self.N_FFT, 1)).T
        )
        frames = audio[indices] * window

        # RFFT & Power spectrum (frames, 1 + n_fft // 2)
        stft = np.fft.rfft(frames, n=self.N_FFT)
        # Transpose to (freq, frames) -> (201, frames)
        magnitudes = (np.abs(stft).T) ** 2

        # Whisper standard: slice time frames to exactly 3000 frames
        if magnitudes.shape[1] > 3000:
            magnitudes = magnitudes[:, :3000]
        elif magnitudes.shape[1] < 3000:
            magnitudes = np.pad(magnitudes, ((0, 0), (0, 3000 - magnitudes.shape[1])), mode="constant")

        # Mel filterbank dot product: (80, 201) @ (201, 3000) -> (80, 3000)
        mel_spec = np.dot(self._mel_filters, magnitudes)

        # Log compression
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec.astype(np.float32)
