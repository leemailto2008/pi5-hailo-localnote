# code_dispatcher/two_layer_audio.py
"""
Task 32: Two-Layer Ring Buffer & Core-Isolated Voice Pipeline for Raspberry Pi 5.
- Layer 1: Real-time hardware capture to POSIX RAM Ring Buffer (Locked to CPU 0, SCHED_FIFO).
- Layer 2: Throttled chunk-by-chunk collector and inference (Locked to CPU 2,3, cpu_threads=2).
- Layer 3: Instant waveform peaks pre-computation for 0-latency WaveSurfer rendering.
"""

import os
import sys
import time
import asyncio
import subprocess
from typing import Dict, Any, Optional, List, Callable, AsyncGenerator

import numpy as np


def extract_waveform_peaks(audio_path: str, num_peaks: int = 200) -> List[float]:
    """
    Extract normalized waveform peaks array directly in Python/ffmpeg.
    Takes ~0.05s on Raspberry Pi 5.
    Returned peaks list is fed directly to WaveSurfer for 0-latency instant rendering.
    """
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return [0.0] * num_peaks

    # Downsample audio to 2000Hz mono PCM to minimize compute
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "2000",
        "-f", "s16le",
        "-"
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_pcm, _ = proc.communicate()
        if not raw_pcm or len(raw_pcm) < 4:
            return [0.0] * num_peaks

        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        n_samples = len(samples)
        if n_samples <= num_peaks:
            return [round(float(abs(s)), 3) for s in samples]

        # Aggregate peaks per block
        block_size = n_samples // num_peaks
        peaks = []
        for i in range(num_peaks):
            block = samples[i * block_size : (i + 1) * block_size]
            if len(block) > 0:
                peak_val = float(np.max(np.abs(block)))
            else:
                peak_val = 0.0
            peaks.append(round(peak_val, 3))

        return peaks
    except Exception as e:
        print(f"[TwoLayerAudio] Error extracting waveform peaks: {e}")
        return [0.0] * num_peaks


def set_process_core_affinity(pid: int, cores: List[int]) -> bool:
    """
    Bind specific Linux process to designated CPU cores using os.sched_setaffinity or taskset.
    """
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(pid, set(cores))
            return True
        else:
            core_str = ",".join(str(c) for c in cores)
            res = subprocess.run(["taskset", "-cp", core_str, str(pid)], capture_output=True, timeout=2)
            return res.returncode == 0
    except Exception as e:
        print(f"[TwoLayerAudio] Failed to set CPU affinity for PID {pid}: {e}")
        return False


def set_process_realtime_priority(pid: int, priority: int = 80) -> bool:
    """
    Promote process to Linux SCHED_FIFO real-time scheduler with given priority (1-99).
    Requires sudo or CAP_SYS_NICE capability.
    """
    try:
        res = subprocess.run(["sudo", "chrt", "-f", "-p", str(priority), str(pid)], capture_output=True, timeout=2)
        return res.returncode == 0
    except Exception:
        return False


class TwoLayerAudioPipe:
    """
    Two-layer decoupled audio recording & chunk streaming engine.
    - Captures to continuous master file on /dev/shm.
    - Concurrently streams 16kHz PCM blocks for real-time VU and Layer 2 chunk extraction.
    """

    def __init__(
        self,
        duration_sec: int,
        output_master_ogg: str,
        chunk_interval_sec: float = 8.0,
        cancellation_token: Optional[asyncio.Event] = None
    ):
        self.duration_sec = duration_sec
        self.output_master_ogg = output_master_ogg
        self.chunk_interval_sec = chunk_interval_sec
        self.cancellation_token = cancellation_token or asyncio.Event()

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._pcm_buffer: bytearray = bytearray()
        self._chunk_queue: asyncio.Queue = asyncio.Queue()
        self._is_active: bool = False

    async def start(
        self,
        source_node: str,
        on_chunk_ready: Optional[Callable[[int, str, float], Any]] = None,
        vu_callback: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> bool:
        """
        Start hardware recording on CPU 0 and Layer 2 chunk slicing worker.
        """
        self._is_active = True
        sample_rate = 16000
        bytes_per_sec = sample_rate * 2  # 16-bit mono = 32,000 bytes/sec

        # Dual output ffmpeg command:
        # Output 1: High quality master OGG file with highpass/lowpass filters
        # Output 2: Raw PCM stdout pipe for real-time VU & Layer 2 chunking
        cmd = [
            "ffmpeg", "-y",
            "-thread_queue_size", "4096",
            "-f", "pulse",
            "-i", source_node,
            "-t", str(self.duration_sec),
            "-af", "aresample=16000:async=1,highpass=f=80,lowpass=f=7600",
            "-c:a", "libopus",
            "-b:a", "64k",
            self.output_master_ogg,
            "-t", str(self.duration_sec),
            "-f", "s16le",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            "pipe:1"
        ]

        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )

            # Apply Layer 1 Hardware Guard: Bind to CPU 0
            if self._proc.pid:
                set_process_core_affinity(self._proc.pid, [0])
                set_process_realtime_priority(self._proc.pid, 80)

            # Spawn reader and chunk distributor task
            asyncio.create_task(
                self._pcm_reader_loop(bytes_per_sec, on_chunk_ready, vu_callback)
            )
            return True

        except Exception as e:
            print(f"[TwoLayerAudio] Failed to spawn ffmpeg: {e}")
            self._is_active = False
            return False

    async def _pcm_reader_loop(
        self,
        bytes_per_sec: int,
        on_chunk_ready: Optional[Callable[[int, str, float], Any]],
        vu_callback: Optional[Callable[[Dict[str, Any]], Any]]
    ):
        """Read 50ms frames from ffmpeg stdout and slice into chunks."""
        if not self._proc or not self._proc.stdout:
            return

        chunk_bytes_target = int(bytes_per_sec * self.chunk_interval_sec)
        frame_bytes = 1600  # 50ms @ 16kHz 16-bit mono
        chunk_idx = 0
        current_chunk_buffer = bytearray()
        chunk_start_time = 0.0

        band_slices = [
            (4, 13), (13, 25), (25, 50), (50, 100),
            (100, 175), (175, 250), (250, 325), (325, 380)
        ]

        try:
            while self._is_active and not self.cancellation_token.is_set():
                data = await self._proc.stdout.read(frame_bytes)
                if not data:
                    break

                current_chunk_buffer.extend(data)

                # VU Meter calculation
                if vu_callback:
                    try:
                        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
                        peak = float(np.max(np.abs(samples))) / 32768.0 if len(samples) > 0 else 0.0
                        rms_db = 20.0 * np.log10(max(1e-5, rms / 32768.0))
                        level_pct = min(100, max(0, int((rms_db + 50.0) * (100.0 / 45.0)))) if rms_db > -50.0 else 0

                        fft_spec = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
                        bands = []
                        for low, high in band_slices:
                            energy = float(np.mean(fft_spec[low:high])) if high <= len(fft_spec) else 0.0
                            b_db = 20.0 * np.log10(max(1.0, energy))
                            b_pct = min(100, max(0, int((b_db - 30.0) * (100.0 / 50.0))))
                            bands.append(b_pct)

                        await vu_callback({
                            "rms": round(rms / 32768.0, 4),
                            "peak": round(peak, 4),
                            "db": round(rms_db, 1),
                            "level_pct": level_pct,
                            "bands": bands,
                            "active": True
                        })
                    except Exception:
                        pass

                # When chunk threshold reached, slice into RAM chunk file
                if len(current_chunk_buffer) >= chunk_bytes_target:
                    chunk_duration = len(current_chunk_buffer) / bytes_per_sec
                    chunk_path = f"/dev/shm/audio_cache/chunk_{chunk_idx:04d}.wav"
                    if not os.path.exists("/dev/shm"):
                        chunk_path = os.path.join(os.path.dirname(self.output_master_ogg), f"chunk_{chunk_idx:04d}.wav")

                    self._write_pcm_wav(chunk_path, bytes(current_chunk_buffer))
                    chunk_idx += 1
                    current_chunk_buffer.clear()

                    if on_chunk_ready:
                        try:
                            res = on_chunk_ready(chunk_idx - 1, chunk_path, chunk_duration)
                            if asyncio.iscoroutine(res):
                                asyncio.create_task(res)
                        except Exception as e:
                            print(f"[TwoLayerAudio] Error in on_chunk_ready callback: {e}")

            # Flush remaining tail buffer as final chunk if length > 1.5 seconds
            if len(current_chunk_buffer) >= (bytes_per_sec * 1.5):
                chunk_duration = len(current_chunk_buffer) / bytes_per_sec
                chunk_path = f"/dev/shm/audio_cache/chunk_{chunk_idx:04d}.wav"
                self._write_pcm_wav(chunk_path, bytes(current_chunk_buffer))
                if on_chunk_ready:
                    try:
                        res = on_chunk_ready(chunk_idx, chunk_path, chunk_duration)
                        if asyncio.iscoroutine(res):
                            asyncio.create_task(res)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[TwoLayerAudio] Reader loop exception: {e}")
        finally:
            self._is_active = False

    def _write_pcm_wav(self, output_wav: str, pcm_data: bytes):
        """Write raw 16kHz mono PCM directly to standard 44-byte WAV header."""
        import struct
        sample_rate = 16000
        bits_per_sample = 16
        channels = 1
        byte_rate = sample_rate * channels * (bits_per_sample // 8)
        block_align = channels * (bits_per_sample // 8)
        data_size = len(pcm_data)

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,
            1,               # PCM format
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        try:
            with open(output_wav, "wb") as f:
                f.write(header)
                f.write(pcm_data)
        except Exception as e:
            print(f"[TwoLayerAudio] Failed to write WAV chunk {output_wav}: {e}")

    async def wait_until_done(self):
        """Wait for ffmpeg master recording to exit."""
        if self._proc:
            await self._proc.wait()
        self._is_active = False

    def abort(self):
        """Cleanly abort ffmpeg recording process."""
        self._is_active = False
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass


_isolated_whisper_model = None

def get_isolated_whisper_model():
    """Lazy load a 2-thread Whisper model specifically throttled for concurrent chunk inference."""
    global _isolated_whisper_model
    if _isolated_whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
            _isolated_whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=2
            )
        except Exception as e:
            print(f"[TwoLayerAudio] Error loading isolated Whisper model: {e}")
            _isolated_whisper_model = False
    return _isolated_whisper_model if _isolated_whisper_model is not False else None


def transcribe_chunk_isolated(chunk_wav_path: str) -> str:
    """
    Safely transcribe an audio chunk with CPU affinity locked to cores 2, 3
    and max 2 threads to prevent starving Layer 1 UART audio capture.
    """
    if not os.path.exists(chunk_wav_path) or os.path.getsize(chunk_wav_path) < 1000:
        return ""

    # Lock current worker thread to CPU cores 2 and 3 if supported
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {2, 3})
    except Exception:
        pass

    model = get_isolated_whisper_model()
    if not model:
        from voice_pipeline import speech_to_text
        return speech_to_text(chunk_wav_path)

    try:
        from voice_pipeline import to_taiwan_traditional, clean_hallucination_repetitions
        segments, _ = model.transcribe(
            chunk_wav_path,
            language="zh",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            hallucination_silence_threshold=2.0,
            beam_size=1
        )
        texts = [s.text.strip() for s in segments if s.text.strip()]
        combined = clean_hallucination_repetitions("".join(texts))
        return to_taiwan_traditional(combined)
    except Exception as e:
        print(f"[TwoLayerAudio] Error transcribing isolated chunk: {e}")
        return ""
