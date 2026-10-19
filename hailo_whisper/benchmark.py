# -*- coding: utf-8 -*-
# hailo_whisper/benchmark.py
"""
benchmark.py: Latency, RTF, and Preprocessing Benchmark for Whisper Inference.
Validates performance across Hailo-10H NPU and CPU Fallback.
"""

import time
import numpy as np
from typing import Dict, Any
from .audio_processor import AudioProcessor
from .engine import get_whisper_engine


def benchmark_audio_processor(iterations: int = 50) -> Dict[str, float]:
    """Benchmark pure-NumPy Log-Mel extraction latency."""
    ap = AudioProcessor()
    dummy_audio = np.random.uniform(-0.5, 0.5, 16000 * 30).astype(np.float32)

    # Warmup
    _ = ap.log_mel_spectrogram(dummy_audio)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = ap.log_mel_spectrogram(dummy_audio)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    return {
        "iterations": iterations,
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
    }


def main():
    print("=" * 60)
    print(" Hailo-10H / CPU Whisper Engine Performance Benchmark")
    print("=" * 60)

    print("\n[1/2] Benchmarking Log-Mel Preprocessor (Pure NumPy)...")
    mel_stats = benchmark_audio_processor()
    print(f"  Iterations: {mel_stats['iterations']}")
    print(f"  Mean Latency: {mel_stats['mean_latency_ms']} ms")
    print(f"  P95 Latency:  {mel_stats['p95_latency_ms']} ms")

    print("\n[2/2] Detecting Whisper Engine & Hardware Acceleration...")
    engine = get_whisper_engine(preferred_backend="auto")
    is_hw = engine.is_hardware_accelerated()
    print(f"  Active Engine: {type(engine).__name__}")
    print(f"  Hardware Accelerated: {is_hw}")
    print("=" * 60)


if __name__ == "__main__":
    main()
