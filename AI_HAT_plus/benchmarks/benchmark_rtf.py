# -*- coding: utf-8 -*-
"""
AI_HAT_plus/benchmarks/benchmark_rtf.py
CPU vs Hailo-8 NPU 語音轉錄即時因子 (Real-Time Factor, RTF) 效能基準評估工具
"""

import os
import sys
import time
import numpy as np
import soundfile as sf

# 將根目錄加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hailo_engine import get_whisper_engine, is_hailo_available

def generate_dummy_audio(duration_sec: int, filepath: str):
    """產生指定長度的 16kHz 單聲道測試音訊"""
    sr = 16000
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # 合成 440Hz 簡諧波與輕微白噪音
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.02 * np.random.normal(0, 1, len(t))
    sf.write(filepath, audio.astype(np.float32), sr)

def run_benchmark():
    print("=" * 80)
    print(" 🎙️ 樹莓派 5 語音轉錄效能基準測試 (STT RTF Benchmark)")
    print("=" * 80)

    hailo_present = is_hailo_available()
    print(f"硬體加速狀態: {'✅ 探測到 Hailo-8 NPU' if hailo_present else 'ℹ️ 純 CPU 模式 (未插入 AI HAT+)'}")

    test_durations = [30, 60, 180]
    temp_dir = os.path.join(os.path.dirname(__file__), "tmp_audio")
    os.makedirs(temp_dir, exist_ok=True)

    print("-" * 80)
    print(f"{'音訊時長':<12} | {'推論後端':<32} | {'轉錄耗時(秒)':<14} | {'即時因子 (RTF)':<14}")
    print("-" * 80)

    for dur in test_durations:
        audio_file = os.path.join(temp_dir, f"test_{dur}s.wav")
        if not os.path.exists(audio_file):
            generate_dummy_audio(dur, audio_file)

        # 測試 CPU 引擎
        engine_cpu = get_whisper_engine(auto_detect=False, model_size="base")
        try:
            res_cpu = engine_cpu.transcribe(audio_file)
            print(f"{dur} 秒 ({dur//60}分鐘) | {res_cpu['engine']:<32} | {res_cpu['duration_sec']:<14.2f} | {res_cpu['rtf']:<14.3f}")
        except Exception as e:
            print(f"{dur} 秒 ({dur//60}分鐘) | CPU 推論失敗: {e}")

        # 若有 Hailo 則測試 Hailo 引擎
        if hailo_present:
            engine_hailo = get_whisper_engine(auto_detect=True)
            res_hailo = engine_hailo.transcribe(audio_file)
            print(f"{dur} 秒 ({dur//60}分鐘) | {res_hailo['engine']:<32} | {res_hailo['duration_sec']:<14.2f} | {res_hailo['rtf']:<14.3f}")

    print("-" * 80)
    print("說明：RTF < 0.1 代表音訊轉錄速度為即時播放的 10 倍以上（數值越低越極速）。\n")

if __name__ == "__main__":
    run_benchmark()
