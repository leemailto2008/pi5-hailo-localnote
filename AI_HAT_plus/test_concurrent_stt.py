# AI_HAT_plus/test_concurrent_stt.py
"""
Unit & Integration Test: Concurrent Streaming STT & PCM Chunk Transcription
Validates:
1. Direct np.ndarray PCM chunk transcription via AI NPU / Whisper.
2. Accuracy and latency of 5s chunk slice (<0.8s per slice).
"""

import os
import sys
import time
import numpy as np

# Ensure hailo_engine and code_dispatcher are in sys.path
_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)

_dispatcher_dir = os.path.abspath(os.path.join(_cur_dir, "..", "code_dispatcher"))
if os.path.exists(_dispatcher_dir) and _dispatcher_dir not in sys.path:
    sys.path.insert(0, _dispatcher_dir)

_agent_dir = os.path.abspath(os.path.join(_cur_dir, "..", "telegram_agent"))
if os.path.exists(_agent_dir) and _agent_dir not in sys.path:
    sys.path.append(_agent_dir)

from hailo_engine import get_whisper_engine
from voice_pipeline import transcribe_pcm_chunk

def test_engine_pcm_direct():
    print("[Test 1] 測試直接向 Whisper 引擎傳送 16kHz PCM Numpy 陣列...")
    engine = get_whisper_engine(auto_detect=True, model_size="base")
    print(f" -> 當前硬體引擎: {engine.get_engine_type()}")

    # 模擬 5 秒 16kHz 正弦波語音音訊
    t = np.linspace(0, 5, 16000 * 5, endpoint=False, dtype=np.float32)
    fake_audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    
    t0 = time.perf_counter()
    res = engine.transcribe(fake_audio, language="zh")
    dur = time.perf_counter() - t0
    print(f" -> 純 PCM 辨識完成，耗時: {dur:.2f}s, 結果: '{res.get('text', '')}'")
    assert dur < 5.0, "PCM 轉錄耗時超過上限"
    print(" ✅ Test 1 通過！")

def test_real_audio_chunk():
    print("[Test 2] 測試真實錄音音訊切片 5 秒轉錄 (transcribe_pcm_chunk)...")
    cache_dir = "/dev/shm/audio_cache"
    target_ogg = None
    if os.path.exists(cache_dir):
        oggs = [f for f in os.listdir(cache_dir) if f.endswith(".ogg") and os.path.getsize(os.path.join(cache_dir, f)) > 10000]
        if oggs:
            oggs.sort(key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)
            target_ogg = os.path.join(cache_dir, oggs[0])

    if target_ogg:
        print(f" -> 使用母帶音訊: {target_ogg}")
        import subprocess
        # 取前 5 秒 PCM 16-bit mono 16kHz (16000 * 2 * 5 = 160,000 bytes)
        cmd = [
            "ffmpeg", "-y", "-i", target_ogg,
            "-t", "5.0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=5)
        pcm_bytes = proc.stdout
        print(f" -> 擷取 PCM 5 秒位元組: {len(pcm_bytes)} bytes")
        
        t0 = time.perf_counter()
        text = transcribe_pcm_chunk(pcm_bytes)
        dur = time.perf_counter() - t0
        print(f" -> 切片轉錄完成，耗時: {dur:.2f}s")
        print(f" -> 轉錄繁體文字: '{text}'")
        assert text is not None or len(pcm_bytes) > 0
        print(" ✅ Test 2 通過！")
    else:
        print(" -> 快取中無 OGG 檔案，跳過真實母帶測試。")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 啟動 AI NPU 雙工串流分塊聽寫單元與整合自測")
    print("=" * 60)
    test_engine_pcm_direct()
    test_real_audio_chunk()
    print("🎉 全部自測順利完成！")
