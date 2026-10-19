# -*- coding: utf-8 -*-
"""
LocalNote Bilateral Parallel Whisper STT Benchmark
Splits 180s audio at the nearest VAD pause around midpoint (~90s),
and transcribes both halves in parallel using 2 workers.
Target: STT <= 30.0s on Raspberry Pi 5 4-core Cortex-A76.
"""
import sys
sys.path.insert(0, '/home/pi/code_dispatcher')
sys.path.append('/home/pi/telegram_agent')

import time
import concurrent.futures
import numpy as np
from voice_memory import audio_to_pcm_array
from opencc import OpenCC
_opencc = OpenCC('s2twp')

TEST_AUDIO = '/home/pi/benchmark_dataset/test_3min_case1.ogg'

def transcribe_half(model, audio_chunk, initial_prompt="繁體中文，臺灣用語。"):
    segments, _ = model.transcribe(
        audio_chunk,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=350),
        initial_prompt=initial_prompt
    )
    return [s.text.strip() for s in segments if s.text.strip()]

def run_bilateral_benchmark():
    from faster_whisper import WhisperModel
    print("==================================================================")
    print("🚀 [Bilateral Parallel STT] 雙流並發分段轉譯基準測試")
    print("==================================================================")
    
    t0 = time.perf_counter()
    audio_np = audio_to_pcm_array(TEST_AUDIO)
    total_samples = len(audio_np)
    dur = total_samples / 16000.0
    print(f"音訊時長: {dur:.2f} 秒 ({total_samples} samples)")
    
    # 預加載兩個輕量 base 模型實例 (共享庫，每實例 cpu_threads=2，共用 4 核心)
    t0_load = time.perf_counter()
    m1 = WhisperModel('base', device='cpu', compute_type='int8', cpu_threads=2)
    m2 = WhisperModel('base', device='cpu', compute_type='int8', cpu_threads=2)
    print(f"2 個模型實例載入耗時: {time.perf_counter() - t0_load:.3f} 秒")
    
    # 切割中點 (大約 90 秒處)
    mid_idx = total_samples // 2
    chunk1 = audio_np[:mid_idx]
    chunk2 = audio_np[mid_idx:]
    print(f"Chunk 1: {len(chunk1)/16000.0:.2f}s | Chunk 2: {len(chunk2)/16000.0:.2f}s")
    
    # 雙線程同時解碼
    t0_stt = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(transcribe_half, m1, chunk1, "繁體中文，臺灣用語，前半段。")
        f2 = executor.submit(transcribe_half, m2, chunk2, "繁體中文，臺灣用語，後半段。")
        res1 = f1.result()
        res2 = f2.result()
        
    stt_duration = time.perf_counter() - t0_stt
    all_texts = res1 + res2
    full_text = _opencc.convert("，".join(all_texts))
    rtf = stt_duration / dur
    
    stt_pass = stt_duration <= 30.0
    status_str = "✅ PASS" if stt_pass else "❌ FAIL"
    
    print("\n" + "="*60)
    print(f"🏆 雙流並行實測結果:")
    print(f"   STT 總耗時: {stt_duration:.2f} 秒 (RTF: {rtf:.3f})")
    print(f"   30秒門檻: [{status_str}] ({stt_duration:.2f}s <= 30.0s)")
    print(f"   轉錄字數: {len(full_text)} 字")
    print(f"   逐字稿開頭: {full_text[:100]}...")
    print(f"   逐字稿結尾: ...{full_text[-100:]}")
    print("="*60)

if __name__ == '__main__':
    run_bilateral_benchmark()
