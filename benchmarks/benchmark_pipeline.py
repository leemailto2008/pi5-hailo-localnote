# -*- coding: utf-8 -*-
"""
LocalNote STT & Meeting Minutes Automated Benchmark Harness (Iteration 4 - Pure Speed)
Optimizations:
1. without_timestamps=True (Skips cross-attention DTW alignment, saves 15-20% latency!)
2. temperature=0.0, condition_on_previous_text=False
3. vad_parameters min_silence=400ms, speech_pad=150ms
Target: STT <= 30.0s, Total <= 60.0s for 180s audio.
"""
import sys
sys.path.insert(0, '/home/pi/code_dispatcher')
sys.path.append('/home/pi/telegram_agent')

import time
import requests
import numpy as np
from voice_memory import audio_to_pcm_array
from opencc import OpenCC
_opencc = OpenCC('s2twp')

TEST_AUDIO = '/home/pi/benchmark_dataset/test_3min_case1.ogg'

def run_pure_speed_test(model_size='base', without_timestamps=True):
    from faster_whisper import WhisperModel
    print("==================================================================")
    print(f"🚀 [Speed Optimization] Model={model_size}, WithoutTimestamps={without_timestamps}")
    print("==================================================================")
    
    t0 = time.perf_counter()
    audio_np = audio_to_pcm_array(TEST_AUDIO)
    dur = len(audio_np) / 16000.0
    pcm_time = time.perf_counter() - t0
    
    # STT
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device='cpu', compute_type='int8', cpu_threads=4)
    segments, info = model.transcribe(
        audio_np,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        without_timestamps=without_timestamps,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=150),
        initial_prompt="繁體中文，臺灣用語，語音筆記。"
    )
    texts = [s.text.strip() for s in segments if s.text.strip()]
    stt_time = time.perf_counter() - t0
    full_text = _opencc.convert("，".join(texts))
    rtf = stt_time / dur if dur > 0 else 0
    stt_pass = stt_time <= 30.0
    stt_status = "✅ PASS" if stt_pass else "❌ FAIL"
    
    print(f"1. STT 耗時: {stt_time:.2f} 秒 (RTF: {rtf:.3f}) | 門檻 (<=30s): [{stt_status}]")
    print(f"   字數: {len(full_text)} 字")
    print(f"   文字預覽: {full_text[:120]}...")
    
    # 2. Fast LLM Meeting Minutes
    t0 = time.perf_counter()
    llm_prompt = f"""請以繁體中文濃縮以下會議重點（100字內）：
【逐字稿】：{full_text[:800]}
格式：
- 會議摘要：
- 待辦事項："""
    
    llm_time = 0.0
    try:
        resp = requests.post(
            'http://127.0.0.1:11434/api/generate',
            json={
                'model': 'qwen2.5:3b-opt',
                'prompt': llm_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'num_predict': 110,
                    'threads': 4
                }
            },
            timeout=30
        )
        if resp.status_code == 200:
            llm_res = resp.json().get('response', '')
            llm_time = time.perf_counter() - t0
            print(f"2. LLM 耗時: {llm_time:.2f} 秒")
        else:
            llm_time = 99.0
    except Exception as e:
        print(f"2. LLM 請求失敗: {e}")
        llm_time = 99.0

    total_time = pcm_time + stt_time + llm_time
    total_pass = total_time <= 60.0
    total_status = "✅ PASS" if total_pass else "❌ FAIL"
    print(f"3. 總耗時 (不含錄音): {total_time:.2f} 秒 | 門檻 (<=60s): [{total_status}]")
    print("==================================================================")
    
    return {
        'stt_time': stt_time,
        'llm_time': llm_time,
        'total_time': total_time,
        'stt_pass': stt_pass,
        'total_pass': total_pass
    }

if __name__ == '__main__':
    run_pure_speed_test(model_size='base', without_timestamps=True)
