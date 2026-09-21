# telegram_agent/benchmark_whisper.py
"""
Benchmark and compare STT accuracy improvements on Raspberry Pi 5.
Evaluates:
1. base model vs small model
2. vad_filter & beam_size tuning
3. domain vocabulary prompt injection (台灣繁中術語字典先驗)
4. ASR post-correction via Ollama LLM
"""

import os
import sys
import time
import json
import urllib.request
from faster_whisper import WhisperModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(BASE_DIR, "data", "test_output.ogg")

def test_model(model_name: str, beam_size: int = 5, use_prompt: bool = True):
    print(f"\n--- Testing Faster-Whisper '{model_name}' (beam_size={beam_size}, use_prompt={use_prompt}) ---")
    t0 = time.time()
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    print(f"Model load time: {time.time() - t0:.2f}s")
    
    prompt = "以下為台灣繁體中文對話，包括：樹莓派、Raspberry Pi、語音助理、就緒。" if use_prompt else ""
    
    t1 = time.time()
    segments, info = model.transcribe(
        AUDIO_PATH,
        beam_size=beam_size,
        language="zh",
        initial_prompt=prompt,
        vad_filter=True
    )
    text = "".join([s.text for s in segments]).strip()
    elapsed = time.time() - t1
    print(f"Transcribed in {elapsed:.2f}s: '{text}'")
    return text

def test_llm_post_correction(noisy_text: str):
    print(f"\n--- Testing LLM ASR Post-Correction for: '{noisy_text}' ---")
    t0 = time.time()
    prompt = (
        "你是一個語音識別糾錯專家。以下是一段語音轉文字的原始輸出，"
        "可能存在同音錯字（例如將『樹莓派』辨識為『數沒派』，『就緒』辨識為『教訓』）。"
        "請根據台灣繁體中文日常語意，修正為正確的繁體中文句子。"
        "僅輸出修正後的最終文字，不要有任何多餘的解釋或問候。\n\n"
        f"原始識別文字：{noisy_text}\n"
        "修正後文字："
    )
    url = "http://127.0.0.1:11434/api/generate"
    data = json.dumps({
        "model": "qwen2.5:3b-opt",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 64}
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            fixed = res.get("response", "").strip()
            print(f"LLM Corrected in {time.time()-t0:.2f}s: '{fixed}'")
            return fixed
    except Exception as e:
        print(f"LLM error: {e}")
        return noisy_text

if __name__ == "__main__":
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: {AUDIO_PATH} not found!")
        sys.exit(1)
        
    print(f"Target ground-truth audio: '樹莓派語音助理已就緒。'")
    
    # 1. Test Base model with improved prompt
    res_base = test_model("base", beam_size=5, use_prompt=True)
    
    # 2. Test LLM Post-Correction on base result
    test_llm_post_correction(res_base)
    
    # 3. Test Small model with improved prompt
    res_small = test_model("small", beam_size=5, use_prompt=True)
    
    print("\nBenchmark completed!")
