# f:\13_prj_raspi5_localnote\benchmarks\loop_engineer_stt.py
"""
Loop Engineering STT Optimizer for Raspberry Pi 5.
Goal: Ingest real 3-minute voice benchmark dataset (/home/pi/benchmark_dataset/test_3min_case1.ogg),
systematically search optimal chunk windows (4s, 6s, 8s, 10s) and VAD filtering,
and ensure the entire pipeline (Recording + STT Completion) completes strictly within 210.0 seconds.
"""

import os
import sys
import time
import numpy as np

# Ensure path to dispatcher modules
sys.path.extend(['/home/pi/code_dispatcher', '/home/pi/telegram_agent'])
import voice_pipeline
from voice_memory import audio_to_pcm_array

# 載入 3 分鐘測試音訊基準庫
ogg_path = '/home/pi/benchmark_dataset/test_3min_case1.ogg'
if not os.path.exists(ogg_path):
    print(f"Error: dataset file not found: {ogg_path}")
    sys.exit(1)

full_pcm = audio_to_pcm_array(ogg_path)
total_samples = len(full_pcm)
total_audio_sec = total_samples / 16000.0

print("=" * 65)
print("🚀 LocalNote STT Loop Engineering 自我調校與極限壓測")
print(f"音訊基準庫: {ogg_path} ({total_audio_sec:.1f}s, {total_samples} samples)")
print("核心驗收約束: 含 3 分鐘錄音及看到 STT 完成，總耗時 <= 210.0 秒")
print("=" * 65 + "\n")

# 預熱 Whisper small 模型
print("[LoopEngineer] 預熱 Whisper small 模型...")
_ = voice_pipeline.get_whisper_model()
print("[LoopEngineer] 模型預熱完成！開始執行多輪超參數迭代...\n")

experiments = [
    {"name": "Exp-1 (Chunk=4.0s)", "chunk_sec": 4.0, "vad_skip": False},
    {"name": "Exp-2 (Chunk=6.0s)", "chunk_sec": 6.0, "vad_skip": False},
    {"name": "Exp-3 (Chunk=8.0s)", "chunk_sec": 8.0, "vad_skip": False},
    {"name": "Exp-4 (Chunk=10.0s)", "chunk_sec": 10.0, "vad_skip": False},
    {"name": "Exp-5 (Chunk=6.0s + VAD Pre-skip)", "chunk_sec": 6.0, "vad_skip": True},
]

results_summary = []

for exp in experiments:
    chunk_sec = exp["chunk_sec"]
    use_vad = exp["vad_skip"]
    name = exp["name"]
    chunk_samples = int(chunk_sec * 16000)
    num_chunks = int(np.ceil(total_samples / chunk_samples))
    
    print("-" * 65)
    print(f"▶ 執行實驗: {name} | 總切片數: {num_chunks}")
    print("-" * 65)
    
    current_worker_time = 0.0
    chunk_durations = []
    skipped_chunks = 0
    segments = []
    
    for i in range(num_chunks):
        slice_ready_time = (i + 1) * chunk_sec
        s = i * chunk_samples
        e = min(total_samples, (i + 1) * chunk_samples)
        pcm_slice = full_pcm[s:e]
        
        # worker 開始推論的時間取決於：切片產出時間 vs worker 前一個切片推論完成時間
        worker_start_time = max(slice_ready_time, current_worker_time)
        
        # 判斷是否 VAD 跳過純靜音
        is_silence = False
        if use_vad:
            peak = float(np.max(np.abs(pcm_slice)))
            rms = float(np.sqrt(np.mean(pcm_slice**2)))
            if peak < 0.04 and rms < 0.003:
                is_silence = True
                skipped_chunks += 1
        
        t0 = time.time()
        if is_silence:
            text = None
            infer_dur = 0.001
        else:
            text = voice_pipeline.speech_to_text(pcm_slice)
            infer_dur = time.time() - t0
        
        chunk_durations.append(infer_dur)
        current_worker_time = worker_start_time + infer_dur
        
        if text and text.strip():
            segments.append(text.strip())
        
        queue_backlog = max(0.0, current_worker_time - slice_ready_time)
        print(f"  [Chunk {i+1:02d}/{num_chunks}] 產出點: {slice_ready_time:5.1f}s | 推論: {infer_dur:4.2f}s | 佇列堆疊: {queue_backlog:4.2f}s")

    total_pipeline_time = max(total_audio_sec, current_worker_time)
    tail_latency = max(0.0, total_pipeline_time - total_audio_sec)
    full_transcript = "，".join(segments)
    
    passed = total_pipeline_time <= 210.0
    status_str = "✅ 達標 (PASS)" if passed else "❌ 未達標 (FAIL)"
    
    print(f"\n>>> 實驗結果: {name}")
    print(f"    - 錄音結束後尾巴延遲 (Tail Latency): {tail_latency:.2f} 秒")
    print(f"    - 總耗時 (含 3 分鐘錄音 + STT 出稿): {total_pipeline_time:.2f} 秒 (門檻: <= 210.0s) -> {status_str}")
    print(f"    - 逐字稿總字數: {len(full_transcript)} 字 | 跳過靜音切片: {skipped_chunks} 個")
    print("-" * 65 + "\n")
    
    results_summary.append({
        "name": name,
        "chunk_sec": chunk_sec,
        "total_time": total_pipeline_time,
        "tail_latency": tail_latency,
        "chars": len(full_transcript),
        "passed": passed
    })

print("\n" + "=" * 75)
print("📊 LOOP ENGINEERING 實驗綜合排行榜 (LEADERBOARD)")
print("=" * 75)
print(f"{'實驗名稱':<32} | {'總耗時 (<=210s)':<14} | {'尾端等待':<10} | {'總字數':<8} | {'判定'}")
print("-" * 75)
for r in results_summary:
    mark = "✅ PASS" if r["passed"] else "❌ FAIL"
    print(f"{r['name']:<32} | {r['total_time']:>8.2f} 秒    | {r['tail_latency']:>6.2f} 秒 | {r['chars']:>6} 字 | {mark}")
print("=" * 75)
