# f:\13_prj_raspi5_localnote\benchmarks\benchmark_latest_user_speech.py
import os
import sys
import time
import numpy as np

sys.path.extend(['/home/pi/code_dispatcher', '/home/pi/telegram_agent'])
import voice_pipeline
from voice_memory import audio_to_pcm_array

ogg_path = '/home/pi/benchmark_dataset/latest_user_speech_3min.ogg'
print(f"載入使用者最新 3 分鐘實測語音: {ogg_path}...")
full_pcm = audio_to_pcm_array(ogg_path)
total_samples = len(full_pcm)
total_audio_sec = total_samples / 16000.0
print(f"總時長: {total_audio_sec:.1f} 秒 ({total_samples} samples)")

chunk_sec = 8.0
chunk_samples = int(chunk_sec * 16000)
num_chunks = int(np.ceil(total_samples / chunk_samples))

print(f"\n--- 模擬 {chunk_sec:.1f}s 切片邊錄邊轉即時串流 (模型: small) ---")
segments = []
worker_ready_time = 0.0
chunk_latencies = []
skipped_silence = 0

for i in range(num_chunks):
    slice_arrival_time = (i + 1) * chunk_sec
    s = i * chunk_samples
    e = min(total_samples, (i + 1) * chunk_samples)
    pcm_slice = full_pcm[s:e]
    
    # 模擬背景 worker 處理時機
    worker_start = max(slice_arrival_time, worker_ready_time)
    
    # VAD 靜音快篩
    peak = float(np.max(np.abs(pcm_slice)))
    rms = float(np.sqrt(np.mean(pcm_slice**2)))
    if peak < 0.035 and rms < 0.0025:
        skipped_silence += 1
        worker_ready_time = worker_start + 0.001
        print(f"[Chunk {i+1:02d}/{num_chunks}] 抵達: {slice_arrival_time:5.1f}s | (VAD 靜音快篩跳過)")
        continue
        
    t0 = time.time()
    seg_text = voice_pipeline.speech_to_text(pcm_slice)
    infer_dur = time.time() - t0
    chunk_latencies.append(infer_dur)
    
    worker_ready_time = worker_start + infer_dur
    queue_backlog = max(0.0, worker_ready_time - slice_arrival_time)
    
    if seg_text and seg_text.strip():
        segments.append(seg_text.strip())
        print(f"[Chunk {i+1:02d}/{num_chunks}] 抵達: {slice_arrival_time:5.1f}s | 推論: {infer_dur:4.2f}s | 佇列堆疊: {queue_backlog:4.2f}s | 辨識: {seg_text.strip()[:28]}...")
    else:
        print(f"[Chunk {i+1:02d}/{num_chunks}] 抵達: {slice_arrival_time:5.1f}s | 推論: {infer_dur:4.2f}s | 佇列堆疊: {queue_backlog:4.2f}s | (無人聲)")

total_wall_clock = max(total_audio_sec, worker_ready_time)
tail_latency = max(0.0, total_wall_clock - total_audio_sec)
full_transcript = "，".join(segments)

passed = total_wall_clock <= 210.0
conclusion = "🚀 極速達標 (PASS)" if passed else "❌ 未達標 (FAIL)"

print("\n" + "=" * 60)
print("📊 使用者最新實測語音 評測結算報告")
print("=" * 60)
print(f"音訊長度: {total_audio_sec:.1f} 秒")
print(f"切片長度: {chunk_sec:.1f} 秒 (總切片數: {num_chunks})")
if chunk_latencies:
    print(f"平均單片推論耗時: {np.mean(chunk_latencies):.2f} 秒 (RTF: {np.mean(chunk_latencies)/chunk_sec:.2f})")
print(f"跳過純靜音切片: {skipped_silence} 片")
print(f"錄音結束後尾巴等待時間 (Tail Latency): {tail_latency:.2f} 秒")
print(f"看到 STT 全部完成的總時長: {total_wall_clock:.2f} 秒 (目標門檻: <= 210.0 秒)")
print(f"驗收結論: {conclusion}")
print(f"轉錄字數: {len(full_transcript)} 字")
print(f"前 150 字預覽:\n{full_transcript[:150]}...")
print("=" * 60)
