import sys
sys.path.insert(0, '/home/pi/code_dispatcher')
import time
from faster_whisper.vad import get_vad_model, get_speech_timestamps, VadOptions
from voice_memory import audio_to_pcm_array

audio_np = audio_to_pcm_array('/home/pi/benchmark_dataset/test_3min_case1.ogg')
print('原始長度:', len(audio_np)/16000.0, '秒')

t0 = time.perf_counter()
vad_model = get_vad_model()
opts = VadOptions(threshold=0.45, min_silence_duration_ms=350, speech_pad_ms=200)
timestamps = get_speech_timestamps(audio_np, vad_options=opts)
vad_time = time.perf_counter() - t0
print(f'VAD 耗時: {vad_time:.3f} 秒, 偵測到有效語音區間數: {len(timestamps)}')

active_samples = sum(c['end'] - c['start'] for c in timestamps)
saved_sec = (len(audio_np) - active_samples) / 16000.0
print(f'純語音長度: {active_samples / 16000.0:.2f} 秒 (剔除靜音: {saved_sec:.2f} 秒，有效壓縮率: {saved_sec/180.0*100:.1f}%)')
