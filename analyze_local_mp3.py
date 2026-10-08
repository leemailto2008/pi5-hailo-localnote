# f:\12_prj_raspi5\analyze_local_mp3.py
import subprocess
import numpy as np

cmd = ['ffmpeg', '-i', r'f:\12_prj_raspi5\debug_user.mp3', '-f', 'f32le', '-ac', '1', '-ar', '16000', 'pipe:1']
proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
data = np.frombuffer(proc.stdout, dtype=np.float32)
duration = len(data) / 16000.0
print(f"Total Duration: {duration:.2f}s ({len(data)} samples)")

print("Interval     | RMS      | Peak     | Clip%   | HF Noise Ratio | ZCR")
print("-" * 65)

for sec in range(0, int(duration), 2):
    chunk = data[sec*16000 : min(len(data), (sec+2)*16000)]
    rms = float(np.sqrt(np.mean(chunk**2)))
    peak = float(np.max(np.abs(chunk)))
    clip_cnt = float(np.sum(np.abs(chunk) > 0.98)) / len(chunk)
    diff = np.diff(chunk)
    hf_ratio = float(np.sqrt(np.mean(diff**2))) / max(1e-5, rms)
    zcr = float(np.sum(np.diff(np.sign(chunk)) != 0)) / (2 * len(chunk))
    print(f"{sec:02d}s - {sec+2:02d}s   | {rms:<8.4f} | {peak:<8.4f} | {clip_cnt:<7.2%} | {hf_ratio:<14.2f} | {zcr:<8.4f}")
