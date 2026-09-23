#!/usr/bin/env python3
import sys
import os
import subprocess
import time

sys.path.insert(0, '/home/pi/pi5-llm-arena/telegram_agent')
from voice_pipeline import get_whisper_model, convert_oga_to_wav, speech_to_text
from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, DATA_DIR

print("=== 1. Checking Whisper Model Initialization ===")
t0 = time.time()
model = get_whisper_model()
t1 = time.time()
print(f"Model loaded: {model is not None} in {t1 - t0:.2f}s")
print(f"Config: size={WHISPER_MODEL_SIZE}, device={WHISPER_DEVICE}, compute_type={WHISPER_COMPUTE_TYPE}")

print("\n=== 2. Checking ffmpeg Availability ===")
res = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
print("ffmpeg path:", res.stdout.strip())

print("\n=== 3. Generating Synthetic WAV & Running STT Benchmark ===")
test_wav = os.path.join(DATA_DIR, "test_synth.wav")
# Generate 3 seconds 16kHz sine wave audio
subprocess.run([
    "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
    "-ar", "16000", "-ac", "1", test_wav
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if os.path.exists(test_wav):
    print("Test WAV generated successfully:", test_wav)
    try:
        segments, info = model.transcribe(test_wav, language="zh")
        text = "".join([s.text for s in segments]).strip()
        print(f"STT on sine wave completed without exception. Result text: '{text}' (Expected blank/noise)")
    except Exception as e:
        print("STT failed on sine wave:", e)
    finally:
        if os.path.exists(test_wav):
            os.remove(test_wav)

print("\n=== 4. Checking Existing OGG / Audio in Data Dir ===")
for f in os.listdir(DATA_DIR):
    if f.endswith((".oga", ".ogg", ".wav")):
        p = os.path.join(DATA_DIR, f)
        print(f"Found existing audio: {f} ({os.path.getsize(p)} bytes)")
        if f.endswith(".ogg") and "test" in f:
            print(f"Testing transcription on {f}...")
            res_text = speech_to_text(p)
            print(f"Transcription result: '{res_text}'")

print("\n=== 5. STT Diagnostics Completed ===")
