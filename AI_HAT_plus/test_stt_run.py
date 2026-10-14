#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import time

sys.path.insert(0, "/home/pi/code_dispatcher")
sys.path.append("/home/pi/telegram_agent")
from voice_pipeline import speech_to_text

audio_path = "/dev/shm/audio_cache/m_1791970272.ogg"
if not os.path.exists(audio_path):
    print(f"Audio file not found: {audio_path}")
    sys.exit(1)

print(f"Testing transcription on: {audio_path} (Size: {os.path.getsize(audio_path)} bytes)")
t0 = time.time()
text = speech_to_text(audio_path)
t_elapsed = time.time() - t0

print("=" * 50)
print(f"⏱️ 轉錄總耗時: {t_elapsed:.2f} 秒")
print("📝 轉錄文字片段:")
print(text[:200] if text else "None")
print("=" * 50)
