import sys
import time
sys.path.insert(0, '/home/pi/telegram_agent')
sys.path.insert(0, '/home/pi/code_dispatcher')
from voice_pipeline import speech_to_text

audio_path = '/dev/shm/audio_cache/m_1792548384.ogg'
print(f"Transcribing {audio_path}...")
t0 = time.time()
text = speech_to_text(audio_path)
elapsed = time.time() - t0
print(f"=== FULL MASTER TRANSCRIPTION RESULT (took {elapsed:.2f}s) ===")
print(text)
print(f"Total length: {len(text) if text else 0} chars")
