import sys
sys.path.insert(0, '/home/pi/telegram_agent')
sys.path.insert(0, '/home/pi/code_dispatcher')
from voice_pipeline import speech_to_text

audio_path = '/dev/shm/audio_cache/m_1792375879.ogg'
print(f"Transcribing {audio_path}...")
text = speech_to_text(audio_path)
print("=== TRANSCRIPTION RESULT ===")
print(text[:500] if text else "(Empty)")
print(f"Total length: {len(text)} chars")
