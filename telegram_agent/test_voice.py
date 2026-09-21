# telegram_agent/test_voice.py
import asyncio
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from telegram_agent.voice_pipeline import text_to_speech, text_to_speech_async, speech_to_text
from telegram_agent.config import TELEGRAM_BOT_TOKEN
import httpx

async def main():
    print(f"1. Checking Telegram Bot Token (length: {len(TELEGRAM_BOT_TOKEN)})...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe")
        print(f"Telegram API getMe status: {resp.status_code}")
        if resp.status_code == 200:
            bot_info = resp.json()
            print(f"✓ Bot Connected: @{bot_info['result']['username']} ({bot_info['result']['first_name']})")
        else:
            print(f"✗ Failed getMe: {resp.text}")

    print("\n2. Testing Edge-TTS Text to Speech...")
    test_text = "樹莓派語音助理已就緒。"
    output_ogg = os.path.join(PARENT_DIR, "telegram_agent", "data", "test_output.ogg")
    tts_ok = await text_to_speech_async(test_text, output_ogg)
    if tts_ok and os.path.exists(output_ogg):
        print(f"✓ TTS Audio Generated: {output_ogg} ({os.path.getsize(output_ogg)} bytes)")
        
        print("\n3. Testing Faster-Whisper Speech to Text...")
        recognized = speech_to_text(output_ogg)
        print(f"✓ STT Recognized Content: '{recognized}'")
    else:
        print("✗ TTS failed to generate audio")

if __name__ == "__main__":
    asyncio.run(main())
