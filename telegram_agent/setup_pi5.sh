#!/usr/bin/env bash
# ==============================================================================
# Setup script for Raspberry Pi 5 Telegram Voice AI Agent
# ==============================================================================
set -e

echo "=== [1/3] Verifying ffmpeg and audio libraries ==="
which ffmpeg || echo "ffmpeg already available at /usr/bin/ffmpeg"

echo "=== [2/4] Installing Python AI Agent dependencies ==="
pip3 install -r /home/pi/pi5-llm-arena/telegram_agent/requirements.txt --break-system-packages

echo "=== [3/4] Verifying Python modules ==="
python3 -c "
import telegram
import edge_tts
import numpy
import requests
print('✓ telegram, edge_tts, numpy, requests imported successfully!')
"

echo "=== [4/4] Environment verification completed! ==="
