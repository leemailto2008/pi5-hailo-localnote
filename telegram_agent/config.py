# f:\12_prj_raspi5\telegram_agent\config.py
"""
Configuration settings for Raspberry Pi 5 Telegram Voice AI Agent.
Handles environment variables, default models, hardware thresholds, and whitelist security.
"""

import os
from typing import List

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Database & Memory storage
MEMORY_DB_PATH = os.path.join(DATA_DIR, "memory.db")

# Telegram Bot Credentials & Access Control
# Set via environment variable or update below
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Whitelist of allowed Telegram User/Chat IDs (integers)
# Set your Telegram user ID to block unauthorized public access
ALLOWED_CHAT_IDS_ENV = os.getenv("ALLOWED_CHAT_IDS", "")
ALLOWED_CHAT_IDS: List[int] = (
    [int(x.strip()) for x in ALLOWED_CHAT_IDS_ENV.split(",") if x.strip().isdigit()]
    if ALLOWED_CHAT_IDS_ENV
    else []
)

# Ollama Engine Endpoints & Models
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434")

# Optimized models on Pi 5
DEFAULT_CHAT_MODEL = "qwen2.5:3b-opt"      # Top-tier Traditional Chinese, ~6 tok/s
DEFAULT_REASONING_MODEL = "deepseek-r1:1.5b-opt" # Rapid reasoning, ~12 tok/s
DEFAULT_CODER_MODEL = "qwen2.5-coder:7b-opt"    # In-depth code review, ~2.8 tok/s
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"    # 768-dim vector embeddings, ~15ms

# Three-Tier Memory Parameters
SHORT_TERM_MEMORY_MAX_ROUNDS = 8  # Max user/assistant pairs in working memory
VECTOR_TOP_K = 3                  # Number of semantic memories to retrieve
VECTOR_SIMILARITY_THRESHOLD = 0.55 # Minimum cosine similarity to inject into prompt

# Voice Pipeline Configurations
# Speech-To-Text (STT)
WHISPER_MODEL_SIZE = "base"       # 'base' or 'small' (quantized int8)
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"     # Optimized for ARM NEON SIMD

# Text-To-Speech (TTS)
# Default Microsoft Neural Voice for Traditional Chinese (Taiwan)
DEFAULT_TTS_VOICE = "zh-TW-HsiaoChenNeural" # Female voice: 曉臻
ALT_TTS_VOICE = "zh-TW-YunJheNeural"        # Male voice: 雲哲

# Hardware Telemetry Thresholds
CPU_TEMP_WARN_THRESHOLD_C = 70.0  # Temperature warning trigger
