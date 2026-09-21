# f:\12_prj_raspi5\telegram_agent\agent.py
"""
Raspberry Pi 5 Telegram Voice AI Agent Core Daemon.
Integrates python-telegram-bot, Three-Tier Memory, Voice Pipeline, and Ollama Engine.
"""

import os
import sys

# Ensure telegram_agent directory is in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import time
import uuid
import logging
import subprocess
from typing import Dict, Any

import requests
from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from config import (
    TELEGRAM_BOT_TOKEN,
    ALLOWED_CHAT_IDS,
    OLLAMA_API_URL,
    DEFAULT_CHAT_MODEL,
    DATA_DIR
)
from memory import ThreeTierMemoryManager
from voice_pipeline import speech_to_text, text_to_speech_async
from tools import execute_tool_call_if_needed

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("Pi5VoiceAgent")

# Memory manager instance
memory_mgr = ThreeTierMemoryManager()

# Global state for active model per chat
active_models: Dict[int, str] = {}


def is_authorized(user_id: int) -> bool:
    """Check if the user is in the authorized whitelist."""
    if not ALLOWED_CHAT_IDS:
        # If no whitelist specified, allow but log warning
        return True
    return user_id in ALLOWED_CHAT_IDS


def get_active_model(chat_id: int) -> str:
    """Get selected Ollama model for the chat."""
    return active_models.get(chat_id, DEFAULT_CHAT_MODEL)


def get_hardware_telemetry() -> Dict[str, Any]:
    """Retrieve Pi 5 real-time CPU temperature and available RAM."""
    temp_c = 0.0
    free_ram_mb = 0
    total_ram_mb = 0
    try:
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2)
        temp_c = float(res.stdout.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        temp_c = 45.0  # Fallback

    try:
        res = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=2)
        for line in res.stdout.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                total_ram_mb = int(parts[1])
                free_ram_mb = int(parts[6])
    except Exception:
        pass

    return {
        "cpu_temp": temp_c,
        "free_ram": free_ram_mb,
        "total_ram": total_ram_mb
    }


# =============================================================================
# Command Handlers
# =============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user = update.effective_user
    if not is_authorized(user.id):
        await update.message.reply_text(f"⚠️ 未授權存取。您的 Telegram User ID 為: `{user.id}`，請聯繫管理員加入白名單。", parse_mode=ParseMode.MARKDOWN)
        return

    msg = (
        f"👋 您好 {user.first_name}！我是您的 **Raspberry Pi 5 邊緣語音 AI 助理**。\n\n"
        f"🧠 **當前推論大腦:** `{get_active_model(user.id)}`\n"
        f"🏛️ **三層記憶金字塔:** 已啟用 (即時緩衝 + 個人畫像 + 語意向量庫)\n"
        f"🎙️ **語音互動:** 直接發送語音訊息，我會用語音回覆您！\n\n"
        f"常用指令：\n"
        f"/status - 檢視樹莓派 5 核心溫度與記憶體硬體遙測\n"
        f"/model <名稱> - 切換 AI 模型 (如 qwen2.5:3b-opt, deepseek-r1:1.5b-opt)\n"
        f"/remember <文字> - 手動將重要資訊記入長期向量庫\n"
        f"/facts - 查看我為您記住的個人事實特徵\n"
        f"/clear - 重置當前短期對話記憶"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    if not is_authorized(update.effective_user.id):
        return
    help_text = (
        "📖 **Raspberry Pi 5 AI 助理操作指南**\n\n"
        "1. **語音互動 (Voice Chat):** 直接按住錄音鍵發送語音訊息，系統將自動進行語音轉錄 $\\to$ 思考 $\\to$ 繁體中文語音回覆。\n"
        "2. **文字對話 (Text Chat):** 直接輸入文字進行問答與代碼審查。\n"
        "3. **長期記憶 (Long-Term Memory):**\n"
        "   - `/remember <內容>`: 主動記憶重要事情\n"
        "   - `/facts`: 列出個人檔案特徵\n"
        "   - `/clear`: 清空短期上下文\n"
        "4. **模型切換 (Switch Models):**\n"
        "   - `/model qwen2.5:3b-opt` (繁中最佳，6 tok/s)\n"
        "   - `/model deepseek-r1:1.5b-opt` (極速思考，12 tok/s)\n"
        "   - `/model qwen2.5-coder:7b-opt` (深度代碼審查)\n"
        "5. **硬體監控 (Hardware Telemetry):**\n"
        "   - `/status`: 查看 CPU 溫度、可用 RAM 與佇列"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command: display Pi 5 hardware metrics."""
    if not is_authorized(update.effective_user.id):
        return

    telem = get_hardware_telemetry()
    chat_id = update.effective_user.id
    current_model = get_active_model(chat_id)
    working_turns = len(memory_mgr.get_working_memory(chat_id)) // 2
    facts_count = len(memory_mgr.get_user_facts(chat_id))

    status_msg = (
        f"📊 **Raspberry Pi 5 邊緣運算節點健康狀態**\n\n"
        f"🌡️ **CPU 溫度:** `{telem['cpu_temp']:.1f} °C` " + ("🟢 (優良)" if telem['cpu_temp'] < 65 else "🟠 (注意)") + "\n"
        f"💾 **實體記憶體:** `{telem['free_ram']:,} MB` 可用 / `{telem['total_ram']:,} MB`\n"
        f"🤖 **運作中模型:** `{current_model}`\n"
        f"🧠 **短期記憶緩衝:** `{working_turns} 輪對話`\n"
        f"📝 **個人事實特徵:** `{facts_count} 條記錄`\n"
        f"🕒 **節點時間:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`"
    )
    await update.message.reply_text(status_msg, parse_mode=ParseMode.MARKDOWN)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /model <model_name> command."""
    if not is_authorized(update.effective_user.id):
        return

    chat_id = update.effective_user.id
    if not context.args:
        # Show list of available models
        try:
            resp = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m.get("name") for m in resp.json().get("models", [])]
                model_list_str = "\n".join([f"- `{m}`" for m in models])
                msg = f"當前使用模型: `{get_active_model(chat_id)}`\n\n可用模型清單：\n{model_list_str}\n\n使用方式：`/model <模型名稱>`"
                await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
                return
        except Exception as e:
            await update.message.reply_text(f"無法取得模型清單: {e}")
            return

    new_model = context.args[0].strip()
    active_models[chat_id] = new_model
    await update.message.reply_text(f"✅ 已成功將推論大腦切換為: `{new_model}`", parse_mode=ParseMode.MARKDOWN)


async def remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /remember <text> command: explicitly store into Tier 3 vector memory."""
    if not is_authorized(update.effective_user.id):
        return

    if not context.args:
        await update.message.reply_text("請輸入要記入長期記憶的內容。例如：`/remember 公司的伺服器密碼是 xyz`", parse_mode=ParseMode.MARKDOWN)
        return

    content = " ".join(context.args).strip()
    chat_id = update.effective_user.id

    success = memory_mgr.store_memory(chat_id, content)
    if success:
        await update.message.reply_text(f"🧠 **已成功固化至長期語意向量記憶庫 (Tier 3)**：\n> {content}", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("❌ 記憶儲存失敗，請檢查本地向量模型 `nomic-embed-text` 是否就緒。")


async def facts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /facts command: display and manage Tier 2 user facts."""
    if not is_authorized(update.effective_user.id):
        return

    chat_id = update.effective_user.id
    facts = memory_mgr.get_user_facts(chat_id)
    if not facts:
        await update.message.reply_text("目前尚未登記任何個人事實特徵 (Tier 2)。您可以在日常對話中告訴我您的偏好，或直接輸入指令。")
        return

    lines = ["📝 **已記住的個人事實特徵 (Tier 2 Profile Facts):**"]
    for k, v in facts.items():
        lines.append(f"- **{k}**: {v}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command: reset Tier 1 working memory."""
    if not is_authorized(update.effective_user.id):
        return

    chat_id = update.effective_user.id
    memory_mgr.clear_working_memory(chat_id)
    await update.message.reply_text("🧹 已清空當前短期對話記憶緩衝區 (Tier 1)。長期記憶與個人事實依然保留。")


# =============================================================================
# Inference & Message Handlers
# =============================================================================

def call_ollama_chat(model: str, messages: list) -> str:
    """Call Ollama /api/chat endpoint."""
    url = f"{OLLAMA_API_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024
        }
    }
    resp = requests.post(url, json=payload, timeout=90)
    if resp.status_code == 200:
        return resp.json().get("message", {}).get("content", "").strip()
    return f"[Ollama 錯誤 {resp.status_code}]: {resp.text}"


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages."""
    user = update.effective_user
    if not is_authorized(user.id):
        return

    user_text = update.message.text.strip()
    chat_id = user.id
    model = get_active_model(chat_id)

    # Show typing action
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Base system instruction
    base_instruction = (
        "你是部署於使用者客廳樹莓派 5 (Raspberry Pi 5) 上的專屬邊緣 AI 語音個人助理。"
        "請一律以繁體中文 (Traditional Chinese, 台灣語境) 親切、精準、專業地回答。"
        "回答力求清晰簡潔，重點條理分明。"
    )

    # Check tool execution (Web search / Taiwan news)
    tool_context = execute_tool_call_if_needed(user_text)
    if tool_context:
        base_instruction = f"{base_instruction}\n\n{tool_context}"

    # Build prompt messages from Three-Tier Memory
    messages = memory_mgr.build_prompt_messages(chat_id, user_text, base_instruction)

    try:
        reply_text = call_ollama_chat(model, messages)
        # Update Tier 1 working memory
        memory_mgr.append_turn(chat_id, user_text, reply_text)
        await update.message.reply_text(reply_text)
    except Exception as e:
        logger.error(f"Text chat error: {e}")
        await update.message.reply_text(f"❌ 推論發生錯誤: {e}")


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle incoming voice/audio messages:
    Telegram Voice (.oga) -> faster-whisper STT -> Three-Tier Memory + LLM -> edge-tts TTS -> Telegram Voice reply
    """
    user = update.effective_user
    if not is_authorized(user.id):
        return

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    chat_id = user.id
    model = get_active_model(chat_id)

    # Send recording audio action
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)

    # Download voice file from Telegram
    voice_file = await voice.get_file()
    temp_id = uuid.uuid4().hex[:8]
    input_oga_path = os.path.join(DATA_DIR, f"voice_in_{temp_id}.oga")
    output_ogg_path = os.path.join(DATA_DIR, f"voice_out_{temp_id}.ogg")

    try:
        await voice_file.download_to_drive(input_oga_path)

        # 1. Speech-To-Text
        transcribed_text = speech_to_text(input_oga_path)
        if not transcribed_text:
            await update.message.reply_text("🔇 無法辨識語音內容，請再試一次或改用文字輸入。")
            return

        logger.info(f"[Voice STT] User {chat_id}: {transcribed_text}")
        await update.message.reply_text(f"🎙️ *您說:* 「{transcribed_text}」", parse_mode=ParseMode.MARKDOWN)

        # 2. LLM Reasoning with Three-Tier Memory & Tools
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
        base_instruction = (
            "你是部署於樹莓派 5 上的邊緣語音個人助理。使用者正使用語音與你交談。"
            "使用者語音轉文字若含有同音或相近錯別字，請依據上下文與樹莓派專案語境自動對齊其真實意圖。"
            "請以繁體中文 (Traditional Chinese, 台灣語音習慣) 回應，語言力求自然、生動、簡潔，"
            "避免過多複雜的排版符號，以便於語音合成流暢朗讀。"
        )

        # Check tool execution (Web search / Taiwan news)
        tool_context = execute_tool_call_if_needed(transcribed_text)
        if tool_context:
            await update.message.reply_text("🌐 *正在為您連線檢索最新台灣即時資訊...*", parse_mode=ParseMode.MARKDOWN)
            base_instruction = f"{base_instruction}\n\n{tool_context}"

        messages = memory_mgr.build_prompt_messages(chat_id, transcribed_text, base_instruction)
        reply_text = call_ollama_chat(model, messages)

        # Update Tier 1 working memory
        memory_mgr.append_turn(chat_id, transcribed_text, reply_text)

        # 3. Text-To-Speech Synthesis
        tts_success = await text_to_speech_async(reply_text, output_ogg_path)

        if tts_success and os.path.exists(output_ogg_path):
            # Send voice message back
            with open(output_ogg_path, "rb") as audio_fp:
                await update.message.reply_voice(
                    voice=audio_fp,
                    caption=f"📝 {reply_text[:200]}..." if len(reply_text) > 200 else f"📝 {reply_text}"
                )
        else:
            # Fallback to text if TTS fails
            await update.message.reply_text(reply_text)

    except Exception as e:
        logger.error(f"Voice pipeline error: {e}")
        await update.message.reply_text(f"❌ 語音處理管線錯誤: {e}")
    finally:
        # Cleanup temporary audio files
        for p in [input_oga_path, output_ogg_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def main():
    """Main daemon entrypoint."""
    token = TELEGRAM_BOT_TOKEN
    if not token:
        print("[Error] TELEGRAM_BOT_TOKEN is not set! Please set it in config.py or environment.")
        sys.exit(1)

    print(f"🚀 Starting Pi 5 Telegram Voice AI Agent (PID: {os.getpid()})...")
    app = ApplicationBuilder().token(token).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("remember", remember_command))
    app.add_handler(CommandHandler("facts", facts_command))
    app.add_handler(CommandHandler("clear", clear_command))

    # Register message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice_message))

    print("🤖 Pi 5 Telegram Agent is running and listening for messages/voice...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
