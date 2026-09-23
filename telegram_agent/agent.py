# f:\12_prj_raspi5\telegram_agent\agent.py
"""
Raspberry Pi 5 Telegram Voice AI Agent Core Daemon.
Integrates:
1. python-telegram-bot
2. Three-Tier Memory Pyramid (Working Buffer, Profile Facts, Vector RAG)
3. Edge Voice Pipeline (faster-whisper STT & edge-tts Neural TTS)
4. Dynamic Evolving Skills Registry & Hardware Telemetry
5. Autonomous Self-Evolution Engine (10 Daily Goals & Proactive Progress Reports)
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
import asyncio
import subprocess
from typing import Dict, Any, List

import requests
from telegram import Update, BotCommand
from telegram.constants import ParseMode, ChatAction
from telegram.request import HTTPXRequest
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
from evolution_engine import evolution_engine
from notifier import broadcast_daily_plan_overview, broadcast_progress_milestone, broadcast_deep_work_step

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


def is_authorized(user_id: int, username: str = "") -> bool:
    """Check whitelist and register subscriber for proactive push notifications."""
    if not ALLOWED_CHAT_IDS:
        evolution_engine.register_chat_subscriber(user_id, username)
        return True
    if user_id in ALLOWED_CHAT_IDS:
        evolution_engine.register_chat_subscriber(user_id, username)
        return True
    return False


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
    if not is_authorized(user.id, user.username or ""):
        await update.message.reply_text(f"⚠️ 未授權存取。您的 Telegram User ID 為: `{user.id}`，請聯繫管理員加入白名單。", parse_mode=ParseMode.MARKDOWN)
        return

    msg = (
        f"👋 您好 {user.first_name}！我是您的 **Raspberry Pi 5 自主進化語音 AI 助理**。\n\n"
        f"🧠 **當前推論大腦:** `{get_active_model(user.id)}`\n"
        f"🏛️ **三層記憶金字塔:** 已啟用 (即時緩衝 + 個人畫像 + 語意向量庫)\n"
        f"🌱 **自主進化引擎:** 每日排定 10 項進化計畫並自主推進與回報\n"
        f"🎙️ **語音互動:** 直接發送語音訊息，我會用語音回覆您！\n\n"
        f"常用指令：\n"
        f"/plan - 檢視今日 10 大自主進化目標與當前進度\n"
        f"/evolve - 立即推進執行下一項未完成的進化目標\n"
        f"/status - 檢視樹莓派 5 核心溫度與記憶體硬體遙測\n"
        f"/model <名稱> - 切換 AI 模型 (如 qwen2.5:3b-opt, deepseek-r1:1.5b-opt)\n"
        f"/remember <文字> - 手動將重要資訊記入長期向量庫\n"
        f"/facts - 查看我為您記住的個人事實特徵\n"
        f"/clear - 重置當前短期對話記憶"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return
    help_text = (
        "📖 **Raspberry Pi 5 AI 助理操作指南**\n\n"
        "1. **自主進化 (Self-Evolution):**\n"
        "   - `/plan`: 查看樹莓派今天為自己排定的 10 大改良目標與完成進度\n"
        "   - `/evolve`: 手動觸發執行下一項進化任務\n"
        "2. **語音互動 (Voice Chat):** 直接按住錄音鍵發送語音訊息，系統將自動進行語音轉錄 $\\to$ 思考 $\\to$ 繁體中文語音回覆。\n"
        "3. **電源與 RTC 管理:** 語音說「關機並在 30 秒後開機」即可調度底層硬體 RTC 晶片通電喚醒。\n"
        "4. **長期記憶 (Long-Term Memory):**\n"
        "   - `/remember <內容>`: 主動記憶重要事情\n"
        "   - `/facts`: 列出個人檔案特徵\n"
        "   - `/clear`: 清空短期上下文\n"
        "5. **模型切換 (Switch Models):**\n"
        "   - `/model qwen2.5:3b-opt` (繁中最佳，6 tok/s)\n"
        "   - `/model deepseek-r1:1.5b-opt` (極速思考，12 tok/s)\n"
        "   - `/model qwen2.5-coder:7b-opt` (深度代碼審查)\n"
        "6. **硬體監控 (Hardware Telemetry):**\n"
        "   - `/status`: 查看 CPU 溫度、可用 RAM 與健康狀態"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def plan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /plan: show today's 10 evolutionary goals and progress."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return
    msg = evolution_engine.format_plans_markdown()
    try:
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.warning(f"Failed to reply with MARKDOWN in /plan, falling back to plain text: {e}")
        await update.message.reply_text(msg)


async def evolve_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /evolve: manually advance the next evolutionary stage/task."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    try:
        await update.message.reply_text("⏳ *正在為您啟動並推進下一階段自主進化任務...*", parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text("⏳ 正在為您啟動並推進下一階段自主進化任務...")

    res = evolution_engine.execute_next_evolution_step()
    if res:
        summary = evolution_engine.get_progress_summary()
        pct = summary["progress_pct"]
        is_fin = res.get("is_finished", False)
        diff = res.get("difficulty", "MEDIUM")
        cur = res.get("stage_current", 1)
        tot = res.get("stage_total", 1)
        rem = res.get("remaining_mins", 0)

        if is_fin:
            reply = (
                f"🎉 **【演化任務完成】: [{res['task_index']}] {res['title']}**\n\n"
                f"📊 **今日總體進度:** `{pct}%` ({summary['completed']}/10 項完成)\n"
                f"⚡ **難度評估:** `{diff}`\n"
                f"📝 **達成成果:**\n{res['full_log']}"
            )
        else:
            reply = (
                f"🧠 **【深度進化階段推進 (Deep Work)】**\n\n"
                f"🎯 **目標:** [{res['task_index']}] {res['title']}\n"
                f"⚡ **難度評估:** `{diff}` (多階段深度運算)\n"
                f"📊 **當前階段:** 第 `{cur}/{tot}` 階段 (`{res.get('progress_pct', 0)}%`)\n"
                f"⏳ **預估剩餘時間:** 約 `{rem}` 分鐘 (系統持續在背景運作)\n\n"
                f"📝 **本階段進展:**\n{res['stage_log']}"
            )

        try:
            await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(reply)
    else:
        await update.message.reply_text("✨ 今日 10 大自主進化任務已全數完成！系統目前處於最佳狀態。")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command: display Pi 5 hardware metrics."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    telem = get_hardware_telemetry()
    chat_id = user.id
    current_model = get_active_model(chat_id)
    working_turns = len(memory_mgr.get_working_memory(chat_id)) // 2
    facts_count = len(memory_mgr.get_user_facts(chat_id))
    summary = evolution_engine.get_progress_summary()

    status_msg = (
        f"📊 **Raspberry Pi 5 邊緣運算節點健康狀態**\n\n"
        f"🌡️ **CPU 溫度:** `{telem['cpu_temp']:.1f} °C` " + ("🟢 (優良)" if telem['cpu_temp'] < 65 else "🟠 (注意)") + "\n"
        f"💾 **實體記憶體:** `{telem['free_ram']:,} MB` 可用 / `{telem['total_ram']:,} MB`\n"
        f"🤖 **運作中模型:** `{current_model}`\n"
        f"🌱 **今日進化進度:** `{summary['progress_pct']}%` ({summary['completed']}/10 項完成)\n"
        f"🧠 **短期記憶緩衝:** `{working_turns} 輪對話`\n"
        f"📝 **個人事實特徵:** `{facts_count} 條記錄`\n"
        f"🕒 **節點時間:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`"
    )
    await update.message.reply_text(status_msg, parse_mode=ParseMode.MARKDOWN)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /model <model_name> command."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    chat_id = user.id
    if not context.args:
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
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    if not context.args:
        await update.message.reply_text("請輸入要記入長期記憶的內容。例如：`/remember 公司的伺服器密碼是 xyz`", parse_mode=ParseMode.MARKDOWN)
        return

    content = " ".join(context.args).strip()
    chat_id = user.id

    success = memory_mgr.store_memory(chat_id, content)
    if success:
        await update.message.reply_text(f"🧠 **已成功固化至長期語意向量記憶庫 (Tier 3)**：\n> {content}", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("❌ 記憶儲存失敗，請檢查本地向量模型 `nomic-embed-text` 是否就緒。")


async def facts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /facts command: display and manage Tier 2 user facts."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    chat_id = user.id
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
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    chat_id = user.id
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
            "num_predict": 350
        }
    }
    resp = requests.post(url, json=payload, timeout=90)
    if resp.status_code == 200:
        return resp.json().get("message", {}).get("content", "").strip()
    return f"[Ollama 錯誤 {resp.status_code}]: {resp.text}"


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    user_text = update.message.text.strip()
    chat_id = user.id
    model = get_active_model(chat_id)

    # Show typing action
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Base system instruction
    base_instruction = (
        "你是部署於使用者客廳樹莓派 5 (Raspberry Pi 5) 上的專屬邊緣 AI 語音個人助理兼自主進化體。"
        "請一律以繁體中文 (Traditional Chinese, 台灣語境) 親切、精準、專業地回答。"
        "回答力求清晰簡潔，重點條理分明。"
    )

    # Check tool execution
    tool_context = execute_tool_call_if_needed(user_text)
    if tool_context:
        if "台灣證券交易所" in tool_context or "股市行情" in tool_context:
            await update.message.reply_text("📈 *正在連線台灣證券交易所 (TWSE) 查詢最新股市與股價行情...*", parse_mode=ParseMode.MARKDOWN)
        elif "電源管理" in tool_context:
            await update.message.reply_text("⚡ *正在排程系統電源狀態 (RTC / Power Control)...*", parse_mode=ParseMode.MARKDOWN)
        elif "硬體監控" in tool_context:
            await update.message.reply_text("🌡️ *正在讀取板載感測器與硬體數據...*", parse_mode=ParseMode.MARKDOWN)
        elif "即時新聞" in tool_context or "聯網搜尋" in tool_context:
            await update.message.reply_text("🌐 *正在為您連線檢索最新台灣即時資訊...*", parse_mode=ParseMode.MARKDOWN)
        elif "自我進化" in tool_context:
            await update.message.reply_text("🛠️ *正在調用代碼模型自建新技能模組並熱載入...*", parse_mode=ParseMode.MARKDOWN)
        base_instruction = f"{base_instruction}\n\n【真實數據與事實嚴格原則】: 若有提供工具數據 (如證交所股價、硬體數據)，必須 100% 依據提供之數字回答，嚴禁憑空猜測或編造虛假行情！\n\n{tool_context}"

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
    if not is_authorized(user.id, user.username or ""):
        return

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    chat_id = user.id
    model = get_active_model(chat_id)

    # Send recording audio action
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
    except Exception:
        pass

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
            try:
                await update.message.reply_text("🔇 無法辨識語音內容，請再試一次或改用文字輸入。")
            except Exception:
                pass
            return

        logger.info(f"[Voice STT] User {chat_id}: {transcribed_text}")
        try:
            await update.message.reply_text(f"🎙️ *您說:* 「{transcribed_text}」", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            try:
                await update.message.reply_text(f"🎙️ 您說: 「{transcribed_text}」")
            except Exception as ack_err:
                logger.warning(f"Could not send transcribed text ack: {ack_err}")

        # 2. LLM Reasoning with Three-Tier Memory & Tools
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
        except Exception:
            pass

        base_instruction = (
            "你是部署於樹莓派 5 上的邊緣語音個人助理。使用者正使用語音與你交談。"
            "使用者語音轉文字若含有同音或相近錯別字，請依據上下文與樹莓派專案語境自動對齊其真實意圖。"
            "請以繁體中文 (Traditional Chinese, 台灣語音習慣) 回應，語言力求自然、生動、簡潔，"
            "避免過多複雜的排版符號，以便於語音合成流暢朗讀。"
        )

        # Check tool execution
        tool_context = execute_tool_call_if_needed(transcribed_text)
        if tool_context:
            try:
                if "台灣證券交易所" in tool_context or "股市行情" in tool_context:
                    await update.message.reply_text("📈 *正在連線台灣證券交易所 (TWSE) 查詢最新股市與股價行情...*", parse_mode=ParseMode.MARKDOWN)
                elif "電源管理" in tool_context:
                    await update.message.reply_text("⚡ *正在排程系統電源生命週期控制 (RTC / Power Control)...*", parse_mode=ParseMode.MARKDOWN)
                elif "硬體監控" in tool_context:
                    await update.message.reply_text("🌡️ *正在讀取板載感測器與硬體數據...*", parse_mode=ParseMode.MARKDOWN)
                elif "即時新聞" in tool_context or "聯網搜尋" in tool_context:
                    await update.message.reply_text("🌐 *正在為您連線檢索最新台灣即時資訊...*", parse_mode=ParseMode.MARKDOWN)
                elif "自我進化" in tool_context:
                    await update.message.reply_text("🛠️ *正在調用代碼模型自建新技能模組並熱載入...*", parse_mode=ParseMode.MARKDOWN)
            except Exception:
                pass
            base_instruction = f"{base_instruction}\n\n【真實數據與事實嚴格原則】: 若有提供工具數據 (如證交所股價、硬體數據)，必須 100% 依據提供之數字回答，嚴禁憑空猜測或編造虛假行情！\n\n{tool_context}"

        messages = memory_mgr.build_prompt_messages(chat_id, transcribed_text, base_instruction)
        reply_text = call_ollama_chat(model, messages)

        # Update Tier 1 working memory
        memory_mgr.append_turn(chat_id, transcribed_text, reply_text)

        # 3. Text-To-Speech Synthesis
        tts_success = await text_to_speech_async(reply_text, output_ogg_path)

        if tts_success and os.path.exists(output_ogg_path):
            try:
                with open(output_ogg_path, "rb") as audio_fp:
                    await update.message.reply_voice(
                        voice=audio_fp,
                        caption=f"📝 {reply_text[:200]}..." if len(reply_text) > 200 else f"📝 {reply_text}"
                    )
            except Exception as voice_err:
                logger.warning(f"Voice send failed, fallback to text: {voice_err}")
                await update.message.reply_text(reply_text)
        else:
            await update.message.reply_text(reply_text)

    except Exception as e:
        logger.error(f"Voice pipeline error: {e}")
        try:
            await update.message.reply_text(f"❌ 語音處理管線錯誤: {e}")
        except Exception:
            pass
    finally:
        for p in [input_oga_path, output_ogg_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# =============================================================================
# Autonomous Evolution Background Worker
# =============================================================================

async def evolution_background_worker(app):
    """
    Autonomous evolution scheduler:
    1. Generates 10 goals every day at 08:00 (or at startup).
    2. Proactively broadcasts the 10 goals overview to Telegram subscribers.
    3. Executes one pending task every 30 minutes.
    4. Proactively broadcasts milestone progress updates upon task completion.
    """
    logger.info("🌱 Autonomous Self-Evolution Engine worker initialized.")
    await asyncio.sleep(5)  # Wait for Telegram bot polling to establish
    
    today_str = time.strftime("%Y-%m-%d")
    evolution_engine.get_or_create_daily_plans(today_str)
    
    # Broadcast today's plan on start if there are subscribers
    broadcast_daily_plan_overview(today_str)

    last_exec_time = time.time()
    last_day_planned = today_str

    while True:
        try:
            now_t = time.time()
            current_day = time.strftime("%Y-%m-%d")
            current_hour = time.strftime("%H")

            # Daily plan reset at 08:00
            if current_day != last_day_planned and current_hour >= "08":
                evolution_engine.get_or_create_daily_plans(current_day)
                broadcast_daily_plan_overview(current_day)
                last_day_planned = current_day

            # Execute next progressive evolution step every 20 minutes (1200s)
            if now_t - last_exec_time >= 1200:
                pending_res = evolution_engine.execute_next_evolution_step(current_day)
                if pending_res:
                    summary = evolution_engine.get_progress_summary(current_day)
                    broadcast_deep_work_step(pending_res, summary["progress_pct"])
                last_exec_time = now_t

        except Exception as e:
            logger.error(f"Error in evolution background worker: {e}")

        await asyncio.sleep(60)


async def post_init_hook(app):
    """Post initialization hook: start background worker coroutine."""
    asyncio.create_task(evolution_background_worker(app))


def main():
    """Main daemon entrypoint."""
    token = TELEGRAM_BOT_TOKEN
    if not token:
        print("[Error] TELEGRAM_BOT_TOKEN is not set! Please set it in config.py or environment.")
        sys.exit(1)

    print(f"🚀 Starting Pi 5 Telegram Voice & Evolution AI Agent (PID: {os.getpid()})...")
    request_config = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=45.0,
        write_timeout=45.0,
        pool_timeout=30.0,
    )
    app = ApplicationBuilder().token(token).request(request_config).post_init(post_init_hook).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("plan", plan_command))
    app.add_handler(CommandHandler("evolve", evolve_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("remember", remember_command))
    app.add_handler(CommandHandler("facts", facts_command))
    app.add_handler(CommandHandler("clear", clear_command))

    # Register message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice_message))

    print("🤖 Pi 5 Telegram Agent with Self-Evolution Engine is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
