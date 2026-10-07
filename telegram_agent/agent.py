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
from voice_pipeline import speech_to_text, text_to_speech_async, record_audio_from_mic
from tools import execute_tool_call_if_needed
from evolution_engine import evolution_engine
from notifier import broadcast_daily_plan_overview, broadcast_progress_milestone, broadcast_deep_work_step, broadcast_message
from bluetooth_manager import (
    scan_devices as bt_scan_devices,
    pair_and_trust as bt_pair_and_trust,
    get_full_bt_status,
    connect_default_device as bt_connect_default,
    disconnect_device as bt_disconnect_device,
    auto_reconnect_tick as bt_auto_reconnect_tick,
    load_bt_config,
    is_device_connected
)

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
        "6. **藍牙音訊耳麥管理 (Bluetooth Headset):**\n"
        "   - `/voice_note [秒數]`: 透過耳麥錄音並轉錄中英對照文字，回傳語音檔與中英字卡 (如 `/voice_note 5`)\n"
        "   - `/bt_scan`: 掃描周圍處於配對模式的耳麥 (10 秒)\n"
        "   - `/bt_pair <編號/MAC>`: 一鍵配對、信任並設為預設耳麥\n"
        "   - `/bt`: 查看目前藍牙連線與耳麥狀態\n"
        "   - `/bt_connect` / `/bt_disconnect`: 手動連線或中斷\n"
        "   - *自動重連通知*: 耳麥開機靠近時自動秒連並透過 Telegram 即時回報！\n"
        "7. **硬體監控 (Hardware Telemetry):**\n"
        "   - `/status`: 查看 CPU 溫度、可用 RAM 與健康狀態\n"
        "8. **網路模式切換 (Network Switch - 離線野外/省電隨選):**\n"
        "   - `/hotspot_a` (預設熱點方案): 切換為獨立離線熱點 `raspi543_AI` (IP: 10.20.0.1，省電/野外模式，附 180 秒防失聯看門狗)\n"
        "   - `/ap` 或 `/hotspot`: 同上快速別名"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def hotspot_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /hotspot_a, /hotspot or /ap: switch Pi 5 to standalone AP mode (10.20.0.1)."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    msg = (
        "📡 *【指令確認：切換為獨立熱點預設方案 A (/hotspot_a)】*\n\n"
        "• *熱點名稱 (SSID):* `raspi543_AI`\n"
        "• *連線密碼:* `raspi543`\n"
        "• *本地閘道網址:* http://10.20.0.1 (Port 80)\n"
        "• *省電策略:* 無連線時關閉高耗能組件，180 秒自動回滾\n\n"
        "🛡️ *【防失聯自動復歸看門狗已啟動】*\n"
        "• 系統已配置 180 秒故障保險 (Fail-Safe)。\n"
        "• 若 3 分鐘內未有手機連線或無操作，樹莓派將**自動安全切回原家用 Wi-Fi**並重連 Telegram，絕不失聯變磚！\n\n"
        "請拿出手機搜尋 Wi-Fi `raspi543_AI` (密碼 `raspi543`)，連線後造訪 http://10.20.0.1 即可開始離線使用！"
    )
    try:
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text("📡 正在切換為獨立熱點 raspi543_AI (10.20.0.1)，已啟動防失聯看門狗...")

    # Allow Telegram to safely flush message before network disconnects
    await asyncio.sleep(2.0)

    # Trigger safe switch to AP mode via Fail-Safe Hotspot Controller
    try:
        sys.path.insert(0, "/home/pi/code_dispatcher")
        import hotspot_controller
        hotspot_controller.switch_to_ap(timeout_sec=180)
    except Exception as e:
        logger.error(f"Failed to switch to hotspot: {e}")
        subprocess.Popen(["sudo", "nmcli", "connection", "up", "Pi5-Hotspot"])


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
# Bluetooth Audio Headset Commands
# =============================================================================

async def bt_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /bt or /bt_status: Display current Bluetooth connection and paired devices."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    status = get_full_bt_status()
    def_name = status.get("default_name") or "未設定"
    def_mac = status.get("default_mac") or "無"
    is_conn = status.get("default_connected", False)

    lines = [
        "🎧 **Raspberry Pi 5 藍牙音訊狀態**\n",
        f"📡 **藍牙控制器:** {'🟢 正常' if status['controller_ok'] else '🔴 異常'} ({status['controller_message']})",
        f"🎯 **預設耳麥:** *{def_name}* (`{def_mac}`)",
        f"🔗 **連線狀態:** {'🟢 已連線 (Connected)' if is_conn else '⚪ 未連線 (Disconnected)'}",
        f"🔄 **自動重連:** {'✅ 啟用中 (每 15 秒探測)' if status['auto_reconnect'] else '❌ 停用'}\n"
    ]

    paired = status.get("paired_devices", [])
    if paired:
        lines.append("📋 **已配對裝置清單 (Paired Devices):**")
        for dev in paired:
            star = "⭐ " if dev["is_default"] else "• "
            conn_tag = " [🟢 連線中]" if dev["connected"] else ""
            lines.append(f"{star}`{dev['mac']}` - {dev['name']}{conn_tag}")
    else:
        lines.append("ℹ️ 尚未配對任何裝置。輸入 `/bt_scan` 搜尋附近的耳麥。")

    lines.append("\n💡 常用指令：")
    lines.append("• `/bt_scan` - 搜尋附近處於配對狀態的耳麥")
    lines.append("• `/bt_pair <編號/MAC>` - 一鍵配對並設為預設耳麥")
    lines.append("• `/bt_connect` - 手動連線預設耳麥")
    lines.append("• `/bt_disconnect` - 中斷藍牙連線")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def bt_scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /bt_scan: Scan for nearby Bluetooth devices (runs 10s)."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    await update.message.reply_text(
        "🔍 **正在掃描周圍藍牙音訊裝置 (約需 10 秒)...**\n"
        "請確保您的耳麥已開機並處於「配對模式 (Pairing Mode / 燈號閃爍)」！",
        parse_mode=ParseMode.MARKDOWN
    )
    await context.bot.send_chat_action(chat_id=user.id, action=ChatAction.TYPING)

    # Run blocking scan in thread pool
    devices = await asyncio.to_thread(bt_scan_devices, 10)

    if not devices:
        await update.message.reply_text(
            "❌ **未搜尋到任何藍牙裝置。**\n"
            "建議：\n"
            "1. 確認耳麥已進入配對狀態 (長按耳麥電源/配對鍵至紅藍閃爍)。\n"
            "2. 靠近樹莓派主機後再試一次 `/bt_scan`。",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    lines = ["🎧 **搜尋到的藍牙裝置清單：**\n"]
    for idx, dev in enumerate(devices, start=1):
        status_tags = []
        if dev.get("connected"):
            status_tags.append("🟢 連線中")
        elif dev.get("paired"):
            status_tags.append("⚪ 已配對")
        tag_str = f" ({', '.join(status_tags)})" if status_tags else ""
        lines.append(f"**[{idx}]** `{dev['mac']}` - *{dev['name']}*{tag_str}")

    lines.append("\n👉 **如何配對：**")
    lines.append("請直接輸入 `/bt_pair <編號>` (例如 `/bt_pair 1`) 或 `/bt_pair <MAC>` 進行綁定！")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def bt_pair_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /bt_pair <index or MAC>: Pair, trust, and connect target device."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    if not context.args:
        await update.message.reply_text(
            "⚠️ 請指定要配對的裝置編號或 MAC 位址！\n"
            "範例：\n"
            "• `/bt_pair 1` (配對上次掃描清單的第 1 個裝置)\n"
            "• `/bt_pair AA:BB:CC:DD:EE:FF`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    target = context.args[0].strip()
    await update.message.reply_text(f"⚡ **正在嘗試與裝置 [{target}] 進行配對與綁定信任...**", parse_mode=ParseMode.MARKDOWN)
    await context.bot.send_chat_action(chat_id=user.id, action=ChatAction.TYPING)

    # Run blocking pair_and_trust in thread pool
    success, msg = await asyncio.to_thread(bt_pair_and_trust, target)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def bt_connect_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /bt_connect: Manually connect to default headset."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    await update.message.reply_text("🔄 正在連線至預設耳麥...")
    await context.bot.send_chat_action(chat_id=user.id, action=ChatAction.TYPING)

    success, msg = await asyncio.to_thread(bt_connect_default)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def bt_disconnect_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /bt_disconnect: Disconnect Bluetooth device."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    target_mac = context.args[0].strip() if context.args else None
    success, msg = await asyncio.to_thread(bt_disconnect_device, target_mac)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def voice_record_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle /voice_note [seconds] or /voice [seconds] command:
    1. Records audio from default microphone (Bluetooth headset) for N seconds.
    2. Transcribes voice to Traditional Chinese via faster-whisper.
    3. Translates Chinese transcript into English via Ollama.
    4. Sends both the voice note audio and bilingual transcript back to Telegram.
    """
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    chat_id = user.id
    duration = 5
    if context.args:
        try:
            duration = int(context.args[0])
            if duration < 1:
                duration = 1
            elif duration > 60:
                duration = 60
        except ValueError:
            await update.message.reply_text(
                "⚠️ **秒數格式錯誤**\n請輸入整數秒數，例如：`/voice_note 5` 或 `/voice_note 10` (支援 1~60 秒)。",
                parse_mode=ParseMode.MARKDOWN
            )
            return

    # Check and ensure Bluetooth headset connection
    config = load_bt_config()
    default_mac = config.get("default_mac")
    default_name = config.get("default_name") or "藍牙耳麥"

    if default_mac and not is_device_connected(default_mac):
        wake_msg = await update.message.reply_text(
            f"🔄 偵測到耳麥 *{default_name}* 處於休眠/離線狀態，正在連線中...",
            parse_mode=ParseMode.MARKDOWN
        )
        await asyncio.to_thread(bt_connect_default)
        if not is_device_connected(default_mac):
            await wake_msg.edit_text(
                f"⚠️ **無法連線至耳麥：{default_name}**\n\n"
                "• 請確認耳麥電源已開機並位於連線範圍內。\n"
                "• 系統背景已啟用自動秒連；耳機開機連上後請再次執行 `/voice_note` 錄音！",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        await wake_msg.delete()

    status_msg = await update.message.reply_text(
        f"🎙️ **正在透過藍牙耳麥錄製語音 ({duration} 秒)...**\n請現在對耳麥麥克風說話！",
        parse_mode=ParseMode.MARKDOWN
    )

    temp_ogg = f"/tmp/voice_note_{uuid.uuid4().hex[:8]}.ogg"
    try:
        # Record from default pulse audio input (Bluetooth headset)
        success = await record_audio_from_mic(duration, temp_ogg)
        if not success or not os.path.exists(temp_ogg):
            await status_msg.edit_text("❌ **錄音失敗：** 無法從麥克風捕獲音訊，請確認藍牙耳麥連線正常。")
            return

        await status_msg.edit_text(f"⏳ **錄音完成 ({duration}s)，正在進行語音轉錄與中英翻譯...**", parse_mode=ParseMode.MARKDOWN)

        # 1. ASR Transcription via faster-whisper (ARM NEON int8)
        zh_text = await asyncio.to_thread(speech_to_text, temp_ogg)

        # 2. English Translation via Ollama
        en_text = ""
        if zh_text and zh_text.strip():
            try:
                en_text = await asyncio.to_thread(translate_text_to_english, zh_text.strip())
            except Exception as tr_err:
                logger.warning(f"Translation failed: {tr_err}")

        # 3. Send Voice Note Audio file to Telegram
        with open(temp_ogg, "rb") as vf:
            await context.bot.send_voice(
                chat_id=chat_id,
                voice=vf,
                duration=duration,
                caption=f"🎙️ **語音備忘錄錄音檔 ({duration} 秒)**",
                parse_mode=ParseMode.MARKDOWN
            )

        # 4. Send Bilingual Transcript Message Card
        if zh_text and zh_text.strip():
            transcript_card = (
                "📝 **語音辨識與中英對照 (Voice Note Transcript)**\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"🇹🇼 **中文文字 (Traditional Chinese):**\n"
                f"{zh_text.strip()}\n\n"
                f"🇬🇧 **英文對照 (English Translation):**\n"
                f"{en_text.strip() if en_text else '*(翻譯產生中)*'}\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏱️ 錄音時長: `{duration} 秒` | ⚡ ASR: `faster-whisper int8`"
            )
        else:
            transcript_card = (
                "📝 **語音辨識結果 (Voice Note Transcript)**\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "🔇 **(未偵測到清晰語音內容)**\n"
                "💡 建議：請靠近耳麥麥克風清晰發話，或增加錄音秒數 (如 `/voice_note 8`)。\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏱️ 錄音時長: `{duration} 秒`"
            )

        await context.bot.send_message(
            chat_id=chat_id,
            text=transcript_card,
            parse_mode=ParseMode.MARKDOWN
        )

        await status_msg.delete()

    except Exception as e:
        logger.error(f"Error in /voice_note command: {e}")
        await status_msg.edit_text(f"❌ 處理語音筆記時發生異常: {e}")
    finally:
        if os.path.exists(temp_ogg):
            try:
                os.remove(temp_ogg)
            except Exception:
                pass


def translate_text_to_english(text: str) -> str:
    """Quick translation of Traditional Chinese text to English via Ollama."""
    if not text.strip():
        return ""
    prompt = f"Translate the following Traditional Chinese text into natural, concise English. Output ONLY the English translation, without quotation marks or explanations:\n\n{text}"
    raw = call_ollama_chat(DEFAULT_CHAT_MODEL, [{"role": "user", "content": prompt}]).strip()
    return raw.replace('"', '').strip()


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


def parse_bilingual_response(raw_reply: str) -> tuple:
    """
    Parse [USER_EN], [ZH], and [EN] components from bilingual Ollama response.
    Returns (user_en, zh_text, en_text).
    """
    text = raw_reply.strip()
    user_en = ""
    zh_text = ""
    en_text = ""
    
    # 1. Extract [USER_EN]
    if "[USER_EN]" in text:
        parts = text.split("[USER_EN]", 1)[1]
        if "[ZH]" in parts:
            user_en = parts.split("[ZH]", 1)[0].strip()
            rest = "[ZH]" + parts.split("[ZH]", 1)[1]
        elif "[EN]" in parts:
            user_en = parts.split("[EN]", 1)[0].strip()
            rest = "[EN]" + parts.split("[EN]", 1)[1]
        else:
            user_en = parts.strip()
            rest = ""
    else:
        rest = text

    # 2. Extract [ZH] and [EN]
    if "[ZH]" in rest and "[EN]" in rest:
        zh_parts = rest.split("[ZH]", 1)[1].split("[EN]", 1)
        zh_text = zh_parts[0].strip()
        en_text = zh_parts[1].strip()
    elif "[ZH]" in rest:
        zh_text = rest.split("[ZH]", 1)[1].strip()
    elif "[EN]" in rest:
        en_text = rest.split("[EN]", 1)[1].strip()
    else:
        zh_text = rest.strip()
        
    return user_en, zh_text, en_text


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages."""
    user = update.effective_user
    if not is_authorized(user.id, user.username or ""):
        return

    user_text = update.message.text.strip()
    chat_id = user.id
    model = get_active_model(chat_id)

    # Quick intent check: switch to hotspot
    lower_text = user_text.lower()
    if any(k in lower_text for k in ["切換熱點", "開熱點", "切到方案a", "切換方案a", "切換到方案a", "啟用熱點", "開啟熱點", "hotspot"]):
        await hotspot_command(update, context)
        return

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
        audio_file_size = os.path.getsize(input_oga_path) if os.path.exists(input_oga_path) else 0
        transcribed_text = speech_to_text(input_oga_path)
        if not transcribed_text:
            logger.warning(f"[Voice STT Failed] User {chat_id}, file size={audio_file_size} bytes. Audio transcription empty.")
            try:
                await update.message.reply_text(
                    "🔇 *無法辨識語音內容*\n"
                    "建議：請靠近麥克風並清晰發話約 2~5 秒後再試一次，或改用文字輸入交談。",
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception:
                try:
                    await update.message.reply_text("🔇 無法辨識語音內容，請靠近麥克風清晰說話約 2~5 秒再試一次。")
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
            "你是部署於樹莓派 5 上的邊緣 AI 雙語語音助理。使用者正使用語音與你交談。"
            "使用者語音轉文字若含有同音或相近錯別字，請依據上下文自動對齊其真實意圖。"
            "請嚴格依據以下三段標籤結構輸出完整中英雙語對照：\n"
            "[USER_EN] 將使用者發話內容翻譯成簡潔道地的英文\n"
            "[ZH] 以親切精煉的繁體中文回答（30~50 字以內，文字簡潔流暢，以利語音朗讀）\n"
            "[EN] 將你的繁體中文回答翻譯成道地流暢的英文\n"
            "嚴禁添加多餘的特殊符號。"
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

        # Parse bilingual sections
        user_en, zh_text, en_text = parse_bilingual_response(reply_text)
        if not zh_text:
            zh_text = reply_text

        # Update Tier 1 working memory with clean Chinese response
        memory_mgr.append_turn(chat_id, transcribed_text, zh_text)

        # 3. Text-To-Speech Synthesis: Synthesize ONLY Chinese for natural voice
        tts_success = await text_to_speech_async(zh_text, output_ogg_path)

        # Build clean bilingual text card for visual display
        card_parts = [
            "🎙️ **語音輸入 (Voice Input):**",
            f"🇹🇼 「{transcribed_text}」"
        ]
        if user_en:
            card_parts.append(f'🇺🇸 "{user_en}"')

        card_parts.append("\n🤖 **助理回覆 (Assistant Reply):**")
        card_parts.append(f"🇹🇼 {zh_text}")
        if en_text:
            card_parts.append(f"🇺🇸 {en_text}")

        bilingual_card = "\n".join(card_parts)

        if tts_success and os.path.exists(output_ogg_path):
            try:
                with open(output_ogg_path, "rb") as audio_fp:
                    await update.message.reply_voice(
                        voice=audio_fp,
                        caption=bilingual_card if len(bilingual_card) <= 1024 else f"🇹🇼 {zh_text[:300]}...",
                        parse_mode=ParseMode.MARKDOWN
                    )
            except Exception as voice_err:
                logger.warning(f"Voice send failed, fallback to text: {voice_err}")
                await update.message.reply_text(bilingual_card, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(bilingual_card, parse_mode=ParseMode.MARKDOWN)

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


async def bluetooth_reconnect_worker(app):
    """
    Autonomous Bluetooth Reconnect & Status Broadcast Worker:
    Periodically checks every 12s if the default headset connection state changes.
    Proactively pushes notifications to Telegram whenever headset auto-connects or disconnects.
    """
    logger.info("🎧 Bluetooth Auto-Reconnect & Status worker initialized.")
    await asyncio.sleep(6)
    while True:
        try:
            msg = await asyncio.to_thread(bt_auto_reconnect_tick)
            if msg:
                # Broadcast status change directly to Telegram subscribers
                await asyncio.to_thread(broadcast_message, msg)
        except Exception as e:
            logger.debug(f"Bluetooth auto-reconnect tick error: {e}")

        await asyncio.sleep(12)


async def post_init_hook(app):
    """Post initialization hook: start background worker coroutines."""
    asyncio.create_task(evolution_background_worker(app))
    asyncio.create_task(bluetooth_reconnect_worker(app))


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

    # Register Voice & Bluetooth handlers
    app.add_handler(CommandHandler(["voice_note", "voice"], voice_record_command))
    app.add_handler(CommandHandler(["bt", "bt_status"], bt_status_command))
    app.add_handler(CommandHandler("bt_scan", bt_scan_command))
    app.add_handler(CommandHandler("bt_pair", bt_pair_command))
    app.add_handler(CommandHandler("bt_connect", bt_connect_command))
    app.add_handler(CommandHandler("bt_disconnect", bt_disconnect_command))

    # Register Network Switch handler
    app.add_handler(CommandHandler(["hotspot", "ap", "hotspot_a"], hotspot_command))

    # Register message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice_message))

    print("🤖 Pi 5 Telegram Agent with Self-Evolution & Bluetooth Engine is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

