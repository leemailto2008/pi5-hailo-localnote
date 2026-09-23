# f:\12_prj_raspi5\telegram_agent\notifier.py
"""
Proactive Telegram Push Notification & Evolution Broadcast Service for Raspberry Pi 5.
Directly communicates via Telegram Bot HTTP API to push daily plans, periodic progress updates,
and critical system alerts to authorized users.
"""

import os
import sys
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Optional

logger = logging.getLogger("Pi5Notifier")

try:
    from telegram_agent.config import TELEGRAM_BOT_TOKEN, ALLOWED_CHAT_IDS
except ImportError:
    try:
        from config import TELEGRAM_BOT_TOKEN, ALLOWED_CHAT_IDS
    except ImportError:
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        ALLOWED_CHAT_IDS = []

try:
    from telegram_agent.evolution_engine import evolution_engine
except ImportError:
    try:
        from evolution_engine import evolution_engine
    except ImportError:
        evolution_engine = None


def send_telegram_message_direct(chat_id: int, text: str, parse_mode: Optional[str] = "Markdown") -> bool:
    """
    Send message directly via Telegram Bot HTTP API with automatic plain-text fallback.
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set; cannot send notification.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True
    except Exception as e:
        logger.warning(f"Failed to send with parse_mode={parse_mode}: {e}. Retrying with plain text...")
        if parse_mode:
            payload.pop("parse_mode", None)
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        return True
            except Exception as e2:
                logger.error(f"Failed to send plain text message to {chat_id}: {e2}")
        else:
            logger.error(f"Failed to send message to {chat_id}: {e}")

    return False


def broadcast_message(text: str, target_chat_ids: Optional[List[int]] = None) -> int:
    """
    Broadcast a message to all authorized subscribers or specified chat IDs.
    Returns the number of successfully delivered messages.
    """
    recipients = set()
    if target_chat_ids:
        recipients.update(target_chat_ids)
    
    # Add from ALLOWED_CHAT_IDS
    if ALLOWED_CHAT_IDS:
        recipients.update(ALLOWED_CHAT_IDS)

    # Add from evolution_engine database
    if evolution_engine:
        recipients.update(evolution_engine.get_all_subscribers())

    if not recipients:
        logger.warning("No recipients found to broadcast message.")
        return 0

    success_count = 0
    for cid in recipients:
        if send_telegram_message_direct(cid, text):
            success_count += 1

    return success_count


def broadcast_daily_plan_overview(target_date: Optional[str] = None) -> int:
    """Broadcast today's 10 Evolutionary Goals to Telegram."""
    if not evolution_engine:
        return 0
    msg = evolution_engine.format_plans_markdown(target_date)
    return broadcast_message(msg)


def broadcast_deep_work_step(step_info: dict, overall_pct: int) -> int:
    """Broadcast progressive deep work stage execution."""
    diff = step_info.get("difficulty", "MEDIUM")
    cur = step_info.get("stage_current", 1)
    tot = step_info.get("stage_total", 1)
    rem = step_info.get("remaining_mins", 0)
    task_idx = step_info.get("task_index", 1)
    title = step_info.get("title", "")
    is_fin = step_info.get("is_finished", False)
    stage_log = step_info.get("stage_log", "").replace("`", "")

    diff_labels = {
        "EASY": "🟢 [EASY: 輕量檢核]",
        "MEDIUM": "🟡 [MEDIUM: 中度調優]",
        "HARD": "🔴 [HARD: 深度演化算力密集]"
    }
    badge = diff_labels.get(diff, "[MEDIUM]")

    if is_fin:
        text = (
            f"🎉 **【演化任務完成】: [{task_idx}] {title}**\n\n"
            f"📊 **今日總體進度:** `{overall_pct}%`\n"
            f"⚡ **評估難度:** `{badge}`\n"
            f"📝 **最終達成成果:**\n{step_info.get('full_log', '')}\n\n"
            f"💡 *排程器已紀錄成果，將持續推進下一項進化目標。*"
        )
    else:
        text = (
            f"🧠 **【深度進化運算中 (Deep Work)】**\n\n"
            f"🎯 **目標:** [{task_idx}] {title}\n"
            f"⚡ **難度評估:** `{badge}`\n"
            f"📊 **階段推進:** 第 `{cur}/{tot}` 階段 (`{step_info.get('progress_pct', 0)}%`)\n"
            f"⏳ **預估本任務尚需時間:** 約 `{rem}` 分鐘 (系統持續在背景多做一陣子)\n\n"
            f"📝 **當前階段進展:**\n{stage_log}\n\n"
            f"💡 *輸入 `/plan` 可查看全體進度清單。*"
        )

    return broadcast_message(text)


def broadcast_progress_milestone(step_info: dict, overall_pct: int) -> int:
    """Backward compatibility wrapper for broadcast_deep_work_step."""
    return broadcast_deep_work_step(step_info, overall_pct)


if __name__ == "__main__":
    print("Testing notifier broadcast:")
    print("Recipients registered:", evolution_engine.get_all_subscribers() if evolution_engine else [])
