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


def broadcast_progress_milestone(task_info: dict, overall_pct: int) -> int:
    """Broadcast milestone completion notification."""
    text = (
        f"📈 **【自我進化進度回報】**\n\n"
        f"✅ 已完成: **[{task_info.get('category', '進化任務')}] {task_info.get('title', '')}**\n"
        f"📊 **當前總體達成率:** `{overall_pct}%`\n\n"
        f"📝 **執行日誌與指標:**\n{task_info.get('result_log', '')}\n\n"
        f"系統將自主依排程繼續推進下一項目標。"
    )
    return broadcast_message(text)


if __name__ == "__main__":
    print("Testing notifier broadcast:")
    print("Recipients registered:", evolution_engine.get_all_subscribers() if evolution_engine else [])
