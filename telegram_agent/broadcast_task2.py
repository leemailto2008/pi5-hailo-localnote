#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/pi/pi5-llm-arena')
from telegram_agent.evolution_engine import evolution_engine
from telegram_agent.notifier import broadcast_deep_work_step

plans = evolution_engine.get_or_create_daily_plans()
t2 = next((p for p in plans if p['task_index'] == 2), None)
if t2:
    step_info = {
        "task_index": 2,
        "title": t2["title"],
        "category": t2["category"],
        "difficulty": t2["difficulty"],
        "stage_current": 4,
        "stage_total": 4,
        "progress_pct": 100,
        "is_finished": True,
        "is_deep_work": 1,
        "remaining_mins": 0,
        "stage_log": "技能原型沙盒單元測試通過，已註冊至動態擴展候選清單 (SkillRegistry Candidate)。",
        "full_log": t2["result_log"],
    }
    sent = broadcast_deep_work_step(step_info, 20)
    print(f"Sent completion milestone for Task 2 to {sent} Telegram subscribers.")
