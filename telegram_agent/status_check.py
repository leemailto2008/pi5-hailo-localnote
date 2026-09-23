#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/pi/pi5-llm-arena')
from telegram_agent.evolution_engine import evolution_engine

plans = evolution_engine.get_or_create_daily_plans()
t2 = next((p for p in plans if p['task_index'] == 2), None)
if t2:
    print("=== TASK 2 DETAILS ===")
    print(f"Title: {t2['title']}")
    print(f"Difficulty: {t2['difficulty']}")
    print(f"Status: {t2['status']}")
    print(f"Progress: {t2['progress_pct']}%")
    print(f"Stage: {t2['stage_current']}/{t2['stage_total']}")
    print("Result Log:")
    print(t2['result_log'])
print("\n=== CURRENT /plan MARKDOWN ===")
print(evolution_engine.format_plans_markdown())
