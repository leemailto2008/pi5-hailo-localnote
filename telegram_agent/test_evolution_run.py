#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, '/home/pi/pi5-llm-arena')
from telegram_agent.evolution_engine import evolution_engine
from telegram_agent.notifier import broadcast_deep_work_step

def main():
    print("=== 1. Checking plans summary ===")
    summary = evolution_engine.get_progress_summary()
    print(f"Date: {summary['plan_date']}, Total: {summary['total']}, Completed: {summary['completed']}, InProgress: {summary['in_progress']}, Pct: {summary['progress_pct']}%")

    print("\n=== 2. Advancing next stage (completing Task 1) ===")
    res = evolution_engine.execute_next_evolution_step()
    if res:
        print(f"Task #{res['task_index']} {res['title']}")
        print(f"Stage: {res['stage_current']}/{res['stage_total']}, Finished: {res['is_finished']}")
        print(f"Remaining mins: {res['remaining_mins']}m, Difficulty: {res['difficulty']}")
        print(f"Stage log: {res['stage_log']}")

        # Test broadcast to Telegram
        print("\n=== 3. Testing Telegram Broadcast Notification ===")
        sent = broadcast_deep_work_step(res, evolution_engine.get_progress_summary()['progress_pct'])
        print(f"Broadcast sent to {sent} subscribers!")

    print("\n=== 4. Formatted Markdown Status for /plan ===")
    md = evolution_engine.format_plans_markdown()
    print(md)

if __name__ == "__main__":
    main()
