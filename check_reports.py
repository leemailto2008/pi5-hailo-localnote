# check_reports.py
import sqlite3
import json
import datetime
from telegram_agent.config import ALLOWED_CHAT_IDS, TELEGRAM_BOT_TOKEN
from telegram_agent.evolution_engine import evolution_engine
from telegram_agent.notifier import broadcast_daily_plan_overview

conn = sqlite3.connect('/home/pi/pi5-llm-arena/telegram_agent/data/evolution.db')
conn.row_factory = sqlite3.Row

print("=== 1. SUBSCRIBERS ===")
subs = [dict(r) for r in conn.execute('SELECT * FROM known_subscribers').fetchall()]
print(f"Known subscribers in DB: {subs}")
print(f"Config ALLOWED_CHAT_IDS: {ALLOWED_CHAT_IDS}")

print("\n=== 2. DAILY PLANS SUMMARY BY DATE ===")
history = [dict(r) for r in conn.execute('SELECT plan_date, COUNT(*) as total, SUM(CASE WHEN status="COMPLETED" THEN 1 ELSE 0 END) as completed FROM daily_plans GROUP BY plan_date ORDER BY plan_date DESC').fetchall()]
print(json.dumps(history, indent=2, ensure_ascii=False))

print("\n=== 3. TODAY'S PLANS DETAILS ===")
today = datetime.date.today().isoformat()
today_plans = [dict(r) for r in conn.execute('SELECT * FROM daily_plans WHERE plan_date=? ORDER BY task_index', (today,)).fetchall()]
for p in today_plans:
    print(f"[{p['status']}] {p['task_index']}. [{p['category']}] {p['title']}")
    if p['result_log']:
        for line in p['result_log'].splitlines():
            print(f"   {line}")
    print()

print("\n=== 4. YESTERDAY'S PLANS (OR PREVIOUS) ===")
yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
yesterday_plans = [dict(r) for r in conn.execute('SELECT * FROM daily_plans WHERE plan_date=? ORDER BY task_index', (yesterday,)).fetchall()]
if yesterday_plans:
    for p in yesterday_plans:
        first_log = p['result_log'].splitlines()[0] if p['result_log'] else ''
        print(f"[{p['status']}] {p['task_index']}. {p['title']} -> {first_log}")
else:
    print(f"No plans found for date: {yesterday}")

print("\n=== 5. BROADCAST TEST ===")
import logging
logging.basicConfig(level=logging.DEBUG)
from telegram_agent.notifier import send_telegram_message_direct
print("Testing direct send to 6226061516:")
res = send_telegram_message_direct(6226061516, "測試訊息：Raspberry Pi 5 自檢回報")
print("Direct send result:", res)

from telegram_agent.notifier import broadcast_daily_plan_overview
print("Testing broadcast_daily_plan_overview:")
b_res = broadcast_daily_plan_overview()
print("Broadcast overview result:", b_res)


