#!/usr/bin/env python3
import sqlite3
import os

db_path = '/home/pi/pi5-llm-arena/telegram_agent/data/memory.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, chat_id, content, created_at FROM episodic_memories ORDER BY id DESC LIMIT 10")
    rows = cur.fetchall()
    print("=== EPISODIC MEMORIES ===")
    for r in rows:
        print(f"[{r[0]}] User {r[1]} ({r[3]}): {r[2]}")

    cur.execute("SELECT chat_id, fact_key, fact_value FROM user_facts")
    facts = cur.fetchall()
    print("\n=== USER FACTS ===")
    for f in facts:
        print(f"User {f[0]}: {f[1]} -> {f[2]}")
