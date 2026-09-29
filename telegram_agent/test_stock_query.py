#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/pi/pi5-llm-arena/telegram_agent')
from tools import execute_tool_call_if_needed

user_prompt = "幫我查一下,台灣新聞裡面,台積電,9月23號,今天的股價多少,還有股市行情。我故意講錯很多次,這麼長,看你分目分辨得出來。"
print("Input:", user_prompt)
res = execute_tool_call_if_needed(user_prompt)
print("\n=== TOOL OUTPUT ===")
print(res)
