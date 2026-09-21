# telegram_agent/tools.py
"""
Evolving Skills & Tool Registry for Raspberry Pi 5 Telegram Voice AI Agent.
Supports built-in system diagnostics (temperature, memory, CPU), real-time web search,
and dynamic skill loading/evolution.
"""

import os
import sys
import subprocess
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
import json
import logging
import importlib.util
from typing import List, Dict, Optional, Callable

logger = logging.getLogger("Pi5Skills")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(BASE_DIR, "skills")
os.makedirs(SKILLS_DIR, exist_ok=True)

# =============================================================================
# Built-in Skills (核心基礎技能)
# =============================================================================

def get_pi5_hardware_status(param: str = "") -> str:
    """
    Query Raspberry Pi 5 hardware telemetry: SoC temperature, RAM usage, CPU load, and uptime.
    """
    results = []
    
    # 1. Temperature via vcgencmd
    try:
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=3)
        temp_str = res.stdout.strip() if res.returncode == 0 else "N/A"
        results.append(f"• 核心溫度 (SoC Temp): {temp_str.replace('temp=', '')}")
    except Exception:
        # Fallback to thermal zone sysfs
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                celsius = int(f.read().strip()) / 1000.0
                results.append(f"• 核心溫度 (SoC Temp): {celsius:.1f}°C")
        except Exception:
            results.append("• 核心溫度: 無法讀取")

    # 2. RAM Available via free
    try:
        res = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=3)
        lines = res.stdout.strip().splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            total, used, free, avail = parts[1], parts[2], parts[3], parts[6] if len(parts) > 6 else parts[3]
            results.append(f"• 記憶體狀態 (RAM): 已用 {used} / 總量 {total} (可用約 {avail})")
    except Exception:
        pass

    # 3. CPU Load & Uptime
    try:
        res = subprocess.run(["uptime"], capture_output=True, text=True, timeout=3)
        uptime_str = res.stdout.strip()
        results.append(f"• 系統運作與負載: {uptime_str}")
    except Exception:
        pass

    return "【樹莓派 5 實體硬體監控數據】:\n" + "\n".join(results)


def fetch_taiwan_news(topic: str = "") -> str:
    """
    Fetch real-time news from Google News Taiwan RSS.
    """
    if topic:
        query = urllib.parse.quote(topic)
        url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    else:
        url = "https://news.google.com/rss?hl=zh-TW&gl=TW&ceid=TW:zh-Hant"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    news_items = []
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml_data = resp.read()
        
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")[:5]
        for it in items:
            title_elem = it.find("title")
            desc_elem = it.find("description")
            title = title_elem.text if title_elem is not None else ""
            desc = re.sub(r'<[^>]+>', '', desc_elem.text).strip() if (desc_elem is not None and desc_elem.text) else ""
            if title:
                news_items.append(f"• {title}\n  (摘要: {desc[:120]})")
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return f"【實時聯網搜尋】: 無法抓取新聞資訊 ({e})"

    if not news_items:
        return f"【實時聯網搜尋】: 未能搜尋到關於『{topic}』的即時新聞。"
        
    return f"【實時聯網搜尋與台灣即時新聞 (主題: {topic or '頭條'})】:\n" + "\n".join(news_items)


# =============================================================================
# Dynamic Skill Registry & Self-Evolution (動態技能登錄與自我進化)
# =============================================================================

class SkillRegistry:
    """
    Central registry for Built-in and Self-Created skills.
    Loads custom python modules dynamically from the `skills/` directory.
    """
    def __init__(self):
        self.skills: Dict[str, Dict] = {}
        self.register_builtin_skills()
        self.load_custom_skills()

    def register_builtin_skills(self):
        self.skills["hardware_status"] = {
            "name": "查詢硬體與板子溫度",
            "desc": "查詢樹莓派5的SoC核心溫度、記憶體RAM負載、CPU負載與運作時間",
            "keywords": ["溫度", "板子溫度", "硬體狀態", "記憶體", "ram", "cpu", "負載", "運作時間", "健康狀態", "幾度"],
            "func": get_pi5_hardware_status
        }
        self.skills["taiwan_news"] = {
            "name": "即時台灣新聞與網頁搜尋",
            "desc": "上網檢索台灣最新頭條新聞或特定時事主題",
            "keywords": ["新聞", "時事", "頭條", "最新消息", "搜尋", "上網找", "查詢", "找一下"],
            "func": fetch_taiwan_news
        }

    def load_custom_skills(self):
        """Scan skills/ folder and dynamically load user/agent generated python skills."""
        if not os.path.exists(SKILLS_DIR):
            return
        for fname in os.listdir(SKILLS_DIR):
            if fname.endswith(".py") and not fname.startswith("__"):
                skill_path = os.path.join(SKILLS_DIR, fname)
                try:
                    mod_name = fname[:-3]
                    spec = importlib.util.spec_from_file_location(mod_name, skill_path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        if hasattr(mod, "SKILL_METADATA") and hasattr(mod, "execute"):
                            meta = getattr(mod, "SKILL_METADATA")
                            self.skills[mod_name] = {
                                "name": meta.get("name", mod_name),
                                "desc": meta.get("desc", ""),
                                "keywords": meta.get("keywords", []),
                                "func": getattr(mod, "execute")
                            }
                            logger.info(f"Dynamically loaded self-created skill: {mod_name}")
                except Exception as e:
                    logger.error(f"Failed to load skill {fname}: {e}")

    def dispatch_skill(self, user_text: str) -> Optional[str]:
        """
        Evaluate user intent and trigger corresponding skill function.
        """
        user_lower = user_text.lower()
        
        # Priority 1: Hardware & Temperature check
        hw_keywords = self.skills["hardware_status"]["keywords"]
        if any(k in user_lower for k in hw_keywords):
            logger.info("Triggered Skill: get_pi5_hardware_status")
            raw_info = self.skills["hardware_status"]["func"]()
            return f"{raw_info}\n請根據以上真實的硬體溫度與數據，以繁體中文親切告知使用者目前板子的狀態。"

        # Priority 2: Web Search & News check
        news_keywords = self.skills["taiwan_news"]["keywords"]
        if any(k in user_lower for k in news_keywords):
            logger.info("Triggered Skill: fetch_taiwan_news")
            clean_q = re.sub(r"(請|幫我|上網|搜尋|查詢|看|找|今天的|今日的|台灣的|新聞|時事)", "", user_text).strip()
            topic = clean_q if clean_q else "台灣新聞"
            raw_info = self.skills["taiwan_news"]["func"](topic)
            return f"{raw_info}\n請根據以上搜尋結果，為使用者統整出重點摘要並語音朗讀回答。"

        # Priority 3: Custom Loaded Skills
        for s_id, s_info in self.skills.items():
            if s_id in ["hardware_status", "taiwan_news"]:
                continue
            if any(k.lower() in user_lower for k in s_info.get("keywords", [])):
                logger.info(f"Triggered Dynamic Skill: {s_id}")
                try:
                    res = s_info["func"](user_text)
                    return f"【技能執行結果 ({s_info['name']})】:\n{res}\n請根據此結果回應使用者。"
                except Exception as e:
                    return f"技能執行出錯: {e}"

        return None


# Global registry singleton
_registry = SkillRegistry()

def execute_tool_call_if_needed(user_text: str) -> Optional[str]:
    """Top-level entry point used by agent.py."""
    return _registry.dispatch_skill(user_text)


def register_new_skill_code(skill_name: str, code: str) -> bool:
    """
    Self-Evolution API: Allows the Agent or User to write a new skill code file dynamically,
    verify its syntax, and hot-reload it into the registry.
    """
    filename = f"{skill_name}.py"
    filepath = os.path.join(SKILLS_DIR, filename)
    try:
        # Compile test first
        compile(code, filepath, "exec")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        _registry.load_custom_skills()
        return True
    except Exception as e:
        logger.error(f"Skill registration failed: {e}")
        return False

if __name__ == "__main__":
    print("1. Testing Hardware Status:")
    print(execute_tool_call_if_needed("幫我查板子的溫度還有幾度"))
    print("\n2. Testing News Search:")
    print(execute_tool_call_if_needed("幫我查今天的新聞"))
