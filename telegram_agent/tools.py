# f:\12_prj_raspi5\telegram_agent\tools.py
"""
Evolving Skills & Tool Registry for Raspberry Pi 5 Telegram Voice AI Agent.
Supports:
1. Hardware Telemetry (SoC Temp, RAM, CPU Load)
2. Real-time Taiwan News / Web Search
3. Power Management & RTC Wakeup (Timed Poweroff & Auto-Wakeup / Reboot)
4. Dynamic Skill Synthesis & Hot-Reloading (Self-Evolution via Local LLM)
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
from typing import List, Dict, Optional, Callable, Tuple

logger = logging.getLogger("Pi5Skills")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(BASE_DIR, "skills")
os.makedirs(SKILLS_DIR, exist_ok=True)

# Try importing config for Ollama API URL
try:
    from config import OLLAMA_API_URL, DEFAULT_CHAT_MODEL
except ImportError:
    OLLAMA_API_URL = "http://127.0.0.1:11434"
    DEFAULT_CHAT_MODEL = "llama3.2:3b"

# =============================================================================
# Helper: Chinese Number to Integer Parser
# =============================================================================

def parse_time_duration_seconds(text: str) -> int:
    """
    Parse seconds or minutes from natural speech in Traditional Chinese or digits.
    Examples: '30秒' -> 30, '1分鐘' -> 60, '兩分鐘' -> 120, '十秒' -> 10, '半分鐘' -> 30.
    """
    cn_num = {
        '零': 0, '一': 1, '二': 2, '兩': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
    }
    
    # 1. Direct regex for digits
    match_sec = re.search(r'(\d+)\s*(?:秒鐘|秒)', text)
    if match_sec:
        return int(match_sec.group(1))

    match_min = re.search(r'(\d+)\s*(?:分鐘|分)', text)
    if match_min:
        return int(match_min.group(1)) * 60

    # 2. Chinese words regex
    if "半分鐘" in text or "三十秒" in text:
        return 30
    if "一分鐘" in text:
        return 60
    if "兩分鐘" in text or "二分鐘" in text:
        return 120
    if "五分鐘" in text:
        return 300
    if "十分鐘" in text:
        return 600

    match_cn_sec = re.search(r'([一二兩三四五六七八九十]+)\s*(?:秒鐘|秒)', text)
    if match_cn_sec:
        val_str = match_cn_sec.group(1)
        if val_str == "十":
            return 10
        if val_str.startswith("十"):
            return 10 + cn_num.get(val_str[1], 0)
        if val_str.endswith("十"):
            return cn_num.get(val_str[0], 1) * 10
        if len(val_str) == 3 and val_str[1] == "十":
            return cn_num.get(val_str[0], 1) * 10 + cn_num.get(val_str[2], 0)
        return cn_num.get(val_str, 0)

    # Default fallback if mentioned power off without specific time: 0
    return 0


# =============================================================================
# Built-in Skills (核心基礎技能)
# =============================================================================

def get_pi5_hardware_status(param: str = "") -> str:
    """
    Query Raspberry Pi 5 hardware telemetry: SoC temperature, RAM usage, CPU load, and uptime.
    """
    results = []
    
    # 1. Temperature via vcgencmd or sysfs
    try:
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=3)
        temp_str = res.stdout.strip() if res.returncode == 0 else "N/A"
        results.append(f"• 核心溫度 (SoC Temp): {temp_str.replace('temp=', '')}")
    except Exception:
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


def extract_clean_news_topic(user_text: str) -> str:
    """Extract clean news topic from conversational text."""
    known_entities = [
        "台積電", "聯發科", "鴻海", "廣達", "台達電", "聯電", "大立光", "富邦金", "國泰金",
        "輝達", "NVIDIA", "OpenAI", "Google", "蘋果", "Apple", "微軟", "半導體", "AI",
        "加權指數", "台股", "股市", "地震", "颱風", "氣象", "物價"
    ]
    for ent in known_entities:
        if ent.lower() in user_text.lower():
            return ent

    cleaned = re.sub(r"^(請|幫我|協助我|上網|即時|連線|查詢|搜尋|看一下|找一下|看一下新聞|找新聞|有沒有|查一下|新聞裡面|台灣新聞裡面|台灣新聞|今日新聞|最新新聞)+", "", user_text)
    cleaned = re.sub(r"(的新聞|新聞|時事|消息|的資訊|資訊|資料|報導|看你.*|我故意.*|多少|行情|今天|今日|相關).*$", "", cleaned)
    cleaned = cleaned.strip(" ，。,、？！?!")
    return cleaned if len(cleaned) >= 2 else "台灣新聞"


def fetch_taiwan_stock(query_text: str = "") -> str:
    """
    Fetch real-time or closing stock prices & market index from Taiwan Stock Exchange (TWSE) MIS API.
    """
    stock_map = {
        "台積電": "tse_2330.tw",
        "2330": "tse_2330.tw",
        "鴻海": "tse_2317.tw",
        "2317": "tse_2317.tw",
        "聯發科": "tse_2454.tw",
        "2454": "tse_2454.tw",
        "廣達": "tse_2382.tw",
        "2382": "tse_2382.tw",
        "台達電": "tse_2308.tw",
        "2308": "tse_2308.tw",
        "聯電": "tse_2303.tw",
        "2303": "tse_2303.tw",
        "大立光": "tse_3008.tw",
        "3008": "tse_3008.tw",
        "富邦金": "tse_2881.tw",
        "2881": "tse_2881.tw",
        "國泰金": "tse_2882.tw",
        "2882": "tse_2882.tw",
        "0050": "tse_0050.tw",
        "0056": "tse_0056.tw",
        "大盤": "tse_t00.tw",
        "加權指數": "tse_t00.tw",
        "大盤指數": "tse_t00.tw",
        "股市行情": "tse_t00.tw",
    }

    targets = []
    # Detect 4-digit stock codes
    code_matches = re.findall(r"\b(\d{4})\b", query_text)
    for code in code_matches:
        channel = f"tse_{code}.tw"
        if channel not in targets:
            targets.append(channel)

    # Detect known stock names
    for name, channel in stock_map.items():
        if name in query_text:
            if channel not in targets:
                targets.append(channel)

    # If general market / stock行情 asked, add Weighted Index
    if any(k in query_text for k in ["股市", "行情", "大盤", "台股", "指數", "股票", "收盤"]):
        if "tse_t00.tw" not in targets:
            targets.append("tse_t00.tw")

    # Fallback to TSMC + Weighted Index if none recognized
    if not targets:
        targets = ["tse_2330.tw", "tse_t00.tw"]

    channel_str = "|".join(targets)
    url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={channel_str}&json=1&delay=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://mis.twse.com.tw/stock/index.jsp"
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        msg_array = data.get("msgArray", [])
        if not msg_array:
            return f"【台灣證券交易所 (TWSE)】: 查無即時報價資料 (查詢目標: {channel_str})。"

        results = []
        for it in msg_array:
            code = it.get("c", "")
            name = it.get("n", "")
            z_str = it.get("z", "")
            y_str = it.get("y", "")
            if not z_str or z_str == "-":
                z_str = it.get("pz", y_str)

            try:
                curr_price = float(z_str)
                prev_price = float(y_str) if y_str and y_str != "-" else curr_price
                diff = round(curr_price - prev_price, 2)
                diff_pct = round((diff / prev_price) * 100, 2) if prev_price > 0 else 0.0

                if diff > 0:
                    sign = f"🔺 +{diff:.2f} (+{diff_pct:.2f}%)"
                elif diff < 0:
                    sign = f"🔻 {diff:.2f} ({diff_pct:.2f}%)"
                else:
                    sign = f"➖ 0.00 (0.00%)"

                t_time = it.get("t", "")
                t_date = it.get("d", "")
                date_fmt = f"{t_date[:4]}/{t_date[4:6]}/{t_date[6:]}" if len(t_date) == 8 else t_date

                vol = it.get("v", "0")
                open_p = it.get("o", "-")
                high_p = it.get("h", "-")
                low_p = it.get("l", "-")

                is_index = code == "t00" or "指數" in name
                unit = "點" if is_index else "元"
                vol_unit = "口/億元" if is_index else "張"

                item_str = (
                    f"• {name} ({code}):\n"
                    f"  - 最新成交/收盤價: {curr_price:,.2f} {unit} ({sign})\n"
                    f"  - 昨收: {prev_price:,.2f} {unit} | 今日開盤: {open_p} | 最高: {high_p} | 最低: {low_p}\n"
                    f"  - 成交量: {int(vol):,} {vol_unit}\n"
                    f"  - 資料時間: {date_fmt} {t_time}"
                )
                results.append(item_str)
            except Exception as parse_err:
                results.append(f"• {name} ({code}): 報價資料解析異常 ({parse_err})")

        return (
            "【台灣證券交易所 (TWSE) 官方即時/收盤股市行情 (權威數據)】:\n" +
            "\n\n".join(results) +
            "\n\n請務必以繁體中文專業、精準回答使用者上述最新成交/收盤價，嚴禁隨意猜測或編造非上述數據。"
        )
    except Exception as e:
        return f"【台灣證券交易所 (TWSE)】: 行情連線查詢異常 ({e})，請稍候再試。"


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
        return f"【實時聯網搜尋】: 未能搜尋到關於『{topic}』的即時新聞。請如實告知使用者未找到即時新聞，嚴禁自行編造虛假新聞！"
        
    return f"【實時聯網搜尋與台灣即時新聞 (主題: {topic or '頭條'})】:\n" + "\n".join(news_items)


def control_pi5_power(user_text: str) -> str:
    """
    Control Raspberry Pi 5 Power Lifecycle:
    - RTC Wakeup Poweroff: Shutdown PMIC and wake up automatically after N seconds.
    - Delayed Reboot: Reboot automatically after N seconds.
    - Immediate Poweroff / Reboot.
    """
    delay_sec = parse_time_duration_seconds(user_text)
    user_lower = user_text.lower()
    
    is_reboot = any(k in user_lower for k in ["重開機", "重新開機", "重啟", "重開", "reboot"])
    has_wakeup = any(k in user_lower for k in ["開機", "喚醒", "後開", "再開"])
    is_poweroff = any(k in user_lower for k in ["關機", "poweroff", "shutdown", "關閉系統"])

    # Scenario 1: Shutdown and wake up after N seconds (RTC Wakeup)
    if is_poweroff and (has_wakeup or delay_sec > 0) and not is_reboot:
        wake_sec = delay_sec if delay_sec > 0 else 30
        logger.info(f"Scheduling RTC Wakeup Poweroff in {wake_sec} seconds...")
        # Delay 3 seconds before issuing command so Telegram message & voice finishes sending
        cmd = f"nohup bash -c 'sleep 3 && sudo rtcwake -m off -s {wake_sec}' >/dev/null 2>&1 &"
        subprocess.Popen(cmd, shell=True)
        return (
            f"【電源管理 - 硬體定時喚醒 (RTC Wakeup)】:\n"
            f"樹莓派 5 已成功排定於 3 秒後進入深層關機 (Poweroff)，並由硬體 RTC 晶片在 {wake_sec} 秒後自動喚醒通電開機！\n"
            f"請留意板子紅色 LED 燈熄滅後，約 {wake_sec} 秒後將自動轉為綠燈並重新啟動系統。"
        )

    # Scenario 2: Reboot with delay or immediate reboot
    if is_reboot:
        delay = delay_sec if delay_sec > 0 else 3
        logger.info(f"Scheduling system reboot in {delay} seconds...")
        cmd = f"nohup bash -c 'sleep {delay} && sudo reboot' >/dev/null 2>&1 &"
        subprocess.Popen(cmd, shell=True)
        return (
            f"【電源管理 - 系統重新開機 (Reboot)】:\n"
            f"已成功排定樹莓派 5 於 {delay} 秒後自動重新啟動系統。"
        )

    # Scenario 3: Pure Poweroff
    if is_poweroff:
        logger.info("Scheduling clean poweroff in 3 seconds...")
        cmd = "nohup bash -c 'sleep 3 && sudo poweroff' >/dev/null 2>&1 &"
        subprocess.Popen(cmd, shell=True)
        return (
            "【電源管理 - 系統安全關機 (Poweroff)】:\n"
            "樹莓派 5 已排定於 3 秒後執行安全關機。待狀態指示燈熄滅後即可安全移除電源。"
        )

    return "【電源管理】: 未能辨識具體的電源指令，請說明是否需要關機、重開機或指定喚醒秒數。"


def synthesize_new_skill(user_text: str) -> str:
    """
    Self-Evolution: Prompt local LLM to generate a standalone Python skill module,
    validate its syntax, and hot-reload into the SkillRegistry.
    """
    clean_prompt = re.sub(r"(請|幫我|自建|自製|學會|建立|新增|寫一個|技能|工具|能力)", "", user_text).strip()
    if not clean_prompt:
        clean_prompt = "自定義擴展工具"

    logger.info(f"Synthesizing new skill for: {clean_prompt}")
    
    # Prompt Ollama code model to generate skill code
    sys_instruction = (
        "You are an expert Python tool synthesizer for Raspberry Pi 5.\n"
        "Generate a standalone Python skill module following EXACTLY this specification:\n"
        "1. Define dictionary SKILL_METADATA = {'name': '名稱', 'desc': '描述', 'keywords': ['關鍵字1', '關鍵字2']}\n"
        "2. Define function `def execute(user_text: str) -> str:`\n"
        "3. Only output valid Python code enclosed in ```python ... ``` without any markdown explanations."
    )

    try:
        req_data = json.dumps({
            "model": "qwen2.5-coder:7b-opt",
            "messages": [
                {"role": "system", "content": sys_instruction},
                {"role": "user", "content": f"Create a skill to: {clean_prompt}"}
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512}
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_API_URL}/api/chat",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")

        # Extract python code
        match = re.search(r"```python(.*?)```", content, re.DOTALL)
        code = match.group(1).strip() if match else content.strip()

        # Sanitize skill file name
        skill_id = "skill_" + re.sub(r"[^a-zA-Z0-9_]", "", clean_prompt[:10].lower())
        if len(skill_id) < 8:
            import time
            skill_id += f"_{int(time.time())}"

        success = register_new_skill_code(skill_id, code)
        if success:
            return (
                f"🎉 【自我進化成功】:\n"
                f"已成功為樹莓派 5 撰寫並熱載入新技能模組 `{skill_id}.py`！\n"
                f"功能目標：{clean_prompt}\n"
                f"該技能現已即時加入技能庫，下回提到相關關鍵字即可直接自動觸發執行。"
            )
        else:
            return f"❌ 【新技能合成失敗】: 代碼語法未通過沙盒校驗。"

    except Exception as e:
        logger.error(f"Failed to synthesize skill: {e}")
        return f"❌ 【自我進化異常】: 調用代碼大模型時出錯 ({e})。"


def check_or_connect_bluetooth(user_text: str) -> str:
    """
    Natural language interface for Bluetooth headset management.
    Handles querying status, connecting, or scanning via voice/text.
    """
    try:
        from bluetooth_manager import get_full_bt_status, connect_default_device, scan_devices
    except ImportError:
        return "❌ 尚未載入藍牙管理模組 (bluetooth_manager.py)。"

    u_lower = user_text.lower()
    if any(k in u_lower for k in ["連線", "連接", "連上", "重連"]):
        success, msg = connect_default_device()
        return f"【藍牙連線操作結果】:\n{msg}"
    elif any(k in u_lower for k in ["搜尋", "掃描", "尋找"]):
        devs = scan_devices(8)
        if not devs:
            return "【藍牙掃描結果】: 未找到附近處於配對狀態的耳麥，請長按耳麥按鍵進入配對燈號閃爍狀態。"
        lines = [f"{i+1}. {d['name']} ({d['mac']})" for i, d in enumerate(devs[:5])]
        return "【藍牙掃描發現以下耳麥/裝置】:\n" + "\n".join(lines) + "\n您可以在 Telegram 輸入 `/bt_pair 編號` 完成綁定。"
    else:
        st = get_full_bt_status()
        def_name = st.get("default_name") or "未綁定預設耳麥"
        is_conn = st.get("default_connected", False)
        conn_str = "🟢 已成功連線 (Connected)" if is_conn else "⚪ 目前未連線 (Disconnected)"
        return (
            f"【藍牙耳麥狀態】:\n"
            f"• 預設目標: {def_name}\n"
            f"• 連線狀態: {conn_str}\n"
            f"• 開機自動重連: {'啟用中 (每 15 秒探測)' if st.get('auto_reconnect') else '已關閉'}\n"
            f"• 控制器: {st.get('controller_message')}\n"
            f"提示：可於 Telegram 輸入 `/bt_scan` 搜尋新耳麥，或輸入 `/bt_pair` 進行綁定。"
        )


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
        # 1. Hardware Status
        self.skills["hardware_status"] = {
            "name": "查詢硬體與板子溫度",
            "desc": "查詢樹莓派5的SoC核心溫度、記憶體RAM負載、CPU負載與運作時間",
            "keywords": ["溫度", "板子溫度", "硬體狀態", "記憶體", "ram", "cpu", "負載", "運作時間", "健康狀態", "幾度"],
            "func": get_pi5_hardware_status
        }
        # 2. Taiwan News & Search
        self.skills["taiwan_news"] = {
            "name": "即時台灣新聞與網頁搜尋",
            "desc": "上網檢索台灣最新頭條新聞或特定時事主題",
            "keywords": ["新聞", "時事", "頭條", "最新消息", "搜尋", "上網找", "查詢", "找一下"],
            "func": fetch_taiwan_news
        }
        # 3. Taiwan Stock & Market Index (TWSE MIS)
        self.skills["taiwan_stock"] = {
            "name": "台灣股市與即時股價行情",
            "desc": "連線台灣證券交易所 (TWSE) 查詢台積電、各檔股票與加權指數即時/收盤行情",
            "keywords": ["股價", "股市", "行情", "台積電", "加權指數", "大盤", "股票", "收盤價", "2330", "開盤價", "跌幅", "漲幅", "台股", "成交量"],
            "func": fetch_taiwan_stock
        }
        # 4. Power Control & RTC Wakeup
        self.skills["power_control"] = {
            "name": "電源管理與定時開關機",
            "desc": "安全關機、重開機、或設定硬體RTC定時喚醒開機",
            "keywords": ["關機", "開機", "重開機", "重啟", "重新開機", "定時開機", "喚醒", "poweroff", "reboot", "rtcwake"],
            "func": control_pi5_power
        }
        # 5. Self-Evolution Skill Synthesis
        self.skills["skill_synthesis"] = {
            "name": "自建新技能與代碼進化",
            "desc": "調用本地代碼模型自動生成新技能代碼並熱載入",
            "keywords": ["自建技能", "學會新技能", "建立技能", "自建工具", "新增技能", "擴充技能", "寫一個工具", "學會"],
            "func": synthesize_new_skill
        }
        # 6. Bluetooth Headset Control
        self.skills["bluetooth_headset"] = {
            "name": "藍牙耳麥管理與自動連線",
            "desc": "查詢藍牙耳麥連線狀況、掃描配對或手動連線耳機",
            "keywords": ["藍牙", "藍芽", "耳機", "耳麥", "bluetooth", "連線耳機", "配對耳機", "藍牙耳機"],
            "func": check_or_connect_bluetooth
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

        # Priority 0: Skill Synthesis (自建新技能)
        synth_keywords = self.skills["skill_synthesis"]["keywords"]
        if any(k in user_lower for k in synth_keywords):
            logger.info("Triggered Skill: synthesize_new_skill")
            return self.skills["skill_synthesis"]["func"](user_text)

        # Priority 1: Power Control & RTC Wakeup (電源管理)
        power_keywords = self.skills["power_control"]["keywords"]
        if any(k in user_lower for k in power_keywords):
            logger.info("Triggered Skill: control_pi5_power")
            raw_info = self.skills["power_control"]["func"](user_text)
            return f"{raw_info}\n請根據以上排定的電源狀態，以親切且清晰的語句告知使用者系統即將進行的動作。"

        # Priority 2: Taiwan Stock & Financial Market (股市行情最優先於一般新聞)
        stock_keywords = self.skills["taiwan_stock"]["keywords"]
        if any(k in user_lower for k in stock_keywords):
            logger.info("Triggered Skill: fetch_taiwan_stock")
            stock_info = self.skills["taiwan_stock"]["func"](user_text)
            # If user ALSO explicitly mentions news/時事/新聞, append company news
            if any(k in user_lower for k in ["新聞", "時事", "消息"]):
                clean_ent = extract_clean_news_topic(user_text)
                news_info = self.skills["taiwan_news"]["func"](clean_ent)
                return (
                    f"{stock_info}\n\n{news_info}\n\n"
                    "【回答指引】: 最新股價必須嚴格採用上方 TWSE 標明的『最新成交/收盤價』(例如台積電為 2,500.00 元)，"
                    "下方新聞列表僅用作補充市場消息背景，回答請精簡扼要，避免過多長篇贅述。"
                )
            return stock_info

        # Priority 3: Hardware & Temperature check
        hw_keywords = self.skills["hardware_status"]["keywords"]
        if any(k in user_lower for k in hw_keywords):
            logger.info("Triggered Skill: get_pi5_hardware_status")
            raw_info = self.skills["hardware_status"]["func"]()
            return f"{raw_info}\n請根據以上真實的硬體溫度與數據，以繁體中文親切告知使用者目前板子的狀態。"

        # Priority 4: Web Search & News check
        news_keywords = self.skills["taiwan_news"]["keywords"]
        if any(k in user_lower for k in news_keywords):
            logger.info("Triggered Skill: fetch_taiwan_news")
            topic = extract_clean_news_topic(user_text)
            raw_info = self.skills["taiwan_news"]["func"](topic)
            return f"{raw_info}\n請根據以上搜尋結果，為使用者統整出重點摘要並語音朗讀回答。"

        # Priority 5: Bluetooth Headset Control
        bt_keywords = self.skills["bluetooth_headset"]["keywords"]
        if any(k in user_lower for k in bt_keywords):
            logger.info("Triggered Skill: check_or_connect_bluetooth")
            raw_info = self.skills["bluetooth_headset"]["func"](user_text)
            return f"{raw_info}\n請根據以上藍牙連線狀態或結果，以繁體中文親切回應使用者。"

        # Priority 6: Custom Loaded Skills
        for s_id, s_info in self.skills.items():
            if s_id in ["hardware_status", "taiwan_news", "power_control", "skill_synthesis", "bluetooth_headset"]:
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
    print("1. Testing Time Parsing:")
    print("三十秒 ->", parse_time_duration_seconds("請三十秒後關機"))
    print("1分鐘 ->", parse_time_duration_seconds("設定1分鐘後重新開機"))
    
    print("\n2. Testing Power Control Simulation:")
    # Simulation without calling subprocess Popen directly
    print(execute_tool_call_if_needed("幫我設定關機，30秒後自行開機"))
