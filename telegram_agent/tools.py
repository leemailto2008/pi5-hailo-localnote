# telegram_agent/tools.py
"""
Real-time tools for Raspberry Pi 5 Telegram Voice AI Agent.
Provides web search, Taiwan real-time news retrieval, and system diagnostic tools.
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("Pi5Tools")

def fetch_taiwan_news(topic: str = "") -> List[Dict[str, str]]:
    """
    Fetch real-time news from Google News Taiwan RSS.
    If topic is provided, search specific keywords, otherwise get Top Taiwan Headlines.
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
            pubdate_elem = it.find("pubDate")
            desc_elem = it.find("description")
            
            title = title_elem.text if title_elem is not None else ""
            pub_date = pubdate_elem.text if pubdate_elem is not None else ""
            # Clean HTML tags from description if any
            desc = ""
            if desc_elem is not None and desc_elem.text:
                desc = re.sub(r'<[^>]+>', '', desc_elem.text).strip()
            
            if title:
                news_items.append({
                    "title": title,
                    "date": pub_date,
                    "snippet": desc[:150]
                })
        logger.info(f"Fetched {len(news_items)} real-time news items for query: '{topic}'")
    except Exception as e:
        logger.error(f"Error fetching Taiwan news: {e}")
        
    return news_items


def check_web_search_intent(user_text: str) -> Optional[str]:
    """
    Detect if the user prompt requires web search or real-time news.
    Returns the search query keyword if intent is matched, else None.
    """
    patterns = [
        r"(?:上網|網路|google|網路上)?(?:搜尋|查詢|找一下|查一下|找找)(.+)",
        r"(?:今天|今日|最新|最近)(?:的)?(?:台灣)?(?:新聞|消息|頭條|時事)",
        r"(?:新聞|時事|頭條)(?:有什麼|報導|資訊)",
        r"查(?:一下|看)?(.+?)(?:新聞|資訊|天氣)"
    ]
    
    # 1. Direct news trigger
    if any(k in user_text for k in ["新聞", "時事", "頭條", "最新消息"]):
        # Extract keyword if possible
        clean = re.sub(r"(請|幫我|上網|搜尋|查詢|看|找|今天的|今日的|台灣的|新聞|時事)", "", user_text).strip()
        return clean if clean else "台灣新聞"
        
    # 2. General search trigger
    for pat in patterns:
        m = re.search(pat, user_text)
        if m and m.groups() and m.group(1):
            q = m.group(1).strip(" ，。？！")
            if len(q) > 1:
                return q

    return None


def execute_tool_call_if_needed(user_text: str) -> Optional[str]:
    """
    Check user intent, execute real-time tools, and return formatted tool context.
    """
    search_query = check_web_search_intent(user_text)
    if search_query:
        logger.info(f"Triggering Web Search tool for query: '{search_query}'")
        news = fetch_taiwan_news(search_query)
        if news:
            lines = [f"【實時聯網搜尋與台灣即時新聞資訊 (主題: {search_query})】:"]
            for i, n in enumerate(news, 1):
                lines.append(f"{i}. {n['title']}")
                if n.get("snippet"):
                    lines.append(f"   摘要: {n['snippet']}")
            lines.append("請根據上述最新的真實新聞資訊，以繁體中文為使用者統整出重點摘要並語音朗讀回覆。")
            return "\n".join(lines)
        else:
            return "【實時聯網搜尋】: 嘗試搜尋但未能即時獲取外部結果，請如實告知使用者。"
            
    return None

if __name__ == "__main__":
    print("Testing tools.py...")
    res = execute_tool_call_if_needed("幫我上網搜尋今天台灣的新聞9月21日")
    print("Result:\n", res)
