#!/usr/bin/env python3
import urllib.request
import json
import re
from typing import Dict, Any, List, Optional

STOCK_MAP = {
    "台積電": "tse_2330.tw",
    "鴻海": "tse_2317.tw",
    "聯發科": "tse_2454.tw",
    "廣達": "tse_2382.tw",
    "台達電": "tse_2308.tw",
    "聯電": "tse_2303.tw",
    "大立光": "tse_3008.tw",
    "富邦金": "tse_2881.tw",
    "國泰金": "tse_2882.tw",
    "0050": "tse_0050.tw",
    "0056": "tse_0056.tw",
    "大盤": "tse_t00.tw",
    "加權指數": "tse_t00.tw",
    "股市行情": "tse_t00.tw",
}

def fetch_taiwan_stock(query_text: str = "") -> str:
    """
    Query real-time or closing stock quotes from Taiwan Stock Exchange (TWSE) MIS API.
    """
    targets = []
    text_lower = query_text.lower()

    # Detect stock codes like 2330
    code_matches = re.findall(r"\b(\d{4})\b", query_text)
    for code in code_matches:
        targets.append(f"tse_{code}.tw")

    # Detect known stock names
    for name, channel in STOCK_MAP.items():
        if name in query_text:
            if channel not in targets:
                targets.append(channel)

    # If asking for general market / stock行情
    if any(k in query_text for k in ["股市", "行情", "大盤", "台股", "指數"]):
        if "tse_t00.tw" not in targets:
            targets.append("tse_t00.tw")

    # Default to TSMC and Weighted Index if none specified
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
            # 'z' is last trade price, if '-' or empty fallback to 'y' (yesterday's close) or 'b'
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
                    f"  - 最新報價: {curr_price:,.2f} {unit} ({sign})\n"
                    f"  - 昨收: {prev_price:,.2f} {unit} | 開盤: {open_p} | 最高: {high_p} | 最低: {low_p}\n"
                    f"  - 成交量: {int(vol):,} {vol_unit}\n"
                    f"  - 資料時間: {date_fmt} {t_time}"
                )
                results.append(item_str)
            except Exception as parse_err:
                results.append(f"• {name} ({code}): 報價資料解析異常 ({parse_err})")

        return (
            "【台灣證券交易所 (TWSE) 官方即時/收盤股市行情】:\n" +
            "\n\n".join(results) +
            "\n\n請以繁體中文專業、精準回答使用者上述真實股市與股價數據，嚴禁隨意猜測或編造非上述數據。"
        )
    except Exception as e:
        return f"【台灣證券交易所 (TWSE)】: 行情連線查詢異常 ({e})，請稍候再試。"

if __name__ == "__main__":
    test_q = "幫我查一下,台灣新聞裡面,台積電,9月23號,今天的股價多少,還有股市行情。"
    print(fetch_taiwan_stock(test_q))
