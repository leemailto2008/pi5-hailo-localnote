#!/usr/bin/env python3
"""LocalNote 測試會議腳本產生器 (deterministic)。

由議題模板 (TOPICS) + 固定 seed 的填值組成指定長度的多人會議，每個議題內建可核對事實 (數字/日期/金額/人名)，
結尾有「總結」段重述決議與待辦，供 e2e 評分：
  - facts          : 逐字稿應出現的全部事實 (含中文數字/阿拉伯數字等同義寫法)
  - recap_facts    : 結尾總結重述的決議/待辦關鍵事實 (summary 應涵蓋)
  - topic_keywords : 各議題關鍵字 (summary 議題覆蓋率)

usage:
  meeting_gen.py --minutes 1 3 5 10 15 30 --out tools/e2e/fixtures
"""
import argparse
import json
import os
import random

CHARS_PER_SEC = 3.85     # edge-tts zh-TW 實測 (不含標點)
LINE_GAP_S = 0.6
TOPIC_GAP_S = 1.2
LEAD_S, TAIL_S = 1.0, 1.5
MAX_AUDIO_S = 14390      # /api/record seconds 上限 14400 (4 小時)，保留 e2e 前後緩衝
SHORT_MINUTES = (1, 3, 5, 10, 15, 30)
LONG_MINUTES = (45, 60, 90, 120, 180, 240)
# 長會議系列共用同一個 seed：議題序列互為前綴 (4 小時版的前段 = 3 小時版的內容)，逐句 TTS 快取可共用，
# 總共只需合成約 4 小時份量的語音 (不必 12.25 小時)。短會議沿用各自的 seed，既有 fixture 與音檔不變。
LONG_SERIES_SEED = 45007
MAX_TOPICS = 1000        # 4 小時約 660 個議題 (舊上限 200 只夠約 70 分鐘)

SPEAKERS = {"A": "zh-TW-HsiaoChenNeural", "B": "zh-TW-YunJheNeural", "C": "zh-TW-HsiaoYuNeural"}

DIGITS = "零一二三四五六七八九"


def zh_num(n: int, colloquial: bool = True) -> str:
    """0..99,999,999 的中文數字；colloquial=True 時首位 2 讀「兩」(兩百、兩千)，10..19 讀「十X」。"""
    if n == 0:
        return "零"
    if n >= 10000:
        hi, lo = divmod(n, 10000)
        s = zh_num(hi, colloquial) + "萬"
        if lo:
            s += ("零" if lo < 1000 else "") + zh_num(lo, colloquial)
        return s
    units = ["千", "百", "十", ""]
    parts, zero = [], False
    ds = [n // 1000 % 10, n // 100 % 10, n // 10 % 10, n % 10]
    first = next(i for i, d in enumerate(ds) if d)
    for i in range(first, 4):
        d = ds[i]
        if d == 0:
            zero = True
            continue
        if zero:
            parts.append("零")
            zero = False
        if i == first and d == 2 and units[i] in ("千", "百") and colloquial:
            parts.append("兩" + units[i])
        elif i == first and d == 1 and units[i] == "十":
            parts.append("十")
        else:
            parts.append(DIGITS[d] + units[i])
    return "".join(parts)


def num_aliases(n: int, unit: str = "") -> list:
    al = {f"{n}{unit}", f"{n:,}{unit}", zh_num(n) + unit, zh_num(n, colloquial=False) + unit}
    if unit == "台":
        al |= {a.replace("台", "臺") for a in al}
    return sorted(al)


def pct_aliases(n: int) -> list:
    return [f"{n}%", f"{n}％", f"百分之{zh_num(n)}", f"百分之{n}"]


def date_text(m: int, d: int) -> str:
    return f"{zh_num(m)}月{zh_num(d)}日"


def date_aliases(m: int, d: int) -> list:
    return [f"{m}月{d}", f"{m}/{d}", f"{zh_num(m)}月{zh_num(d)}", f"{m}月{zh_num(d)}"]


def money_aliases(n: int) -> list:
    return [f"{n}萬", zh_num(n) + "萬", zh_num(n, False) + "萬", f"{n * 10000:,}", f"{n * 10000}"]


POOL = {
    "company": ["大成電子", "宏遠科技", "聯華精密", "東元工業", "新光材料", "永豐光電", "台揚通訊", "華新機電"],
    "person": ["王經理", "李協理", "張工程師", "林主任", "黃專員", "吳副理", "劉課長", "蔡組長", "楊工程師", "許經理"],
    "product": ["智慧電錶", "藍牙耳機", "工業閘道器", "溫控模組", "車用鏡頭", "電源供應器"],
    "city": ["台北", "新竹", "台中", "高雄", "桃園", "台南"],
    "dept": ["研發部", "業務部", "生產部", "品保部", "資訊部"],
    "weekday": ["週一", "週二", "週三", "週四", "週五"],
    "part": ["電源晶片", "外殼模具", "天線模組", "電池芯"],
    "issue": ["藍牙斷線", "開機過慢", "韌體當機", "耗電過高"],
    "defect": ["按鍵失靈", "外殼刮傷", "接觸不良", "螢幕亮點"],
    "course": ["資安", "專案管理", "品質工程", "簡報技巧"],
    "system": ["新版 ERP", "倉儲管理", "客服工單", "人資差勤"],
    "location": ["二號倉庫", "三樓產線", "地下停車場", "裝配區"],
}


def topic_library():
    """每個模板回傳 (keyword, lines[(spk, text)], facts{label: aliases}, recap{decision, todo, facts})."""
    def order(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); qty = r.choice([120, 150, 200, 300, 500]); pct = r.choice([5, 6, 8, 10, 12])
        trial = r.choice([30, 50, 60, 80]); wd = r.choice(POOL["weekday"])
        lines = [("A", f"第{zh_num(k)}個議題是客戶訂單。{s['city']}的{s['company']}希望在{date_text(m, d)}前拿到{zh_num(qty)}台{s['product']}樣品。"),
                 ("B", f"他們對單價有意見，希望降價百分之{zh_num(pct)}。"),
                 ("C", f"我建議先簽{zh_num(trial)}台的試單，價格維持新報價，等試用結果再談。"),
                 ("A", f"同意，這件事由{s['person']}負責，{wd}前回覆客戶。")]
        facts = {"樣品數量": num_aliases(qty, "台"), "交期": date_aliases(m, d), "降價": pct_aliases(pct),
                 "試單數量": num_aliases(trial, "台"), "負責人": [s["person"]], "客戶": [s["company"]]}
        return "訂單", lines, facts, {"decision": f"先簽{zh_num(trial)}台試單", "todo": f"{s['person']}{wd}前回覆{s['company']}",
                                       "facts": {"試單數量": facts["試單數量"], "負責人": [s["person"]]}}

    def firmware(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); n = r.choice([3, 5, 10]); n2 = r.choice([20, 30, 50, 100])
        lines = [("B", f"第{zh_num(k)}個議題是{s['product']}的韌體更新。目前回報的問題是{s['issue']}，修正版預計{date_text(m, d)}上線。"),
                 ("A", f"測試的部分請{s['person']}安排，至少要做{zh_num(n)}輪回歸測試。"),
                 ("C", f"上線前先在{zh_num(n2)}台設備上試跑三天，沒有問題再全面更新。")]
        facts = {"上線日期": date_aliases(m, d), "回歸輪數": num_aliases(n, "輪"), "試跑台數": num_aliases(n2, "台"),
                 "負責人": [s["person"]], "問題": [s["issue"]]}
        return "韌體", lines, facts, {"decision": f"韌體修正版{date_text(m, d)}上線", "todo": f"{s['person']}安排回歸測試",
                                       "facts": {"上線日期": facts["上線日期"], "負責人": [s["person"]]}}

    def budget(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); money = r.choice([80, 120, 150, 200, 350]); hw = r.choice([20, 30, 40, 50]); hc = r.choice([2, 3, 4])
        lines = [("A", f"第{zh_num(k)}個議題是預算。明年{s['dept']}的預算是{zh_num(money)}萬元，其中硬體佔{zh_num(hw)}萬元。"),
                 ("B", f"人力的部分要再增加{zh_num(hc)}位工程師。"),
                 ("A", f"預算表請{s['person']}在{date_text(m, d)}前送到財務部。")]
        facts = {"總預算": money_aliases(money), "硬體預算": money_aliases(hw), "增加人力": num_aliases(hc, "位"),
                 "截止日": date_aliases(m, d), "負責人": [s["person"]]}
        return "預算", lines, facts, {"decision": f"{s['dept']}明年預算{zh_num(money)}萬元", "todo": f"{s['person']}送預算表",
                                       "facts": {"總預算": facts["總預算"], "負責人": [s["person"]]}}

    def hiring(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); hc = r.choice([2, 3, 5]); n = r.choice([8, 12, 15]); n2 = r.choice([3, 4, 6])
        lines = [("C", f"第{zh_num(k)}個議題是招募。{s['dept']}目前缺{zh_num(hc)}個職缺，已經面試了{zh_num(n)}位候選人。"),
                 ("A", f"下週會再安排{zh_num(n2)}場面試，由{s['person']}主持。"),
                 ("B", f"希望在{date_text(m, d)}前完成報到。")]
        facts = {"職缺數": num_aliases(hc, "個"), "已面試": num_aliases(n, "位"), "面試場次": num_aliases(n2, "場"),
                 "報到日": date_aliases(m, d), "負責人": [s["person"]]}
        return "招募", lines, facts, {"decision": f"{date_text(m, d)}前完成報到", "todo": f"{s['person']}主持面試",
                                       "facts": {"報到日": facts["報到日"], "負責人": [s["person"]]}}

    def supplier(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); days = r.choice([7, 10, 14, 21]); pct = r.choice([85, 90, 92, 95]); wd = r.choice(POOL["weekday"])
        lines = [("B", f"第{zh_num(k)}個議題是供應商。{s['company']}的{s['part']}交期延到{date_text(m, d)}，比原本晚了{zh_num(days)}天。"),
                 ("C", f"替代料已經在驗證，目前良率是百分之{zh_num(pct)}。"),
                 ("A", f"請{s['person']}跟對方確認，{wd}回報。")]
        facts = {"新交期": date_aliases(m, d), "延遲天數": num_aliases(days, "天"), "良率": pct_aliases(pct),
                 "供應商": [s["company"]], "負責人": [s["person"]]}
        return "供應商", lines, facts, {"decision": "替代料持續驗證", "todo": f"{s['person']}{wd}回報{s['company']}的交期",
                                         "facts": {"負責人": [s["person"]], "供應商": [s["company"]]}}

    def quality(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); n = r.choice([6, 9, 12, 18]); p1 = r.choice([3, 4, 5]); p2 = 1
        lines = [("A", f"第{zh_num(k)}個議題是品質異常。上個月{s['product']}的客訴有{zh_num(n)}件，主要是{s['defect']}。"),
                 ("B", f"產線已經加開一站檢查，不良率從百分之{zh_num(p1)}降到百分之{zh_num(p2)}。"),
                 ("C", f"{s['person']}會在{date_text(m, d)}前提出改善報告。")]
        facts = {"客訴件數": num_aliases(n, "件"), "原不良率": pct_aliases(p1), "新不良率": pct_aliases(p2),
                 "報告期限": date_aliases(m, d), "負責人": [s["person"]], "缺陷": [s["defect"]]}
        return "品質", lines, facts, {"decision": "產線加開一站檢查", "todo": f"{s['person']}在{date_text(m, d)}前提出改善報告",
                                       "facts": {"負責人": [s["person"]], "報告期限": facts["報告期限"]}}

    def expo(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); money = r.choice([15, 25, 30, 45]); n = r.choice([4, 6, 8]); hc = r.choice([3, 5, 6]); wd = r.choice(POOL["weekday"])
        lines = [("C", f"第{zh_num(k)}個議題是行銷活動。{date_text(m, d)}在{s['city']}有一場展覽，攤位費{zh_num(money)}萬元。"),
                 ("A", f"預計展出{zh_num(n)}款產品，需要{zh_num(hc)}位同仁支援。"),
                 ("B", f"文宣請{s['person']}負責，{wd}前交初稿。")]
        facts = {"展覽日期": date_aliases(m, d), "攤位費": money_aliases(money), "展出款數": num_aliases(n, "款"),
                 "支援人數": num_aliases(hc, "位"), "負責人": [s["person"]], "城市": [s["city"]]}
        return "展覽", lines, facts, {"decision": f"參加{s['city']}展覽", "todo": f"{s['person']}{wd}前交文宣初稿",
                                       "facts": {"負責人": [s["person"]], "城市": [s["city"]]}}

    def travel(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); days = r.choice([3, 4, 5]); money = r.choice([8, 12, 15])
        city2 = r.choice(["東京", "新加坡", "上海", "首爾", "曼谷"])
        lines = [("A", f"第{zh_num(k)}個議題是出差安排。{s['person']}下個月要去{city2}拜訪{s['company']}，行程{zh_num(days)}天。"),
                 ("B", f"預算抓{zh_num(money)}萬元，機票已經訂好了。"),
                 ("C", f"回來後在{date_text(m, d)}的週會上報告。")]
        facts = {"出差天數": num_aliases(days, "天"), "出差預算": money_aliases(money), "報告日": date_aliases(m, d),
                 "出差人": [s["person"]], "目的地": [city2]}
        return "出差", lines, facts, {"decision": f"{s['person']}出差{city2}", "todo": f"{s['person']}週會報告",
                                       "facts": {"出差人": [s["person"]], "目的地": [city2]}}

    def training(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); n = r.choice([20, 30, 40]); n2 = n - r.choice([5, 8, 12]); wd = r.choice(POOL["weekday"])
        lines = [("B", f"第{zh_num(k)}個議題是教育訓練。{date_text(m, d)}有一場{s['course']}課程，名額{zh_num(n)}人。"),
                 ("C", f"目前已經報名{zh_num(n2)}人，還剩{zh_num(n - n2)}個名額。"),
                 ("A", f"請各部門在{wd}前把名單交給{s['person']}。")]
        facts = {"課程日期": date_aliases(m, d), "名額": num_aliases(n, "人"), "已報名": num_aliases(n2, "人"),
                 "剩餘名額": num_aliases(n - n2, "個"), "負責人": [s["person"]], "課程": [s["course"]]}
        return "訓練", lines, facts, {"decision": f"{s['course']}課程{date_text(m, d)}開課", "todo": f"各部門{wd}前把名單交給{s['person']}",
                                       "facts": {"課程": [s["course"]], "負責人": [s["person"]]}}

    def golive(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); pct = r.choice([80, 85, 90, 95]); hours = r.choice([2, 3, 4, 6])
        lines = [("C", f"第{zh_num(k)}個議題是{s['system']}系統上線。資料轉換已經完成百分之{zh_num(pct)}。"),
                 ("A", f"正式切換訂在{date_text(m, d)}晚上，預計停機{zh_num(hours)}小時。"),
                 ("B", f"{s['person']}負責通知所有使用者。")]
        facts = {"轉換進度": pct_aliases(pct), "切換日": date_aliases(m, d), "停機時數": num_aliases(hours, "小時"),
                 "負責人": [s["person"]]}
        return "上線", lines, facts, {"decision": f"{date_text(m, d)}晚上切換系統", "todo": f"{s['person']}通知使用者",
                                       "facts": {"切換日": facts["切換日"], "負責人": [s["person"]]}}

    def stock(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); qty = r.choice([800, 1200, 1500, 2000]); weeks = r.choice([3, 4, 6]); qty2 = r.choice([500, 1000, 3000])
        lines = [("A", f"第{zh_num(k)}個議題是庫存。{s['product']}目前庫存{zh_num(qty)}台，大約可以撐{zh_num(weeks)}週。"),
                 ("B", f"下一批{zh_num(qty2)}台預計{date_text(m, d)}到貨。"),
                 ("C", f"如果{s['city']}的訂單提前，要請{s['person']}調貨。")]
        facts = {"庫存量": num_aliases(qty, "台"), "可撐週數": num_aliases(weeks, "週"), "下批數量": num_aliases(qty2, "台"),
                 "到貨日": date_aliases(m, d), "負責人": [s["person"]]}
        return "庫存", lines, facts, {"decision": f"下一批{zh_num(qty2)}台{date_text(m, d)}到貨", "todo": f"{s['person']}視訂單調貨",
                                       "facts": {"下批數量": facts["下批數量"], "負責人": [s["person"]]}}

    def safety(k, r, s):
        m, d = r.randint(1, 12), r.randint(1, 28); n = r.choice([4, 6, 8, 10])
        lines = [("B", f"第{zh_num(k)}個議題是工安。上週{s['location']}發生一起小事故，沒有人受傷。"),
                 ("A", f"已經加裝{zh_num(n)}支警示燈，並在{date_text(m, d)}前完成全員複訓。"),
                 ("C", f"負責人是{s['person']}。")]
        facts = {"警示燈": num_aliases(n, "支"), "複訓期限": date_aliases(m, d), "地點": [s["location"]], "負責人": [s["person"]]}
        return "工安", lines, facts, {"decision": f"{date_text(m, d)}前完成全員複訓", "todo": f"{s['person']}負責工安複訓",
                                       "facts": {"複訓期限": facts["複訓期限"], "負責人": [s["person"]]}}

    return [order, firmware, budget, hiring, supplier, quality, expo, travel, training, golive, stock, safety]


def est_seconds(lines) -> float:
    chars = sum(sum(1 for c in t if c.isalnum()) for _, t in lines)
    return chars / CHARS_PER_SEC + LINE_GAP_S * len(lines)


def build_meeting(minutes: float, seed: int = None, target_s: float = None) -> dict:
    """target_s：覆寫目標秒數 (gen_test_audio.py 依實際合成長度校正時使用)。"""
    target = min(target_s if target_s else minutes * 60.0, MAX_AUDIO_S)
    if seed is None:
        seed = LONG_SERIES_SEED if minutes >= min(LONG_MINUTES) else int(minutes * 1000) + 7
    r = random.Random(seed)
    lib = topic_library()
    order = list(range(len(lib)))
    r.shuffle(order)
    opening = [("A", "好，我們開始今天的週會，今天有幾個議題要討論。")]
    topics, budget_s = [], target - LEAD_S - TAIL_S - est_seconds(opening)
    k = 0
    while True:
        k += 1
        fn = lib[order[(k - 1) % len(lib)]]
        slots = {key: r.choice(vals) for key, vals in POOL.items()}
        kw, lines, facts, recap = fn(k, r, slots)
        n_recap = min(3, k)
        recap_est = est_seconds([("A", "好，最後總結一下。")] + [("A", t["recap"]["decision"] + "，" + t["recap"]["todo"] + "。")
                                                             for t in (topics + [{"recap": recap}])[:n_recap]] + [("A", "今天就到這裡，謝謝大家。")])
        used = sum(t["est_s"] for t in topics) + TOPIC_GAP_S * len(topics)
        # 加入此議題後若超出預算不到半個議題長度仍加入 (取最接近目標的長度)；實際長度由合成後量測再裁切
        if topics and used + est_seconds(lines) + TOPIC_GAP_S + recap_est > budget_s + 0.5 * est_seconds(lines):
            break
        topics.append({"index": k, "keyword": kw, "lines": [{"spk": a, "text": b} for a, b in lines],
                       "facts": {f"T{k}{kw}-{lab}": al for lab, al in facts.items()}, "recap": recap, "est_s": round(est_seconds(lines), 1)})
        if len(topics) >= MAX_TOPICS:
            break
    recap_topics = topics[:3] if len(topics) <= 3 else [topics[0], topics[len(topics) // 2], topics[-1]]
    closing = [{"spk": "A", "text": "好，最後總結一下。"}]
    recap_facts = {}
    for i, t in enumerate(recap_topics):
        closing.append({"spk": "BC"[i % 2], "text": f"{t['keyword']}方面，決議是{t['recap']['decision']}，待辦事項是{t['recap']['todo']}。"})
        for lab, al in t["recap"]["facts"].items():
            recap_facts[f"T{t['index']}{t['keyword']}-{lab}"] = al
    closing.append({"spk": "A", "text": "今天就到這裡，謝謝大家。"})
    segments = [{"name": "opening", "lines": [{"spk": a, "text": b} for a, b in opening]}] + \
               [{"name": f"T{t['index']}{t['keyword']}", "lines": t["lines"]} for t in topics] + [{"name": "closing", "lines": closing}]
    facts = {}
    for t in topics:
        facts.update(t["facts"])
    est = LEAD_S + TAIL_S + sum(est_seconds([(l["spk"], l["text"]) for l in s["lines"]]) for s in segments) + TOPIC_GAP_S * (len(segments) - 1)
    return {"schema": 1, "title": f"測試週會 {minutes:g} 分鐘", "minutes": minutes, "target_s": round(target, 1), "est_s": round(est, 1),
            "speakers": SPEAKERS, "gaps": {"lead_s": LEAD_S, "tail_s": TAIL_S, "line_s": LINE_GAP_S, "segment_s": TOPIC_GAP_S},
            "segments": segments, "facts": facts, "recap_facts": recap_facts,
            "topic_keywords": sorted({t["keyword"] for t in topics}), "n_topics": len(topics),
            "lines": [[l["spk"], l["text"]] for s in segments for l in s["lines"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", nargs="+", type=float, default=list(SHORT_MINUTES), help=f"長會議：{' '.join(map(str, LONG_MINUTES))}")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for m in a.minutes:
        mt = build_meeting(m)
        p = os.path.join(a.out, f"meeting_{m:g}min.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(mt, f, ensure_ascii=False, indent=1)
        print(f"{p}: topics={mt['n_topics']:2d} lines={len(mt['lines']):3d} facts={len(mt['facts']):3d} "
              f"recap_facts={len(mt['recap_facts'])} est={mt['est_s']:.0f}s target={mt['target_s']:.0f}s")


if __name__ == "__main__":
    main()
