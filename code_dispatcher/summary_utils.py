# code_dispatcher/summary_utils.py
"""會議紀要 (LLM 輸出) 後處理，純函式，供 meeting_minutes 使用並可離線單元測試。"""
import re
from typing import Dict, List, Optional

# 小模型 (qwen3:1.7b) 的紀要偶爾混出簡體字：2026-10-29 檢查 228 的 29 份 e2e／實錄紀要，11 份含簡體 (应、许、条、温…)，
# 例如「应商」讓議題關鍵字比對落空。逐字稿已由 Whisper 引擎轉成台灣繁體 (OpenCC s2twp)，紀要也走同一套轉換；沒有 OpenCC 時原樣輸出。
_CC = None


def to_taiwan_traditional(text: str) -> str:
    global _CC
    if not text:
        return text or ""
    if _CC is None:
        try:
            import opencc
            _CC = opencc.OpenCC("s2twp.json")
        except Exception:
            _CC = False
    if not _CC:
        return text
    try:
        return _CC.convert(text)
    except Exception:
        return text


# 紀要用逐字稿本身的語言 (2026-11-03 使用者指定：英文會議就出英文紀要，不翻譯)。
# 逐字稿裡中文字與英文字母的比例：英文一個字約 4–5 個字母，中文一個詞 1–2 個字；字母數超過中文字數 2 倍才算英文會議。
# 中文會議夾雜的英文術語 (ERP、test、bug) 遠低於這個比例。
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """回傳 "en" 或 "zh"。"""
    cjk = len(_CJK_RE.findall(text or ""))
    latin = len(_LATIN_RE.findall(text or ""))
    return "en" if latin >= 20 and latin > 2 * cjk else "zh"


# 程式內部一律用中文段落標題處理 (ensure_sections、待辦表對照逐字稿…都以「## 摘要／待辦／決議」為鍵)，
# 英文紀要在模型輸出後先換成內部標題，處理完再換回英文。
SECTION_MISSING_NOTE_EN = "(Not generated: the model ran out of output length. Please refer to the transcript.)"
_EN_HEADINGS = {"摘要": "Summary", "待辦": "Action Items", "決議": "Decisions"}
_EN_HEADING_RE = re.compile(r"^\s*#{1,4}\s*\**\s*(summary|key points|action items?|to-?dos?|tasks|decisions?)\s*\**\s*:?\s*$", re.I | re.M)
_EN_TODO_HEADER_RE = re.compile(r"^\|\s*(?:item|task|action(?: item)?)\s*\|\s*(?:owner|assignee|responsible)\s*\|\s*(?:due|deadline|time|date|when)\s*\|\s*$", re.I | re.M)


def to_internal_sections(md: str) -> str:
    """英文紀要 → 內部格式：段落標題、待辦表頭、「None」佔位換成中文。內容文字不動。"""
    if not md:
        return md or ""

    def head(m):
        k = m.group(1).lower()
        name = "摘要" if k in ("summary", "key points") else ("決議" if k.startswith("decision") else "待辦")
        return f"## {name}"
    md = _EN_HEADING_RE.sub(head, md)
    md = _EN_TODO_HEADER_RE.sub("| 事項 | 負責人 | 時間 |", md)
    md = re.sub(r"^\|\s*(?:none|n/a)\s*\|", "| 無 |", md, flags=re.I | re.M)
    return re.sub(r"^(\s*[-*•]\s*)(?:none|n/a)\.?\s*$", r"\1無", md, flags=re.I | re.M)


def localize_minutes(md: str, lang: str) -> str:
    """內部格式 → 顯示用：英文紀要把段落標題、待辦表頭、「無」與「未產生」說明換回英文；中文紀要原樣。"""
    if lang != "en" or not md:
        return md or ""
    for zh, en in _EN_HEADINGS.items():
        md = re.sub(rf"^## {zh}\s*$", f"## {en}", md, flags=re.M)
    md = md.replace("| 事項 | 負責人 | 時間 |", "| Item | Owner | Due |").replace(SECTION_MISSING_NOTE, SECTION_MISSING_NOTE_EN)
    md = re.sub(r"^\|\s*無\s*\|", "| None |", md, flags=re.M)
    return re.sub(r"^(\s*-\s*)無\s*$", r"\1None", md, flags=re.M)


# 紀要裡的日期也去掉空白：小模型常把逐字稿的「6月14日」寫成「6月14 日」或「6 月 14 日」(10-31 10 分鐘 e2e)
_SPACED_DATE_RE = re.compile(r"(?<!\d)(1[0-2]|[1-9])\s*月\s*(3[01]|[12]\d|[1-9])\s*日")


def tidy_dates(text: str) -> str:
    return _SPACED_DATE_RE.sub(r"\1月\2日", text or "")


_EMPTY_CELL = {"", "-", "—", "無", "N/A", "n/a"}


def _summary_table_to_bullets(lines: List[str]) -> List[str]:
    """「待辦」以外的段落寫成表格時改回條列。
    - 10-31 228 307 份紀要中 12 份把「## 摘要」畫成表：小模型把系統提示的「議題、關鍵數字、日期、金額、人名」當成欄位，
      常有整欄「-」和重複的日期。
    - 10 份「## 決議」有表格列或單獨一條分隔線 (待辦表寫到一半延伸過來)。
    表頭與分隔線丟掉，每列寫成「- 議題：值，值」，空白與重複的格子略過；段落裡已有其他條目時，模型自己寫的「- 無」也拿掉。"""
    out: List[str] = []
    section, converted = None, False
    seen = set()          # 已有的條目：表格轉出來的條目若重複就略過 (11-02 229 1 分鐘 e2e：兩列都只剩「產線加開檢查」)
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("## "):
            section, converted = s[3:].strip(), False
        if s.startswith("- "):
            seen.add(s)
        if section and section != "待辦" and s.startswith("|"):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if _is_sep(s) or _is_sep(nxt):          # 分隔線、表頭
                continue
            vals: List[str] = []
            for c in s.strip("|").split("|"):
                c = re.sub(r"^\d+[\.、)](?!\d)\s*", "", c.strip())     # 「1. 供應商」去編號，「3.28L」不動
                if c not in _EMPTY_CELL and c not in vals:
                    vals.append(c)
            if vals:
                bullet = f"- {vals[0]}：{'，'.join(vals[1:])}" if len(vals) > 1 else f"- {vals[0]}"
                if bullet not in seen:
                    seen.add(bullet)
                    out.append(bullet)
                converted = True
            continue
        if converted and s in ("- 無", "-無"):
            continue
        out.append(line)
    return out


def clean_summary_markdown(md: str, max_items: Optional[Dict[str, int]] = None) -> str:
    """LLM 紀要後處理：轉台灣繁體、移除重複行 (小模型偶發重複迴圈直到 num_predict 用盡)，並限制摘要/決議點數。"""
    limits = max_items or {"摘要": 5, "決議": 3}
    out, seen, section, count = [], set(), None, 0
    for line in tidy_dates(to_taiwan_traditional(md or "")).splitlines():
        s = line.strip()
        if s.startswith("|") and set(s) <= set("| "):   # 整列空白的表格列 (小模型偶爾連印 20 多列「| | | |」)
            continue
        if s.startswith("## "):
            section, count = s[3:].strip(), 0
            out.append(s)
            continue
        if not s:
            out.append("")
            continue
        is_item = s.startswith(("-", "*", "•")) or re.match(r"^\d+[\.、)]", s) is not None
        is_row = s.startswith("|") and not set(s) <= set("|-: ")
        if is_item and re.match(r"^[-*•]\s*no (?:additional|other|further|specific|more)\b", s, re.I):
            continue            # 「- No additional decisions made」：英文紀要的填充句
        if is_item and _CLOSING_FILLER_RE.search(s):     # 「…主持面試，今天就到這裡」：結尾客套話不是內容
            s = _CLOSING_FILLER_RE.sub("", s).rstrip("，,、 ")
            line = s
            if len(_BULLET_RE.sub("", s).strip()) < 2:
                continue
        if is_item or is_row:
            if s in seen:
                continue
            seen.add(s)
        if is_item:
            count += 1
            lim = limits.get(section or "")
            if lim and count > lim:
                continue
        out.append(line.rstrip())
    # 摘要表格在點數限制之後才改成條列：表格常有 6–13 列，先轉再限 5 點會丟掉後面議題的日期與負責人
    return re.sub(r"\n{3,}", "\n\n", "\n".join(_fix_table_separators(_join_split_tables(_summary_table_to_bullets(out))))).strip()


_CLOSING_FILLER_RE = re.compile(r"[，,。\s]*(?:好\s*[，,]?\s*)?(?:今天(?:就|先)?(?:到|開到)(?:這裡|這邊)|謝謝(?:大家|各位))[^，。\n]*")


def _join_split_tables(lines: List[str]) -> List[str]:
    """表格中間夾了空行或多一條分隔線 (11-02 3 分鐘 e2e：「| 劉客送預算表 |…」、空行、「|---|---|---|」、再接兩列)，
    後半段會變成沒有表頭的另一張表。把表格列之間的空行去掉，只保留表頭下面那一條分隔線。"""
    joined: List[str] = []
    for i, line in enumerate(lines):
        if not line.strip():
            prev = next((l.strip() for l in reversed(joined) if l.strip()), "")
            nxt = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
            if prev.startswith("|") and nxt.startswith("|"):
                continue
        joined.append(line)
    out: List[str] = []
    run = 0                      # 目前這張表已有幾列 (不含被丟掉的分隔線)
    for line in joined:
        s = line.strip()
        if s.startswith("|"):
            if _is_sep(s) and run != 1:
                continue
            run += 1
        else:
            run = 0
        out.append(line)
    return out



# 待辦表的負責人／時間必須出自逐字稿 (2026-10-29 228／229 提示詞 A/B：9 份逐字稿 × 2 次，現行提示詞的 94 個待辦欄位中
# 18 個是逐字稿與講稿都沒有的，例如「工程師A」「記錄員」「陳經理」「今日」)；改提示詞會連帶少寫 70% 的正確欄位，所以改在後處理擋。
_TODO_SKIP = {"", "-", "無", "待定", "未定", "未指定", "全體", "n/a", "N/A"}


def _ground_norm(text: str) -> str:
    return to_taiwan_traditional(text or "").replace(" ", "").replace("　", "").replace("臺", "台").lower()


_EN_BY_NAME_RE = re.compile(r"\bby\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b")
_EN_NOT_NAMES = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
                 "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
                 "next", "the", "end", "tomorrow", "today"}


def _fill_todo_cells(md: str) -> str:
    """模型常把負責人、期限寫在「事項」欄，另外兩欄留「-」(11-03 英文 e2e：「| Reply to Lotus Tech by Wednesday | John | - |」、
    「| Confirm firmware testing by Kevin | - | - |」)。空欄時從事項欄取出來，事項只留動作；之後仍經 ground_todo_table 對照逐字稿。"""
    out, section = [], None
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = s[3:].strip()
        if section == "待辦" and s.startswith("|") and not _is_sep(s):
            cols = [c.strip() for c in s.strip("|").split("|")]
            if len(cols) >= 3 and cols[0] not in ("事項", "無") and (cols[1] in _TODO_SKIP or cols[2] in _TODO_SKIP):
                item = cols[0]
                if not _CJK_RE.search(item) and _LATIN_RE.search(item):      # 短的英文事項不到 detect_language 的 20 個字母
                    if cols[2] in _TODO_SKIP and (mt := _EN_TIME_RE.search(item)):
                        cols[2], item = mt.group(1), (item[:mt.start()] + item[mt.end():]).strip(" ,;")
                    if cols[1] in _TODO_SKIP:
                        mo = next((m for m in _EN_BY_NAME_RE.finditer(item) if m.group(1).split()[0].lower() not in _EN_NOT_NAMES), None)
                        if mo:
                            cols[1], item = mo.group(1), (item[:mo.start()] + item[mo.end():]).strip(" ,;")
                else:
                    if cols[2] in _TODO_SKIP and (mt := _TIME_RE.search(item)):
                        cols[2] = mt.group(1).replace(" ", "")
                    if cols[1] in _TODO_SKIP and (mo := _OWNER_RE.search(item)):
                        cols[1] = mo.group(1)
                if len(item) >= 2:
                    cols[0] = item
                line = "| " + " | ".join(cols) + " |"
        out.append(line)
    return "\n".join(out)


def ground_todo_table(md: str, transcript: str) -> str:
    """待辦表每列的負責人、時間欄若在逐字稿中找不到，改成「-」 (多人時只留找得到的人)；事項欄與其他段落不動。
    空的負責人／時間欄先從事項欄補 (_fill_todo_cells)，補上的也一樣要在逐字稿裡找得到。"""
    if not md or not transcript or "## 待辦" not in md:
        return md
    md = _fill_todo_cells(md)
    tr = _ground_norm(transcript)
    out, section = [], None
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = s[3:].strip()
        if section == "待辦" and s.startswith("|") and not _is_sep(s):
            cols = [c.strip() for c in s.strip("|").split("|")]
            if len(cols) >= 3 and cols[0] != "事項":
                for i in (1, 2):
                    if cols[i] in _TODO_SKIP or _ground_norm(cols[i]) in tr:
                        continue
                    parts = [x for x in re.split(r"[、,，/]|與|和", cols[i]) if x.strip()]
                    kept = [x.strip() for x in parts if _ground_norm(x) in tr]
                    cols[i] = "、".join(kept) if kept and len(parts) > 1 else "-"
                line = "| " + " | ".join(cols) + " |"
        out.append(line)
    return "\n".join(out)

def _is_sep(line: str) -> bool:
    return line.strip().startswith("|") and set(line.strip()) <= set("|-: ")


def _fix_table_separators(lines: List[str]) -> List[str]:
    """表頭下一列不是分隔線時補上 (2026-10-29 228 e2e：小模型偶爾漏掉 |---|，待辦表就無法以表格顯示)。"""
    out: List[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        s = line.strip()
        starts_table = s.startswith("|") and not _is_sep(s) and (i == 0 or not lines[i - 1].strip().startswith("|"))
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if starts_table and nxt.startswith("|") and not _is_sep(nxt):
            out.append("|" + "---|" * max(1, s.count("|") - 1))
    return out


# 長逐字稿分段 (2026-10-27)：Hailo qwen3:1.7b 單次請求約 1500 字就 "Conversation context is full" (HTTP 500)。
# 即時逐字稿是以「，」串接的單一長段，沒有換行，舊版按 \n\n 切段 → 4328 / 5995 字都只切成 1 段。
# 改為先依句末標點、再依逗號切句，貪婪打包成 <= max_chars 的區塊；單句過長則硬切。
_SENT_END_RE = re.compile(r"(?<=[。！？!?；;\n])")
_CLAUSE_RE = re.compile(r"(?<=[，,、：:])")


def _pieces(text: str, max_chars: int) -> List[str]:
    out: List[str] = []
    for sent in _SENT_END_RE.split(text):
        if not sent.strip():
            continue
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        for clause in _CLAUSE_RE.split(sent):
            while len(clause) > max_chars:
                out.append(clause[:max_chars])
                clause = clause[max_chars:]
            if clause.strip():
                out.append(clause)
    return out


def split_transcript(text: str, max_chars: int = 1000) -> List[str]:
    """把逐字稿切成每段 <= max_chars 的區塊，盡量切在句子或子句邊界，內容不增不減 (只去頭尾空白)。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    # 平均分段：先算出需要幾段，再以平均長度為目標切，避免最後剩一小段 (多一次 LLM 呼叫)
    target = len(text) / -(-len(text) // max_chars)
    chunks: List[str] = []
    cur = ""
    for piece in _pieces(text, max_chars):
        if cur and (len(cur) + len(piece) > max_chars or len(cur) >= target):
            chunks.append(cur.strip())
            cur = ""
        cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    # 尾段過短 (< 15%) 時併入前一段，容許超出上限 10%：Hailo 實際上限約 1500 字，1000 已留足餘裕
    if len(chunks) >= 2 and len(chunks[-1]) < 0.15 * max_chars and len(chunks[-2]) + len(chunks[-1]) <= 1.1 * max_chars:
        chunks[-2:] = [chunks[-2] + chunks[-1]]
    return [c for c in chunks if c]


def tidy_live_transcript(text: str) -> str:
    """即時切片串接成的逐字稿 → 紀要用的全文：去掉亂碼字元、合併連續標點。錄音中預先整理與錄音結束後用同一套，前段才會逐字相同。"""
    t = (text or "").strip().replace("\ufffd", "")
    return re.sub(r"([，、。！？\s]){2,}", r"\1", t)


_STABLE_CUT_CHARS = "。！？!?；;\n，,、：:"


def take_stable_chunk(text: str, start: int, max_chars: int = 1000, lag: int = 1000) -> Optional[int]:
    """錄音中預先整理 (PI5_SUMMARY_PREMAP)：從 text[start:] 取一段 <= max_chars 的區塊，回傳結束位置。
    後面至少還要有 lag 字才取：最後一段可能是結尾總結 (split_with_recap 要另外切)，標點也可能還會被合併。
    盡量切在句子／子句邊界 (區塊後 40% 內最後一個標點之後)；找不到就硬切。"""
    if len(text) - start < max_chars + lag:
        return None
    window = text[start:start + max_chars]
    cut = max(window.rfind(c) for c in _STABLE_CUT_CHARS)
    end = start + (cut + 1 if cut >= int(max_chars * 0.6) else max_chars)
    return end


_RECAP_RE = re.compile(r"(最後總結|總結一下|做個總結|做一下總結|總結來說|最後來總結|以下總結"
                       r"|(?i:to wrap up|to sum up|to summari[sz]e|let me summari[sz]e|let'?s recap|to recap|in summary))")


def split_with_recap(text: str, max_chars: int = 1000, tail_ratio: float = 0.3, min_recap: int = 30) -> List[str]:
    """會議常以總結收尾，但總結只是最後一段的一小部分，小模型整理該段時常略過。
    在逐字稿最後 tail_ratio 範圍內找總結用語，找到就把總結切成獨立的最後一段 (長度需在 min_recap..max_chars)。"""
    text = (text or "").strip()
    start = int(len(text) * (1 - tail_ratio))
    m = _RECAP_RE.search(text, start)   # 尾段中第一個總結用語
    if m is not None:
        head, recap = text[:m.start()].strip(), text[m.start():].strip()
        if min_recap <= len(recap) <= max_chars and head:
            return split_transcript(head, max_chars) + [recap]
    return split_transcript(text, max_chars)


def is_recap_chunk(piece: str) -> bool:
    """split_with_recap() 切出的結尾總結段 (以總結用語開頭)。"""
    return bool(_RECAP_RE.match((piece or "").strip()))


# 結尾總結裡的客套話：從子句中「去掉」而不是整句丟掉 (STT 常把「…出稿今天就到這裡」黏成一句，整句丟會連待辦一起丟)
_RECAP_FILLER_SUB = re.compile(r"今天(?:就|先)*(?:到|開到)(?:這裡|這邊)?|謝謝(?:大家|各位)?|^(?:好|那|嗯|ok|OK)(?=\s|$)")
_RECAP_INTRO_RE = re.compile(r"(?:我們)?(?:最後)?(?:來)?(?:總結|整理|結論)(?:一下)?")


_EN_RECAP_FILLER_RE = re.compile(r"^(?:okay|ok|so|alright|right)?[,\s]*(?:to wrap up|to sum up|to recap|let'?s recap|in summary"
                                 r"|let me summari[sz]e|to summari[sz]e)?[,.:\s]*|(?:that'?s all for today|thank you,? everyone"
                                 r"|thanks,? everyone|thank you all|see you next week)[.!]?", re.I)


def _recap_notes_en(recap: str, max_lines: int = 6, max_len: int = 160) -> str:
    """英文的結尾總結：一句一行 (句子本身就是「Topic: we decided…, and John will…」)，去掉開場與道謝。"""
    out = []
    for sent in re.split(r"(?<=[.!?;])\s+", recap or ""):
        t = _EN_RECAP_FILLER_RE.sub("", sent.strip()).strip(" .")
        if len(t) >= 8:
            out.append("- " + t[:max_len])
    return "\n".join(out[:max_lines])


def recap_notes(recap: str, max_lines: int = 6, max_len: int = 50) -> str:
    """長會議的結尾總結改用逐字稿原句：依逗號／句號切成子句，去掉客套話；遇到「X方面」就換行 (一行一個議題)，
    同一議題內再依序打包成 <= max_len 字的條列。

    2026-10-30 228 離線 A/B (13 份 10／15／30 分鐘長逐字稿，`stage/exp_recap.py`、`recap_label_eval.py`)：口頭總結本身已是
    「X 方面，決議是…，待辦事項是某人在某日前…」，交給 qwen3:1.7b 再改寫、每行又截在 40 字時，負責人與期限常被丟掉；
    改用原句後回顧事實 (逐字稿內) 0.61 → 0.81、事實保留 0.176 → 0.230，且少一次 LLM 呼叫。
    依「方面」分行比單純依長度打包，議題覆蓋 0.815 → 0.831、事實保留 0.223 → 0.230。
    英文總結 (2026-11-03 起紀要跟著逐字稿語言) 改由 _recap_notes_en() 一句一行。
    """
    if detect_language(recap) == "en":
        return _recap_notes_en(recap, max_lines)
    clauses = [c.strip() for c in re.split(r"[，,。；;！!？?\n]", recap or "") if c.strip()]
    if clauses:   # 「最後總結一下任體方面」：去掉黏在第一個子句前面的總結用語
        clauses[0] = re.sub(r"^.*?(?:最後總結一下|總結一下|最後總結|做個總結|做一下總結)", "", clauses[0]).strip() or clauses[0]
    clauses = [_RECAP_FILLER_SUB.sub("", c).strip() for c in clauses]
    clauses = [c for c in clauses if len(c) >= 3 and not _RECAP_INTRO_RE.fullmatch(c)]
    groups: List[List[str]] = []
    for c in clauses:
        if "方面" in c or not groups:
            groups.append([c])
        else:
            groups[-1].append(c)
    lines: List[str] = []
    for g in groups:
        if len(g) == 1 and re.fullmatch(r".{0,6}方面", g[0]):
            continue        # 只剩「依應商方面」：後面的內容 STT 沒聽出來，單獨一行沒有資訊 (11-02 喇叭 30 分鐘)
        cur = ""
        for c in g:
            if cur and len(cur) + 1 + len(c) > max_len:
                lines.append(cur)
                cur = c
            else:
                cur = f"{cur}，{c}" if cur else c
        if cur:
            lines.append(cur)
    return "\n".join("- " + l[:max_len] for l in lines[:max_lines])


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.、)])\s*")
# 小模型常自行加上與議題無關的行 (「會議日期：7.25」「出席：…」)，會佔掉每段的條數名額
_META_RE = re.compile(r"^(?:會議)?(?:日期|時間|地點|出席|與會|主持|記錄|主題)\s*[：:]")
_CUT_AT = "，。；、,;）)"


def compact_notes(text: str, max_lines: int = 4, max_len: int = 40) -> str:
    """map 階段輸出的確定性收斂：小模型常不守行數/字數規則，且 num_predict 用盡時會留下半句。
    保留前 max_lines 條、每條 <= max_len 字 (盡量切在標點)，去掉亂碼殘句與重複，統一為「- 」條列。
    這讓 reduce 的輸入與 (照抄時的) 輸出長度都有上限，紀要不會在待辦表中途被截斷。"""
    out, seen = [], set()
    for line in (text or "").splitlines():
        t = _BULLET_RE.sub("", line).strip().replace("\ufffd", "")
        if t.startswith("#") or t.startswith("|") or len(t) < 6 or _META_RE.match(t):
            continue
        if len(t) > max_len:
            cut = max((t.rfind(c, 0, max_len) for c in _CUT_AT), default=-1)
            t = t[:cut] if cut >= max_len // 2 else t[:max_len]
            t = t.rstrip("，、,;； ")
        if t in seen:
            continue
        seen.add(t)
        out.append("- " + t)
        if len(out) >= max_lines:
            break
    return "\n".join(out)


def map_notes_complete(text: str, max_lines: int = 4, max_len: int = 40) -> bool:
    """串流 map 的提早停止條件：compact_notes() 會留下的 max_lines 條都已經寫完 (後面再生成也不會改變結果)。
    第 max_lines 條要嘛已換行，要嘛已超過 max_len 字 (之後會被截在前 max_len 字內)。
    2026-11-01 228 15 段 A/B：每段 40.1 → 32.2 s，保留事實 31.8 → 34.2 (非串流自己重跑一次是 32.3)。"""
    text = text or ""
    head = text[:text.rfind("\n") + 1]
    if len(compact_notes(head, max_lines, max_len).splitlines()) >= max_lines:
        return True
    tail = _BULLET_RE.sub("", text[len(head):]).strip()
    return len(tail) > max_len and len(compact_notes(text, max_lines, max_len).splitlines()) >= max_lines


SECTION_MISSING_NOTE = "（模型輸出長度不足，此段未產生，請參考逐字稿）"

# 逐字稿中明講的「決議是…」(STT 端 fix_meeting_terms 已把 決意/捷一… 改回「決議」)。
# 只取到下一個標點或「待辦」為止，30 字內沒有斷點的長串不取 (多半是 STT 連成一片的錯字)。
_SPOKEN_DECISION_RE = re.compile(r"決議(?:是|為|：|:)\s*([^，,。；;！!？?\n]{2,30}?)\s*(?=待辦|[，,。；;！!？?\n]|$)")


def fill_decisions_from_transcript(md: str, transcript: str, max_items: int = 3) -> str:
    """紀要的「決議」段是空的 (無／未產生) 但逐字稿明講了「決議是…」時，直接用逐字稿原句補上。

    2026-10-30 228：60 份實錄與 e2e 紀要有 24 份決議段是空的，其中 21 份逐字稿裡就有「決議是…」。
    qwen3:1.7b 常把決議寫進摘要或漏掉；原句出自逐字稿，不會捏造。決議段已有內容時不動。
    """
    if not md or not transcript:
        return md
    if re.search(r"^## 決議[ \t]*\Z", md, re.M):
        md += "\n"                     # 2026-10-31 e2e：模型輸出以「## 決議」結尾、後面沒有換行也沒有條目
    m = re.search(r"(^## 決議[ \t]*\n)(.*?)(?=^## |\Z)", md, re.S | re.M)
    if not m:
        return md
    items = [l.strip()[2:].strip() for l in m.group(2).splitlines() if l.strip().startswith("- ")]
    if any(x and x not in ("無", SECTION_MISSING_NOTE) for x in items):
        return md
    # 已經以同一句條列在其他段落的不再重複 (e2e 門檻 duplicate_lines = 0)
    existing = {re.sub(r"\s+", "", l.strip()[2:]) for l in md.splitlines() if l.strip().startswith("- ")}
    found: List[str] = []
    for d in _SPOKEN_DECISION_RE.findall(transcript):
        d = d.strip()
        if d and d not in found and re.sub(r"\s+", "", d) not in existing:
            found.append(d)
    if not found:
        return md
    body = "\n".join(f"- {d}" for d in found[:max_items])
    rest = md[m.end():]
    return md[:m.start(2)] + body + ("\n\n" + rest if rest else "")


_SECTION_ORDER = ("摘要", "待辦", "決議")


def _reorder_sections(lines: List[str]) -> List[str]:
    """三段順序固定為摘要、待辦、決議 (10-31 228 322 份紀要中 21 份把「## 決議」寫在「## 待辦」前面)。
    只在三段各出現一次時才調整，其他段落位置不動。"""
    blocks: List[List[str]] = [[]]
    for line in lines:
        if line.startswith("## "):
            blocks.append([])
        blocks[-1].append(line)
    titles = [b[0][3:].strip() if b and b[0].startswith("## ") else None for b in blocks]
    slots = [i for i, t in enumerate(titles) if t in _SECTION_ORDER]
    if sorted(titles[i] for i in slots) != sorted(_SECTION_ORDER):
        return lines
    by_title = {titles[i]: blocks[i] for i in slots}
    for i, t in zip(slots, _SECTION_ORDER):
        blk = by_title[t]
        blocks[i] = blk if blk[-1].strip() == "" or i == slots[-1] else blk + [""]
    return [l for b in blocks for l in b]


def _pad_todo_rows(lines: List[str], empty_row: str = "| 無 | - | - |") -> List[str]:
    """待辦表格列的格數少於表頭時補「-」(例：模型寫到「| 出差報告 |」就停，10-31 5 分鐘 e2e)。
    表格只有表頭沒有資料列時補一列「| 無 | - | - |」(11-01 5 分鐘 e2e：模型只印了 20 多列空白列，清掉後表格是空的)。"""
    out, section, width, rows, sep_at = [], None, 0, 0, None

    def close_table():
        if section == "待辦" and width and not rows and sep_at is not None:
            out.insert(sep_at + 1, empty_row)

    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            close_table()
            section, width, rows, sep_at = s[3:].strip(), 0, 0, None
        if section == "待辦" and s.startswith("|") and s.endswith("|"):
            if _is_sep(s):
                sep_at = len(out)
            else:
                cells = s.count("|") - 1
                if not width:
                    width = cells
                else:
                    rows += 1
                    if cells < width:
                        line = s + " - |" * (width - cells)
        out.append(line)
    close_table()
    return out


def _merge_duplicate_sections(lines: List[str]) -> List[str]:
    """同一個段落標題出現兩次時 (11-02 5 分鐘 e2e：「## 決議」寫了兩次、沒有「## 待辦」)，第二段的內容併到第一段後面。"""
    blocks: List[List[str]] = [[]]
    for line in lines:
        if line.strip().startswith("## "):
            blocks.append([])
        blocks[-1].append(line)
    out_blocks: List[List[str]] = []
    first: Dict[str, List[str]] = {}
    for b in blocks:
        title = b[0].strip()[3:].strip() if b and b[0].strip().startswith("## ") else None
        if title in _SECTION_ORDER and title in first:
            body = [l for l in b[1:] if l.strip()]
            dst = first[title]
            while dst and not dst[-1].strip():
                dst.pop()
            dst.extend(body + [""])
            continue
        if title in _SECTION_ORDER:
            first[title] = b
        out_blocks.append(b)
    return [l for b in out_blocks for l in b]


def _todo_table_from_bullets(lines: List[str]) -> List[str]:
    """模型沒寫「## 待辦」，但別段的條目裡有負責人與期限 (「- 待辦：劉課長主持面試，週一前回報」) 時，
    用這些條目組出待辦表，而不是標示「未產生」；以「待辦」開頭的條目移進表格後就不再留在原段。
    負責人與時間之後仍會由 ground_todo_table() 對照逐字稿。"""
    rows, keep = [], []
    for line in lines:
        t = line.strip()
        got = _todo_parts(t[2:].strip()) if t.startswith("- ") else None
        if got:
            row = f"| {got[0]} | {got[1]} | {got[2].replace(' ', '')} |"
            if row not in rows:
                rows.append(row)
            if _TODO_LEAD_RE.match(t[2:].strip()):
                continue
        keep.append(line)
    if not rows:
        return lines
    while keep and not keep[-1].strip():
        keep.pop()
    return keep + ["", "## 待辦", "| 事項 | 負責人 | 時間 |", "|---|---|---|"] + rows[:6]


_TODO_LEAD_RE = re.compile(r"^待辦(?:事項)?\s*(?:是|為|：|:)")


def ensure_sections(md: str) -> str:
    """紀要被 num_predict 截斷時，補上缺少的段落標題並明確標示未產生 (不捏造內容)；
    最後一列若是被截斷的半行表格也一併移除。三段順序調成摘要、待辦、決議，待辦表格太短的列補「-」。"""
    lines = (md or "").rstrip().splitlines()
    todo_cut = False      # 待辦表寫到半行就被截斷：表格若因此變空，標示「未產生」而不是「無」
    if lines and lines[-1].lstrip().startswith("|") and not lines[-1].rstrip().endswith("|"):
        lines.pop()
        heads = [l.strip()[3:].strip() for l in lines if l.strip().startswith("## ")]
        todo_cut = bool(heads) and heads[-1] == "待辦"
    if lines and lines[-1].strip() in ("##", "#"):
        lines.pop()
    lines = _merge_duplicate_sections(lines)
    if not any(l.strip() == "## 待辦" for l in lines):
        lines = _todo_table_from_bullets(lines)
    text = "\n".join(lines).rstrip()
    for sec in ("## 摘要", "## 待辦", "## 決議"):
        if sec not in text:
            text += f"\n\n{sec}\n- {SECTION_MISSING_NOTE}"
    # 補上的缺段也要排回原位 (10-31 3 分鐘 e2e：模型先寫決議、待辦被截斷，補上的「## 待辦」原本會落在最後)
    empty_row = f"| {SECTION_MISSING_NOTE} | - | - |" if todo_cut else "| 無 | - | - |"
    return "\n".join(_pad_todo_rows(_reorder_sections(text.strip().splitlines()), empty_row)).strip()


# 長會議紀要組裝 (2026-10-27)：qwen3:1.7b 無法跨 20+ 條重點做歸納 (實測會照抄、待辦表留空、決議擠掉)，
# 因此 map 之後不再交給 LLM reduce，而是以程式從各段重點組出三段式紀要：內容全部來自 LLM 的分段重點，不捏造。
# 姓名部分排除數量詞與動詞 (避免「增加三位工程師」「需楊工程師」被當成負責人)
# 「是」也排除：結尾總結「待辦事項是林主任…」的「項是林主任」不能整串當成負責人 (10-31 10 分鐘 e2e)
_NAME_CHAR = r"(?:(?![需請加由交給與和的位名個人是一二三四五六七八九十兩])[\u4e00-\u9fff])"
_OWNER_RE = re.compile(r"(?:負責人(?:是|為)?\s*)?(" + _NAME_CHAR + r"{1,3}(?:經理|副理|協理|主任|專員|工程師|公程師|處長|課長|組長|主管|老師))")
_TIME_RE = re.compile(r"(\d{1,2}\s*月\s*\d{1,2}\s*日|\d{1,2}[./-]\d{1,2}(?:日)?|\d{1,2}\s*日前|下?週[一二三四五六日]|月底|下週|明天|今天(?![就先到開])|年底)")
_DECISION_RE = re.compile(r"決定|決議|定案|確定|通過|同意|維持|改為|改成|暫停|取消"
                          r"|\b(?:decided|decision|agreed|approved|confirmed|keep the|go with|postponed?|cancel(?:l?ed)?|switch(?:ed)? to"
                          r"|sign a|signed an?|trial order|release (?:the )?fix|released)\b",
                          re.I)
# 英文紀要的待辦：「John will send the quote by Friday」「Owner: Mary, due Nov 5」
# 負責人：「John will reply…」，或第三人稱動詞「John replies／Kevin owns／Susan handles…」(map 階段的英文重點常這樣寫)。
# 「Lotus Tech wants…」「Mary says…」這類不是待辦的動詞排除。
_EN_STATE_VERBS = r"(?:is|was|has|does|wants|needs|asks|says|thinks|expects|hopes|requests|likes|gets|seems|looks|feels|agrees|knows)"
_EN_OWNER_RE = re.compile(r"\b(?!(?:And|But|So|Then|Also|Or|Okay|Ok)\b)([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+"
                          r"(?:(?:will|to|should|needs to|is to|must|is responsible for)\s+(.+)"
                          r"|((?!" + _EN_STATE_VERBS + r"\b)[a-z]+(?<!s)s(?<!ss)\s+.+))"
                          r"|(?i:owner|assigned to)\s*:?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)")
_EN_TIME_RE = re.compile(r"\b(?:by|before|on|due|until)\s+((?:next\s+)?(?:mon|tues|wednes|thurs|fri|satur|sun)day"
                         r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?"
                         r"|\d{1,2}/\d{1,2}|tomorrow|today|next week|(?:the )?end of (?:the )?(?:week|month))\b", re.I)


def _todo_parts_en(line: str):
    """英文條目 → (事項, 負責人, 時間) 或 None：要同時有人名 (大寫開頭) 與期限。"""
    mt = _EN_TIME_RE.search(line)
    mo = _EN_OWNER_RE.search(line)
    if not (mo and mt):
        return None
    owner = mo.group(1) or mo.group(4)
    if owner.split()[0].lower() in ("we", "they", "i", "you", "it", "this", "that", "the", "all", "everyone", "team") \
            or owner.split()[0].lower() in _EN_NOT_NAMES:
        return None
    action = (mo.group(2) or mo.group(3) or "").replace(mt.group(0), "")
    action = re.split(r"[,;.]", action)[0].strip()
    if len(action) < 3:      # 「Owner: Mary Chen, due Nov 5, confirm sample capacity」：取沒有負責人與期限的那個子句
        rest = [c.strip() for c in re.split(r"[:,;.]", line)
                if c.strip() and owner not in c and mt.group(0) not in c and c.strip().lower() not in ("owner", "assigned to")]
        action = rest[-1] if rest else re.split(r"[:,;.]", line)[0].strip()
    return action[:40], owner, mt.group(1)


def _note_lines(notes: str) -> List[str]:
    return [l[2:].strip() for l in (notes or "").splitlines() if l.startswith("- ") and l[2:].strip()]


# 「待辦事項十個部門週四前…」：STT 常漏掉「是」，有「事項」兩字時「是」可省略 (11-02 30 分鐘 e2e)
_TODO_MARK_RE = re.compile(r"待辦事項\s*(?:是|為|：|:)?\s*|待辦\s*(?:是|為|：|:)\s*")
_CLAUSE_SPLIT_RE = re.compile(r"[，；;。]|,(?!\d)")          # 不切「1,200」這類數字
_ACTION_LEAD_RE = re.compile(r"^(?:在|於|要|會|將|請|負責|之前|以前|前)+")


def _todo_parts(line: str):
    """回傳 (事項, 負責人, 時間) 或 None。
    有「待辦事項是…」時只看它後面那一句：負責人從這句取，時間先找這句、沒有才用整行的 (決議的期限)，
    事項寫成去掉負責人與時間後剩下的動作。
    例：「訂單方面，決議是先簽50臺，待辦事項是林主任週三前回覆華新」→ (回覆華新, 林主任, 週三)。
    原本會取第一個子句「訂單方面」當事項，時間也可能抓到決議裡的日期。"""
    parts = _TODO_MARK_RE.split(line)
    if len(parts) > 1:
        seg = _CLAUSE_SPLIT_RE.split(parts[-1])[0].strip()
        mo = _OWNER_RE.search(seg)
        if mo:
            mt = _TIME_RE.search(seg)
            if mt:
                action = seg.replace(mo.group(0), "", 1).replace(mt.group(0), "", 1)
            else:
                mt = _TIME_RE.search(line)
                action = seg.replace(mo.group(0), "", 1)
            if mt:
                action = _ACTION_LEAD_RE.sub("", action.strip()).strip()
                topic = _CLAUSE_SPLIT_RE.split(parts[0])[0].strip()
                topic = re.sub(r"\s*待辦(?:事項)?\s*$", "", topic).strip()
                item = action if len(action) >= 2 else (topic or _CLAUSE_SPLIT_RE.split(line)[0].strip() or line)
                return item[:20], mo.group(1), mt.group(1)
    mo, mt = _OWNER_RE.search(line), _TIME_RE.search(line)
    if not (mo and mt):
        return _todo_parts_en(line) if detect_language(line) == "en" else None
    first = _CLAUSE_SPLIT_RE.split(line)[0].strip()
    if mo.end() <= len(first):
        # 負責人就在第一個子句裡 (STT 把「待辦事項是」聽成別的字時，整段總結常沒有標點)：事項取負責人後面的動作
        rest = first[mo.end():].replace(mt.group(0), "", 1)
        rest = _ACTION_LEAD_RE.sub("", rest.strip()).strip()
        if len(rest) >= 4:
            return rest[:20], mo.group(1), mt.group(1)
    return (first[:20] or line[:20]), mo.group(1), mt.group(1)


def extract_todos(lines: List[str], max_rows: int = 6) -> List[str]:
    """同時出現負責人與期限的重點 → 待辦表格列「| 事項 | 負責人 | 時間 |」。"""
    rows, seen = [], set()
    for line in lines:
        got = _todo_parts(line)
        if not got:
            continue
        item, owner, when = got
        if item in seen:
            continue
        seen.add(item)
        rows.append(f"| {item} | {owner} | {when.replace(' ', '') if _CJK_RE.search(when) else when} |")   # 英文日期保留空白
        if len(rows) >= max_rows:
            break
    return rows


def _drop_listed_todo_clause(decisions: List[str], todo_rows: List[str]) -> List[str]:
    """結尾總結的決議條目常是「X方面，決議是…，待辦事項是某人某日前…」。待辦那半句已經寫進待辦表
    (認得出負責人與時間) 時，決議只留前半句，不在兩段重複；待辦表沒收到的照舊整句保留，負責人不會不見。"""
    listed = {r.split("|")[1].strip() for r in todo_rows if r.count("|") >= 4}
    out = []
    for line in decisions:
        parts = _TODO_MARK_RE.split(line, 1)
        got = _todo_parts(line) if len(parts) == 2 else None
        if not got or got[0] not in listed:
            out.append(line)
            continue
        # 只拿掉待辦那一個子句；後面若還接著別的議題 (STT 漏了「方面」時常黏成一行) 照樣保留
        seg = _CLAUSE_SPLIT_RE.split(parts[1], 1)[0]
        head = re.sub(r"[，,；;、\s]*待辦(?:事項)?\s*$", "", parts[0]).rstrip("，,；;、 ")
        rest = parts[1][len(seg):].lstrip("，,；;。、 ")
        trimmed = f"{head}，{rest}" if head and rest else (head or rest)
        out.append(trimmed if len(head) >= 4 else line)
    return out


def _topic_key(line: str) -> str:
    head = re.split(r"[：:]", line, 1)
    return head[0].strip() if len(head) == 2 and 1 <= len(head[0].strip()) <= 8 else line[:4]


def assemble_long_minutes(chunk_notes: List[str], max_summary: int = 8, max_decisions: int = 3, tail_lines: int = 4,
                          max_todos: int = 6) -> str:
    """chunk_notes：每段的 compact_notes() 輸出 (依時間順序)。
    摘要：最後一段 (會議結尾常有總結) 先保留 tail_lines 條，其餘名額每段先取第 1 條、有剩再取第 2 條…，維持時間順序；
    待辦/決議由規則從全部重點擷取。"""
    per_all = [_note_lines(tidy_dates(to_taiwan_traditional(n))) for n in chunk_notes]
    all_lines = [l for p in per_all for l in p]
    decisions = [l for l in all_lines if _DECISION_RE.search(l)][:max_decisions]
    # 選進「決議」的條目不再放進「摘要」(2026-10-29 228 30 分鐘 e2e：同一條同時出現在兩段，duplicate_lines 不過門檻)；
    # 名額留給其他議題。全部條目都是決議時才退回原本的選法。
    per = [[l for l in p if l not in decisions] for p in per_all]
    if not any(per):
        per = per_all
    picked = []
    if len(per) > 1 and per[-1]:
        picked = [(len(per) - 1, r, l) for r, l in enumerate(per[-1][:tail_lines])]
    body = per[:-1] if picked else per
    # 第一輪只收「議題還沒出現過」的條目 (以「議題名稱：」或前 4 字為鍵)，避免摘要被同一類議題洗版；名額有剩再補
    used = {_topic_key(l) for _, _, l in picked}
    for diverse in (True, False):
        rank = 0
        while len(picked) < max_summary and any(len(p) > rank for p in body):
            for seg, lines in enumerate(body):
                if len(lines) > rank and len(picked) < max_summary:
                    item = (seg, rank, lines[rank])
                    if item in picked or (diverse and _topic_key(lines[rank]) in used):
                        continue
                    picked.append(item)
                    used.add(_topic_key(lines[rank]))
            rank += 1
    picked.sort()
    todos = extract_todos(all_lines, max_rows=max_todos)
    decisions = _drop_listed_todo_clause(decisions, todos)
    out = ["## 摘要"] + [f"- {l}" for _, _, l in picked] if picked else ["## 摘要", "- 無"]
    out += ["", "## 待辦", "| 事項 | 負責人 | 時間 |", "|---|---|---|"] + (todos or ["| 無 | - | - |"])
    out += ["", "## 決議"] + ([f"- {l}" for l in decisions] or ["- 無"])
    return "\n".join(out)



def _minutes_section(md: str, title: str):
    return re.search(rf"(^## {title}[ \t]*\n)(.*?)(?=^## |\Z)", md, re.S | re.M)


_DATE_TOKEN_RE = re.compile(r"\d{1,2}月\d{1,2}日")


_EN_STOP = {"the", "and", "for", "with", "after", "at", "on", "of", "to", "by", "a", "an", "in", "we", "is", "was", "be", "new"}


def _en_words(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _EN_STOP:
            continue
        for suf in ("ed", "es", "s", "th", "st", "nd", "rd", "e"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def _already_said(core: str, md_norm: str, decisions: List[str]) -> bool:
    """結尾總結的決議是否已經寫在紀要裡：原句出現過、同一個日期已在決議段，或與某條決議的用字重疊七成以上。"""
    norm = lambda x: re.sub(r"[\s，,。、：:]", "", x)
    c = norm(core)
    if not c or c in md_norm:
        return True
    if not _CJK_RE.search(core):         # 英文：字母集合沒有意義，改看單字重疊 (粗略詞幹：released／release、15th／15)
        cw = _en_words(core)
        return bool(cw) and any(len(cw & _en_words(x)) >= 0.6 * len(cw) for x in decisions)
    dates = _DATE_TOKEN_RE.findall(c)
    if dates and any(d in norm(x) for d in dates for x in decisions):
        return True
    return any(len(set(c) & set(norm(x))) >= 0.7 * len(set(c)) for x in decisions)


def merge_recap_into_minutes(md: str, transcript: str, max_rows: int = 6, max_decisions: int = 5) -> str:
    """單次紀要 (短會議) 補上結尾口頭總結：qwen3:1.7b 常只寫前面幾個議題，把最後「X方面，決議是…，待辦事項是…」漏掉。
    與長會議相同，總結用逐字稿原句 (recap_notes)；待辦表補上負責人還沒出現的列，決議段補上還沒寫到的決議。
    內容全部出自逐字稿，負責人與時間仍會再經 ground_todo_table() 對照。

    2026-11-02 離線重放 59 份 1–5 分鐘 e2e 紀要：回顧事實 (逐字稿內) 0.825 → 0.90x，10 份變好、0 份變差，沒有重複條目。
    """
    if not md or not transcript:
        return md
    parts = split_with_recap(transcript)
    if len(parts) < 2 or not is_recap_chunk(parts[-1]):
        return md
    lines = _note_lines(recap_notes(parts[-1]))
    if not lines:
        return md
    m = _minutes_section(md, "待辦")
    rows = extract_todos(lines, max_rows=max_rows)
    if m and rows:
        body = m.group(2).rstrip("\n").splitlines()
        table = [l for l in body if l.strip().startswith("|") and not _is_sep(l.strip())]
        data = table[1:] if table else []
        placeholder = [l for l in data if l.count("|") >= 2 and l.split("|")[1].strip() in ("無", SECTION_MISSING_NOTE)]
        table_text = "\n".join(data)       # 負責人可能被寫在「事項」欄 (「| 王經理主持面試 | - | - |」)
        real = [l for l in data if l not in placeholder]
        add = [r for r in rows if r.split("|")[2].strip() not in table_text][:max(0, max_rows - len(real))]
        if add and table:
            new_body = [l for l in body if l not in placeholder] + add
            rest = md[m.end():]
            md = md[:m.start(2)] + "\n".join(new_body) + ("\n\n" + rest.lstrip("\n") if rest.strip() else "\n")
    m = _minutes_section(md, "決議")
    if m:
        items = [l.strip() for l in m.group(2).splitlines() if l.strip().startswith("- ")]
        real = [l for l in items if l[2:].strip() not in ("無", SECTION_MISSING_NOTE)]
        tm = _minutes_section(md, "待辦")
        todo_rows = [l.strip() for l in (tm.group(2).splitlines() if tm else []) if l.strip().startswith("|")]
        md_norm = re.sub(r"[\s，,。、：:]", "", md)
        add = []
        for line in lines:
            if "決議" in line:
                d = _drop_listed_todo_clause([line], todo_rows)[0]
                core = _CLAUSE_SPLIT_RE.split(re.split(r"決議\s*(?:是|為|：|:)", d, 1)[-1])[0].strip()
            elif not _CJK_RE.search(line) and (md_ := _DECISION_RE.search(line)):     # 英文總結：「we agreed on …」
                d = line
                core = re.split(r"[,;]", line[md_.start():])[0].strip()
            else:
                continue
            if len(core) >= 4 and not _already_said(core, md_norm, [x[2:] for x in real] + add):
                add.append(d)
        add = add[:max(0, max_decisions - len(real))]
        if add:
            rest = md[m.end():]
            body = "\n".join(real + [f"- {x}" for x in add])
            md = md[:m.start(2)] + body + ("\n\n" + rest.lstrip("\n") if rest.strip() else "\n")
    return md.rstrip() + "\n" if md.endswith("\n") else md.rstrip()


def keep_alive_plan(n_chunks: int, sec_per_chunk: int = 75, cap: int = 1800, margin: int = 120) -> List[Optional[int]]:
    """分段整理時每一段請求要帶的 hailo-ollama keep_alive (秒)；None = 用預設 0 (請求完成即卸載)。

    hailo-ollama 0.5.1 的 keep_alive 從模型載入開始計時、後續請求不會延長；到期若撞上下一段請求會 HTTP 500。
    因此把各段分成「載入視窗」：每個視窗最多 (cap - margin) // sec_per_chunk 段，視窗最後一段帶 0 主動卸載，
    下一段重新載入並重新計時。4 小時會議約 55 段 → 3 個視窗、多 2 次載入 (每次約 8–15 s)。"""
    window = max(1, (cap - margin) // sec_per_chunk)
    plan: List[Optional[int]] = []
    for i in range(n_chunks):
        pos = i % window
        size = min(window, n_chunks - (i - pos))
        remaining = size - pos
        plan.append(None if remaining == 1 else margin + sec_per_chunk * remaining)
    return plan


def summary_limits(n_chunks: int) -> Dict[str, int]:
    """長會議紀要的條目上限隨長度增加 (30 分鐘約 7 段 → 摘要 8／待辦 6／決議 3；4 小時約 55 段 → 15／11／6)。"""
    return {"max_summary": min(16, 8 + max(0, n_chunks - 8) // 6),
            "max_todos": min(12, 6 + n_chunks // 10),
            "max_decisions": min(6, 3 + n_chunks // 15)}
