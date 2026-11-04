# code_dispatcher/summary_utils.py
"""會議紀要 (LLM 輸出) 後處理，純函式，供 meeting_minutes 使用並可離線單元測試。"""
import difflib
import re
import unicodedata
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


# 英文句子裡的功能詞 (the、is、we、to…) 約佔單字的 38–40% (11-03 英文 e2e 逐字稿 0.398／0.384)；
# 夾在中文裡的英文 (「這個 release 的 schedule 要 delay，QA 的 regression test 還沒 pass」) 幾乎都是名詞、動詞。
# 英文字母數只按「看起來是英文句子」的比例計算，台灣工程會議常見的中英夾雜才不會被當成英文會議。
_EN_FUNCTION_WORDS = frozenset("the a an is are was were be been we you they he she it i to of and in on for with that this "
                               "these those will would can could should have has had our your their by at from so but not do "
                               "does did there as or if all its let's lets what which who".split())


def english_share(text: str) -> float:
    """英文字母裡屬於英文句子的比例 (0–1)：功能詞比例 ÷ 0.35，上限 1。"""
    words = re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text or "")
    if not words:
        return 0.0
    return min(1.0, sum(1 for w in words if w.lower() in _EN_FUNCTION_WORDS) / (0.35 * len(words)))


def detect_language(text: str) -> str:
    """回傳 "en" 或 "zh"。英文字母 (按英文句子比例折算) 超過中文字 2 倍才算英文 (約等於英文講話時間超過四成五)。"""
    cjk = len(_CJK_RE.findall(text or ""))
    latin = len(_LATIN_RE.findall(text or ""))
    if latin < 20:
        return "zh"
    if cjk == 0:
        return "en"
    return "en" if latin * english_share(text) > 2 * cjk else "zh"


_TRANSLATE_HEAD_RE = re.compile(r"^\s*【[^】\n]{1,20}】\s*$")


def split_translation_sections(text: str) -> List[tuple]:
    """網頁送來翻譯的紀要「【摘要】\n• …\n\n【待辦】\n• …」→ [(標題或 None, 內容), …]。
    小模型一次翻整份時常把段落標題漏掉或不翻 (11-03 測試)，所以分段翻、標題由網頁直接給目標語言。"""
    blocks: List[list] = []
    for line in (text or "").splitlines():
        if _TRANSLATE_HEAD_RE.match(line):
            blocks.append([line.strip(), []])
        else:
            if not blocks:
                blocks.append([None, []])
            blocks[-1][1].append(line)
    return [(h, "\n".join(b).strip()) for h, b in blocks if h or "\n".join(b).strip()]


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


# 系統提示要求「寫出該議題的關鍵數字、日期、金額與人名」，小模型偶爾把這幾個字當欄位名附在條目後面：
# 「…交期延至10月10日，關鍵數字：10月10日，金額：無，人名：曦代料」(11-04 3 分鐘 e2e；最近 26 份紀要 1 份、舊的 104 份 3 份)。
# 值是「無」、前面已經寫過、或「人名」後面不是「姓＋職稱」時整段拿掉；「負責人：X」是正常寫法，不動。
_LABEL_ECHO_RE = re.compile(r"\s*[，,；;]\s*(關鍵數字|數字|金額|人名|日期|預計時間|時間|公司名?)\s*[：:]\s*([^，,；;。]*)")
_TODO_HEADER_ECHO = {"事項", "待辦", "待辦事項", "Item", "item"}
_DECISION_LABEL_RE = re.compile(r"^[-*•]\s*(?:決議|決定|結論)\s*[：:]\s*(?=\S)")
_EMPTY_VALUES = {"", "-", "無", "不明", "未提及", "未知", "無提及", "待定", "未定", "n/a", "N/A"}


def _drop_label_echo(item: str) -> str:
    def rep(m):
        val = m.group(2).strip()
        said = re.sub(r"\s", "", item[:m.start()])
        if val in _EMPTY_VALUES or re.sub(r"\s", "", val) in said or (m.group(1) == "人名" and not _OWNER_RE.fullmatch(val)):
            return ""
        return m.group(0)
    return _LABEL_ECHO_RE.sub(rep, item)


def clean_summary_markdown(md: str, max_items: Optional[Dict[str, int]] = None) -> str:
    """LLM 紀要後處理：轉台灣繁體、移除重複行 (小模型偶發重複迴圈直到 num_predict 用盡)，並限制摘要/決議點數。"""
    limits = max_items or {"摘要": 5, "決議": 3}
    out, seen, section, count = [], set(), None, 0
    seen_header: Dict[str, bool] = {}
    for line in tidy_dates(to_taiwan_traditional(md or "")).replace("\ufffd", "").splitlines():   # 模型偶爾吐出半個字元 (「�品保部」)
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
            if len(re.sub(r"[\W_]", "", _BULLET_RE.sub("", s))) < 2:     # 只剩標點 (「-，謝謝」拿掉後的「-，」)
                continue
        if is_row and section == "待辦" and not s.strip("|").split("|")[0].strip():
            continue        # 事項欄空白的列 (「|  | 許經理 | 11月28日 |」)：先拿掉，結尾總結／正文交辦會補上完整的列
        if is_row and section == "待辦" and s.strip("|").split("|")[0].strip() in _TODO_HEADER_ECHO and seen_header.get("待辦"):
            # 模型把表頭「事項」當內容 (11-04 夜間 3 分鐘實錄：「| 事項 | 李協理 | 週五回報 |」)：時間欄裡的動作拿來當事項
            cells = [c.strip() for c in s.strip("|").split("|")]
            mt = _TIME_RE.search(cells[2]) if len(cells) >= 3 else None
            rest = (cells[2][:mt.start()] + cells[2][mt.end():]).strip(" 前") if mt else ""
            if not mt or len(rest) < 2:
                continue
            when = mt.group(1) + ("前" if cells[2][mt.end():].startswith("前") else "")
            s = line = f"| {rest.lstrip('前')} | {cells[1]} | {when} |"
        if is_row and section == "待辦":
            seen_header["待辦"] = True
        if is_row and _CLOSING_FILLER_RE.search(s):      # 「| 王經理主持面試，今天就到這裡 | 王經理 | - |」(541cd0c 3 分鐘 e2e)
            cells = s.strip().strip("|").split("|")
            cells[0] = " " + (_CLOSING_FILLER_RE.sub("", cells[0]).strip(" ，,、") or cells[0].strip()) + " "
            s = line = "|" + "|".join(cells) + "|"
        if is_item and section == "決議" and _DECISION_LABEL_RE.match(s):
            s = line = _DECISION_LABEL_RE.sub("- ", s)       # 「- 決議：8月26日前完成報到」：段名已是決議 (138 份紀要 11 份)
        if is_item and section in ("摘要", "決議") and _LABEL_ECHO_RE.search(s):
            s = line = _drop_label_echo(s)
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
    lines = _drop_placeholder_bullets(_strip_table_cells_from_bullets(
        _fix_table_separators(_join_split_tables(_summary_table_to_bullets(out)))))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _strip_table_cells_from_bullets(lines: List[str]) -> List[str]:
    """待辦表的列被寫進摘要／決議的條目 (11-03 使用者測試：決議「- 確認是否需要重新整理紀錄 | - | - |」)：
    條目只留第一格，其餘有內容的格子放進括號 (「加開一站檢查（黃專員，2月15日）」)；和待辦表事項完全相同的決議拿掉
    (待辦表已有)。段落因此變空時補「- 無」。"""
    todo_items = {c.strip() for l in lines if l.strip().startswith("|") and not _is_sep(l)
                  for c in [l.strip().strip("|").split("|")[0]]}
    out: List[str] = []
    section, kept = None, None
    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            if kept == 0:
                out.append("- 無")
            section, kept = s[3:].strip(), None
        if section in ("摘要", "決議") and s.startswith("- ") and "|" in s:
            cells = [c.strip() for c in s[2:].split("|")]
            first, extra = cells[0], [c for c in cells[1:] if c and c not in _TODO_SKIP]
            kept = kept or 0
            if len(first) < 2 or (section == "決議" and first in todo_items):
                continue
            line = f"- {first}（{'，'.join(extra)}）" if extra else f"- {first}"
        if section in ("摘要", "決議") and s.startswith("- "):
            kept = (kept or 0) + 1
        out.append(line)
    if kept == 0:
        out.append("- 無")
    return out


def _drop_placeholder_bullets(lines: List[str]) -> List[str]:
    """段落已有內容時，模型多寫的「- 無」(英文紀要是「- None」，已換成內部的「- 無」) 拿掉 (11-03 英文 60 秒以上 e2e 的決議段)。"""
    blocks: List[List[str]] = [[]]
    for line in lines:
        if line.strip().startswith("## "):
            blocks.append([])
        blocks[-1].append(line)
    out: List[str] = []
    for b in blocks:
        bullets = [l for l in b if l.strip().startswith("- ")]
        if any(l.strip() not in ("- 無", "-無") for l in bullets):
            b = [l for l in b if l.strip() not in ("- 無", "-無")]
        out.extend(b)
    return out


# 11-04 ad981d9 5 分鐘 e2e 的決議出現「- 今日就到這裡」：模型把「今天」寫成「今日」；也有「- 今日會議結束，謝謝」
_CLOSING_FILLER_RE = re.compile(r"[，,。\s]*(?:好\s*[，,]?\s*)?(?:今[天日](?:就|先|做)?(?:到|開到)(?:這裡|這邊)|今[天日](?:的)?會議(?:到此)?結束"
                                r"|謝謝(?:大家|各位)|謝謝(?=[\s!！。.]*$))[^，。\n]*")


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


# 模型常自己替日期補上年份 (11-03 英文 e2e：「September 2, 2023」「Feb 28, 2024」，逐字稿只說 September 2)。
# 日期後面的四位數年份若逐字稿裡沒有出現，就拿掉；中文「2023年9月2日」同理。
_EN_DATE_YEAR_RE = re.compile(r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?),?\s+((?:19|20)\d{2})\b", re.I)
_ZH_DATE_YEAR_RE = re.compile(r"((?:19|20)\d{2})\s*年\s*(?=\d{1,2}\s*月)")


def drop_unspoken_years(md: str, transcript: str) -> str:
    if not md:
        return md or ""
    tr = transcript or ""
    md = _EN_DATE_YEAR_RE.sub(lambda m: m.group(0) if m.group(2) in tr else m.group(1), md)
    return _ZH_DATE_YEAR_RE.sub(lambda m: m.group(0) if m.group(1) in tr else "", md)


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


_EN_PRONOUNS = ("we", "they", "i", "you", "it", "this", "that", "the", "all", "everyone", "team")


def _fill_en_owner_from_transcript(md: str, transcript: str) -> str:
    """英文待辦表負責人仍空著、事項欄只提到一個人名，而逐字稿裡這個人正是負責人 (「Kevin owns the testing」「Susan will handle it」)
    時補上。11-03 en_long：「| Confirm firmware testing with Kevin | - | March 15 |」。收件人 (「send your names to Mary」) 不算負責人。"""
    owners = set()
    for m in _EN_OWNER_RE.finditer(transcript or ""):
        o = m.group(1) or m.group(4)
        if o and o.split()[0].lower() not in _EN_PRONOUNS and o.split()[0].lower() not in _EN_NOT_NAMES:
            owners.add(o.split()[0])
    if not owners:
        return md
    out, section = [], None
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = s[3:].strip()
        if section == "待辦" and s.startswith("|") and not _is_sep(s):
            cols = [c.strip() for c in s.strip("|").split("|")]
            if len(cols) >= 3 and cols[1] in _TODO_SKIP and not _CJK_RE.search(cols[0]):
                names = {w for w in re.findall(r"\b[A-Z][a-z]+\b", cols[0]) if w in owners}
                if len(names) == 1:
                    cols[1] = names.pop()
                    line = "| " + " | ".join(cols) + " |"
        out.append(line)
    return "\n".join(out)


_EN_WEEKDAY_ABBR = {"mon": "Monday", "tue": "Tuesday", "tues": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
                    "thur": "Thursday", "thurs": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}


def _edit1(a: str, b: str) -> bool:
    """a、b 相差至多一個字元 (替換、插入或刪除)。"""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            diff += 1
            if diff > 1:
                return False
            if len(a) == len(b):
                i += 1
            j += 1
        else:
            i += 1
            j += 1
    return diff + (len(b) - j) <= 1


def _snap_en_todo_cells(md: str, transcript: str) -> str:
    """英文待辦表：模型把人名拼錯 (11-04 en e2e「Kevn」「Kedin」，逐字稿是 Kevin) 時換成逐字稿裡只差一個字母、
    開頭相同的那個名字；期限寫成縮寫 (「Wed」) 而逐字稿有全名時寫全名。"""
    names = set(re.findall(r"\b[A-Z][a-z]{2,}\b", transcript or ""))
    low = (transcript or "").lower()
    out, section = [], None
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = s[3:].strip()
        if section == "待辦" and s.startswith("|") and not _is_sep(s):
            cols = [c.strip() for c in s.strip("|").split("|")]
            if len(cols) >= 3 and not _CJK_RE.search(cols[0] + cols[1] + cols[2]):
                words = cols[1].split()
                w = unicodedata.normalize("NFKD", words[0]).encode("ascii", "ignore").decode() if words else ""   # Kévin → Kevin
                if re.fullmatch(r"[A-Z][a-z]{2,}", w):
                    near = {n for n in names if n != w and n[0] == w[0] and _edit1(n, w)}
                    if w in names:      # STT 自己也拼錯過一次 (「Kevn will arrange…」「Kevin owns the testing」)：取出現較多、較長的那個
                        cnt = lambda x: len(re.findall(rf"\b{x}\b", transcript))
                        near = {n for n in near if len(n) > len(w) and cnt(n) >= cnt(w)}
                    if len(near) == 1:
                        cols[1] = " ".join([near.pop()] + words[1:])
                    elif w != words[0] and w in names:
                        cols[1] = " ".join([w] + words[1:])
                full = _EN_WEEKDAY_ABBR.get(cols[2].lower().rstrip("."))
                if full and full.lower() in low:
                    cols[2] = full
                line = "| " + " | ".join(cols) + " |"
        out.append(line)
    return "\n".join(out)


def _fill_en_due_from_transcript(md: str, transcript: str) -> str:
    """英文待辦表有負責人、期限卻空著，而逐字稿裡「這個人 + 這件事的動詞 + by <時間>」只有一句時補上期限
    (11-04 英文 e2e：「| Reply to Lotus Tech | John | - |」，逐字稿「John will reply to the customer by Wednesday」；
    25 份英文輸出 184 列中 49 列期限空著)。動詞取事項的第一個字，對不上就不補 (「Sign trial order」不拿 reply 的期限)。"""
    sents = [x for x in re.split(r"(?<=[.!?])\s+", transcript or "") if _EN_TIME_RE.search(x)]
    if not sents:
        return md
    out, section = [], None
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = s[3:].strip()
        if section == "待辦" and s.startswith("|") and not _is_sep(s):
            cols = [c.strip() for c in s.strip("|").split("|")]
            if (len(cols) >= 3 and cols[2] in _TODO_SKIP and cols[1] not in _TODO_SKIP
                    and not _CJK_RE.search(cols[0] + cols[1]) and cols[0].split()):
                owner, verb = cols[1].split()[0], cols[0].split()[0].lower()
                verb = re.sub(r"(?:e?s|ed|ing)$", "", verb) if len(verb) > 4 else verb
                hits = {_EN_TIME_RE.search(x).group(1) for x in sents
                        if re.search(rf"\b{re.escape(owner)}\b", x) and re.search(rf"\b{re.escape(verb)}", x, re.I)}
                if len(hits) == 1:
                    cols[2] = hits.pop()
                    line = "| " + " | ".join(cols) + " |"
        out.append(line)
    return "\n".join(out)


def ground_todo_table(md: str, transcript: str) -> str:
    """待辦表每列的負責人、時間欄若在逐字稿中找不到，改成「-」 (多人時只留找得到的人)；事項欄與其他段落不動。
    空的負責人／時間欄先從事項欄補 (_fill_todo_cells)，補上的也一樣要在逐字稿裡找得到。"""
    if not md or not transcript or "## 待辦" not in md:
        return md
    md = _fill_en_due_from_transcript(_fill_en_owner_from_transcript(_snap_en_todo_cells(_fill_todo_cells(md), transcript), transcript),
                                      transcript)
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
    return _drop_same_task_rows(_drop_ownerless_duplicate_rows(_drop_empty_item_rows("\n".join(out))))


# 事項只有議題名稱或段落名稱、不是動作 (11-05 3 分鐘 e2e：「| 招募方面 | 王經理 | - |」「| 決議 | - | - |」)
_LABEL_ONLY_ITEM_RE = re.compile(r"^(?:決議|待辦(?:事項)?|摘要|總結|[\u4e00-\u9fff]{0,6}方面)$")


def _drop_empty_item_rows(md: str) -> str:
    """事項欄是空的、或只有「X方面／決議」這類標籤沒有期限的列拿掉，多出來的空白格去掉
    (11-04 夜間 3 分鐘實錄：「|  | 王經理 | - |  |」)。整張表都拿光時留一列「無」。"""
    out, section, dropped, kept, last_row = [], None, 0, 0, None
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("## "):
            section = s[3:].strip()
        if section == "待辦" and s.startswith("|") and not _is_sep(s):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not cells[0] or (len(cells) >= 3 and _LABEL_ONLY_ITEM_RE.match(cells[0]) and cells[2] in _TODO_SKIP):
                dropped += 1
                continue
            kept += cells[0] not in _TODO_HEADER_ECHO
            while len(cells) > 3 and not cells[-1]:
                cells.pop()
            line = "| " + " | ".join(cells) + " |"
        if section == "待辦" and s.startswith("|"):
            last_row = len(out)
        out.append(line)
    if dropped and not kept and last_row is not None:
        out.insert(last_row + 1, "| 無 | - | - |")
    return "\n".join(out)


def _drop_ownerless_duplicate_rows(md: str) -> str:
    """同一事項 (去空白後相同) 已有一列寫了負責人時，負責人是「-」的那列拿掉。
    例：模型寫「| 回報星光材料的交期 | 理協理 | 週五 |」，結尾總結補上「李協理」那列，對照逐字稿後前一列的負責人變成「-」。"""
    rows = [l.strip() for l in md.splitlines() if l.strip().startswith("|") and not _is_sep(l)]
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    owned = {re.sub(r"\s", "", c[0]) for c in cells if len(c) >= 3 and c[0] != "事項" and c[1] not in _TODO_SKIP}
    drop = {r for r, c in zip(rows, cells) if len(c) >= 3 and c[0] != "事項" and c[1] in _TODO_SKIP
            and re.sub(r"\s", "", c[0]) in owned}
    if not drop:
        return md
    return "\n".join(l for l in md.splitlines() if l.strip() not in drop)

def _drop_same_task_rows(md: str) -> str:
    """同一負責人、同一件事寫了兩列時只留一列 (11-05 離線 3／5 分鐘紀要：每份平均 0.6／0.7 列是重複的)：
    - 一列沒有期限 (結尾總結「待辦事項是劉課長送預算表」)，另一列有期限且事項有兩個以上相連的字相同 → 拿掉沒期限的
      (「| 送預算表 | 劉課長 | - |」與「| 劉課長送預算表 | 劉課長 | 10月7日前 |」)；
    - 期限相同 (不計「前」) 且事項有兩個以上相連的字相同 → 留第一列 (「| 主持面試 | 劉課長 | 8月26日 |」與「| 面試：劉課長主持 | 劉課長 | 8月26日前 |」)。
    比對時事項去掉負責人與常見字。"""
    rows = [l.strip() for l in md.splitlines() if l.strip().startswith("|") and not _is_sep(l)]
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    data = [(r, c) for r, c in zip(rows, cells) if len(c) >= 3 and c[0] not in _TODO_HEADER_ECHO and c[0] not in _EMPTY_CELL
            and c[1] not in _TODO_SKIP and _CJK_RE.search(c[0])]      # 英文事項逐字比對沒有意義 (「Test firmware」「Test 100 units」)
    core = lambda c: "".join(ch for ch in re.sub(r"[\s，,。、：:|\-]", "", c[0].replace(c[1], "")) if ch not in _TODO_COMMON_CHARS)
    when = lambda c: re.sub(r"[\s前]", "", c[2])
    drop = set()
    for i, (ri, ci) in enumerate(data):
        for j, (rj, cj) in enumerate(data):
            if i == j or ri in drop or rj in drop or ci[1] != cj[1] or _longest_common_run(core(ci), core(cj)) < 2:
                continue
            if ci[2] in _TODO_SKIP and cj[2] not in _TODO_SKIP:
                drop.add(ri)
            elif ci[2] not in _TODO_SKIP and when(ci) == when(cj) and i > j:
                drop.add(ri)
    if not drop:
        return md
    return "\n".join(l for l in md.splitlines() if l.strip() not in drop)


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


def _recap_notes_en(recap: str, max_lines: int = 6, max_len: int = 160, topics=()) -> str:
    """英文的結尾總結：一句一行 (句子本身就是「Topic: we decided…, and John will…」)，去掉開場與道謝。
    topics：正文的議題名稱。STT 沒斷句時 (「training names go to mary by thursday, firmware we decided…, and Kevin owns
    the testing customer order」，78 份英文 e2e 有 6 份) 在「, 議題名稱」前再切開；太長時截在字與字之間，不留半個字。"""
    out = []
    # 多字的議題名稱也認第一個字 (正文「the firmware update for…」，總結只說「firmware」)；要接在逗號後面才切
    names = sorted({x.lower() for t in topics for x in (t, t.split()[0]) if len(x) >= 5}, key=len, reverse=True)
    split_re = re.compile(r",\s+(?=(?:" + "|".join(re.escape(n) for n in names) + r")\b)", re.I) if names else None
    # 句尾黏著下一個議題 (「Kevin owns the testing customer order」)：多字議題名稱、前面不是冠詞或介系詞時切掉
    multi = [n for n in names if " " in n]
    tail_re = re.compile(r"\b(\w+)[\s,]+(?:" + "|".join(re.escape(n) for n in multi) + r")\s*$", re.I) if multi else None
    keep_words = {"the", "a", "an", "on", "for", "to", "of", "with", "about", "our", "this", "that", "new"}
    for sent in re.split(r"(?<=[.!?;])\s+", recap or ""):
        pieces = split_re.split(sent) if split_re and len(sent) > 60 else [sent]
        for piece in pieces:
            t = _EN_RECAP_FILLER_RE.sub("", piece.strip()).strip(" .,")
            if tail_re:
                t = tail_re.sub(lambda m: m.group(0) if m.group(1).lower() in keep_words else m.group(1), t).strip(" ,")
            if len(t) > max_len:
                t = t[:max_len].rsplit(" ", 1)[0].rstrip(" ,")
            if len(t) >= 8:
                out.append("- " + t)
    return "\n".join(out[:max_lines])


# 結尾總結常見的議題名稱 (講稿與真實會議的週會常用語)；STT 漏了逗號時用來找出下一個議題從哪裡開始
_RECAP_TOPICS = ("教育訓練", "行銷活動", "倉儲管理", "客戶訂單", "品質異常", "供應商", "新系統", "訓練", "工安", "公安", "展覽",
                 "預算", "招募", "品質", "訂單", "庫存", "出差", "韌體", "上線", "倉儲", "行銷", "採購", "客訴", "研發", "人事")


def _split_topic_start(clause: str) -> List[str]:
    """「待辦事項是黃專員週三回報宏宇科技的交期展覽方面」：STT 沒在下一個議題前斷句，「X方面」黏在上一句後面。
    從「方面」往前找議題名稱 (已知議題名或前一個英文碎片的邊界)，拆成兩句；「方面」在句首附近時不動。"""
    i = clause.find("方面")
    if i <= 6:
        return [clause]
    head = clause[:i]
    for t in _RECAP_TOPICS:
        if head.endswith(t):
            cut = i - len(t)
            break
    else:
        m = re.search(r"[A-Za-z\s\-]+([\u4e00-\u9fff]{2,4})$", head)
        if not m:
            return [clause]
        cut = i - len(m.group(1))
    before, after = clause[:cut].rstrip(" ,，-"), clause[cut:]
    return [before, after] if len(before) >= 3 else [clause]


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
    clauses = [x for c in clauses for x in _split_topic_start(c)]
    groups: List[List[str]] = []
    for c in clauses:
        # 「X方面」開新的一行；議題名稱被 STT 聽丟時，同一行出現第二個「決議是」也另起一行
        # (11-03 夜間 120 分鐘：「決議是七月四日…待辦事項，是蔡組長…，決議是3月19日完成報到」)
        second_decision = groups and re.match(r"決議\s*(?:是|為)", c) and any("決議" in x for x in groups[-1])
        if "方面" in c or not groups or second_decision:
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
    表格只有表頭沒有資料列時補一列「| 無 | - | - |」(11-01 5 分鐘 e2e：模型只印了 20 多列空白列，清掉後表格是空的)。
    分隔線被截斷 (「| 事項 | 負責人 | 時間 |」後面只有「|---」，11-04 夜間 3 分鐘實錄) 時連分隔線一起補，
    否則後面補進來的列不會顯示成表格。"""
    out, section, width, rows, sep_at, head_at = [], None, 0, 0, None, None

    def close_table():
        if section == "待辦" and width and not rows:
            if sep_at is not None:
                out.insert(sep_at + 1, empty_row)
            elif head_at is not None and out[head_at].strip().strip("|").split("|")[0].strip() not in ("無", SECTION_MISSING_NOTE):
                out[head_at + 1:head_at + 1] = ["|" + "---|" * width, empty_row]

    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            close_table()
            section, width, rows, sep_at, head_at = s[3:].strip(), 0, 0, None, None
        if section == "待辦" and s.startswith("|") and s.endswith("|"):
            if _is_sep(s):
                sep_at = len(out)
            else:
                cells = s.count("|") - 1
                if not width:
                    width, head_at = cells, len(out)
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
    return "\n".join(_fill_empty_sections(_pad_todo_rows(_reorder_sections(text.strip().splitlines()), empty_row), empty_row)).strip()


def _fill_empty_sections(lines: List[str], empty_row: str) -> List[str]:
    """段落標題在、內容卻是空的 (11-04 3 分鐘 e2e：模型寫了「## 待辦」就直接接「## 決議」)：待辦補一個只有「無」的表格，
    其他段補「- 無」。待辦表在才能讓後面 merge_recap_into_minutes() 把結尾總結的待辦補進去。"""
    out: List[str] = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        s = lines[i].strip()
        if s.startswith("## "):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines) or lines[j].strip().startswith("## "):
                out += (["| 事項 | 負責人 | 時間 |", "|---|---|---|", empty_row] if s == "## 待辦" else ["- 無"])
        i += 1
    return out


# 長會議紀要組裝 (2026-10-27)：qwen3:1.7b 無法跨 20+ 條重點做歸納 (實測會照抄、待辦表留空、決議擠掉)，
# 因此 map 之後不再交給 LLM reduce，而是以程式從各段重點組出三段式紀要：內容全部來自 LLM 的分段重點，不捏造。
# 姓名部分排除數量詞與動詞 (避免「增加三位工程師」「需楊工程師」被當成負責人)
# 「是」也排除：結尾總結「待辦事項是林主任…」的「項是林主任」不能整串當成負責人 (10-31 10 分鐘 e2e)
# 負責人名字的字：排除常接在名字前面的字。「市／示／事／像／象」是 STT 把「事項是」聽壞的殘渣
# (11-03 夜間 10 分鐘實錄：「待會兒時向市李協理安排」→ 負責人寫成「向市李協理」)，不會是名字的第一個字。
_NAME_CHAR = r"(?:(?![需請加由交給與和的位名個人是一二三四五六七八九十兩市示事像象])[\u4e00-\u9fff])"
_OWNER_RE = re.compile(r"(?:負責人(?:是|為)?\s*)?(" + _NAME_CHAR + r"{1,3}(?:經理|副理|協理|主任|專員|工程師|公程師|處長|課長|組長|主管|老師))")
# 「今天」只有當期限用才算 (今天下班前、今天內)；「今天的週會」「今天有幾個議題」「今天這到 (今天就到這裡)」不是期限
# (11-03 夜間 3 分鐘實錄：「待辦事項是王經理主持面試今天這到」的時間被填成「今天」)。
# 「上週三樓產線發生事故」的「週三」是過去，不是期限 (11-04：129 份紀要的待辦表 341 列中 8 列「公安：上週三事故 | 許經理 | 週三」)
# STT 偶爾把日期寫成國字 (「七月一日前送到財務部」，110 份逐字稿 38 份共 204 次；阿拉伯數字 3999 次)，一樣算期限
_TIME_RE = re.compile(r"((?:1[0-2]|0?[1-9])\s*月\s*(?:3[01]|[12]\d|0?[1-9])\s*日|"
                      r"(?:十[一二]?|[一二三四五六七八九]|1[0-2]|0?[1-9])\s*月\s*(?:三十一?|二?十[一二三四五六七八九]?|[一二三四五六七八九])\s*[日號]|\d{1,2}[./-]\d{1,2}(?:日)?|\d{1,2}\s*日前|(?<!上)(?<!上\s)下?週[一二三四五六日]|月底|下週|明天|今天(?![就先到開這說講的有是])|年底)")
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
# STT 常在「待辦事項」與負責人之間多打標點 (「待辦事項,是蔡組長,負責工安複訓」，95 份裡 11 次)，標點一起吃掉
_TODO_MARK_RE = re.compile(r"待辦事項\s*[,，、.]?\s*(?:是|為|：|:)?\s*[,，、.]?\s*|待辦\s*(?:是|為|：|:)\s*[,，、.]?\s*")
_CLAUSE_SPLIT_RE = re.compile(r"[，；;。]|,(?!\d)")          # 不切「1,200」這類數字
_ACTION_LEAD_RE = re.compile(r"^(?:在|於|要|會|將|請|負責|之前|以前|前|是)+")   # 「劉課長是訂單調貨」


def _clean_action(item: str) -> str:
    """事項去掉 STT 插進來的書名號／引號 (「李協理《市安排回測是》」) 與殘留的「前」。"""
    item = re.sub(r"[《》〈〉「」『』]", "", item or "").strip()
    return _ACTION_LEAD_RE.sub("", re.sub(r"(?<=.{2})前$", "", item)).strip() or item


def _todo_parts(line: str):
    """回傳 (事項, 負責人, 時間) 或 None。
    有「待辦事項是…」時只看它後面那一句：負責人從這句取，時間先找這句、沒有才用整行的 (決議的期限)，
    事項寫成去掉負責人與時間後剩下的動作。
    例：「訂單方面，決議是先簽50臺，待辦事項是林主任週三前回覆華新」→ (回覆華新, 林主任, 週三)。
    原本會取第一個子句「訂單方面」當事項，時間也可能抓到決議裡的日期。"""
    parts = _TODO_MARK_RE.split(line)
    if len(parts) > 1:
        clauses = _CLAUSE_SPLIT_RE.split(parts[-1])
        seg = clauses[0].strip()
        if len(clauses) > 1 and _OWNER_RE.fullmatch(seg):      # 「蔡組長,負責工安複訓」：負責人自成一句時接上下一句
            seg = seg + clauses[1].strip()
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
                return _clean_action(item)[:20], mo.group(1), mt.group(1)
    mo, mt = _OWNER_RE.search(line), _TIME_RE.search(line)
    if not (mo and mt):
        return _todo_parts_en(line) if detect_language(line) == "en" else None
    first = _CLAUSE_SPLIT_RE.split(line)[0].strip()
    if mo.end() <= len(first):
        # 負責人就在第一個子句裡 (STT 把「待辦事項是」聽成別的字時，整段總結常沒有標點)：事項取負責人後面的動作
        rest = first[mo.end():].replace(mt.group(0), "", 1)
        rest = _ACTION_LEAD_RE.sub("", rest.strip()).strip()
        if len(rest) >= 4:
            return _clean_action(rest)[:20], mo.group(1), mt.group(1)
    if re.fullmatch(r".{0,6}方面", first):
        # 「招募方面，決議是12月3日前完成報到，拜半事像是王經理主持面試」：事項取負責人那一句的動作，不取議題名
        owner_clause = next((c for c in _CLAUSE_SPLIT_RE.split(line) if mo.group(0) in c), "")
        act = owner_clause[owner_clause.find(mo.group(0)) + len(mo.group(0)):].replace(mt.group(0), "", 1)
        act = _ACTION_LEAD_RE.sub("", act.strip()).strip()
        if len(act) >= 2:
            return _clean_action(act)[:20], mo.group(1), mt.group(1)
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


def _drop_listed_todo_clause(decisions: List[str], todo_rows: List[str], known=None) -> List[str]:
    """結尾總結的決議條目常是「X方面，決議是…，待辦事項是某人某日前…」。待辦那半句已經寫進待辦表
    (認得出負責人與時間) 時，決議只留前半句，不在兩段重複；待辦表沒收到的照舊整句保留，負責人不會不見。
    known((事項, 負責人, 時間)) 另外判斷這件事是否已在表上 (merge_recap_into_minutes 用，見 _todo_known)。"""
    listed = {r.split("|")[1].strip() for r in todo_rows if r.count("|") >= 4}
    table = [(r.split("|")[1].strip(), r.split("|")[2].strip()) for r in todo_rows if r.count("|") >= 4]
    out = []
    for line in decisions:
        parts = _TODO_MARK_RE.split(line, 1)
        got = (_todo_parts(line) or _recap_timeless_todo(line)) if len(parts) == 2 else None
        if got and known is not None and known(got):
            got = ("", "", "")
        elif len(parts) == 2 and not (got and got[0] in listed) and _clause_in_table(_CLAUSE_SPLIT_RE.split(parts[1], 1)[0], table):
            got = ("", "", "")       # 待辦子句沒有期限 (「待辦事項是劉課長送預算表」)，但同一負責人、同一件事已在待辦表
        elif got and got[0] not in listed:
            got = None
        if not got:
            out.append(line)
            continue
        if not parts[0].strip(" -，,：:") and known is not None:
            continue            # 整條就是待辦 (模型寫的「- 待辦：劉課長主持面試」)，表上已有：整條拿掉
        # 只拿掉待辦那一個子句；後面若還接著別的議題 (STT 漏了「方面」時常黏成一行) 照樣保留。
        # 負責人自成一句時 (「待辦事項是李協理，周會報告」) 連下一句一起拿掉
        pieces = _CLAUSE_SPLIT_RE.split(parts[1], 2)
        seg = pieces[0]
        if len(pieces) > 1 and _OWNER_RE.fullmatch(seg.strip()):
            m2 = re.match(re.escape(seg) + r"[，；;。,]\s*" + re.escape(pieces[1]), parts[1])
            seg = m2.group(0) if m2 else seg
        head = re.sub(r"[，,；;、\s]*待辦(?:事項)?\s*$", "", parts[0]).rstrip("，,；;、 ")
        rest = parts[1][len(seg):].lstrip("，,；;。、 ")
        trimmed = f"{head}，{rest}" if head and rest else (head or rest)
        out.append(trimmed if len(head) >= 4 else line)
    return out


def _clause_in_table(seg: str, table: List[tuple]) -> bool:
    """「劉課長送預算表」是否已在待辦表：負責人相同，且去掉負責人後的動作出現在該列事項裡 (或用字重疊六成以上)。"""
    mo = _OWNER_RE.search(seg or "")
    if not mo:
        return False
    action = re.sub(r"[\s，,。]", "", seg.replace(mo.group(0), "", 1))
    if len(action) < 2:
        return False
    for item, owner in table:
        if mo.group(1) not in owner and mo.group(1) not in item:
            continue
        it = re.sub(r"\s", "", item)
        if action in it or len(set(action) & set(it)) >= 0.6 * len(set(action)):
            return True
    return False


def _topic_key(line: str) -> str:
    head = re.split(r"[：:]", line, 1)
    return head[0].strip() if len(head) == 2 and 1 <= len(head[0].strip()) <= 8 else line[:4]


# 結尾總結的「X方面，__是…」：__ 是「決議」被 STT 聽壞 (爭一、旬一式、隨意、只一…，每次都不一樣)；
# 只用在最後一段 (總結原句)，把這種句子當決議，不讓它佔摘要名額
_RECAP_STATEMENT_RE = re.compile(r"^[\u4e00-\u9fff]{1,6}方面\s*[，,、]?\s*[\u4e00-\u9fff]{1,3}\s*是")


def _recap_statement(line: str) -> bool:
    """結尾總結的決議句：「X方面，__是…」，或「__是先簽五十臺試單待辦事項是林主任…」(「方面」與「決議」都聽壞，
    但後面接著待辦事項：總結的句型是「決議是…，待辦事項是…」，前半句就是決議)。11-05 60 份喇叭實錄：11 句，
    原本被選進摘要、而且帶著待辦那半句 (night7_m10「隨時韌體修正版10月11日上線待辦事項《市李協理安排回測是》」)。"""
    if _RECAP_STATEMENT_RE.match(line):
        return True
    parts = _TODO_MARK_RE.split(line, 1)
    return len(parts) == 2 and len(re.sub(r"[\s，,、：:]", "", parts[0])) >= 6


def _long_todo_rows(per_all: List[List[str]], body_todos: List[str], max_todos: int) -> List[str]:
    """長會議待辦表：結尾總結的待辦 > 逐字稿正文明講的交辦 > 各段筆記抽出的列。
    11-04 離線 35 份 10–60 分鐘 e2e：各段筆記抽出的列常是議題摘要 (「公安：上週二事故未受傷 | 許經理 | 週二」)，
    負責人與期限對得上講稿同一議題的只有 65%；正文交辦 86%，而且每份約 25 列 (表只有 6–12 列)。
    名額不夠時正文交辦平均取自整場會議 (不只取前面)；顯示順序：正文交辦、筆記、結尾總結。"""
    cells = lambda l: [x.strip() for x in l.strip().strip("|").split("|")]
    tail = per_all[-1] if len(per_all) > 1 else []
    tail_rows = extract_todos(tail, max_rows=max_todos) if tail else []
    note_rows = extract_todos([l for p in (per_all[:-1] if tail else per_all) for l in p], max_rows=max_todos * 2)
    chosen = list(tail_rows[:max_todos])
    body = []
    for r in body_todos:
        if not _todo_known(cells(r), [cells(x) for x in chosen + body]):
            body.append(r)
    room = max_todos - len(chosen)
    if len(body) > room > 0:
        body = [body[round(i * (len(body) - 1) / max(1, room - 1))] for i in range(room)] if room > 1 else body[:1]
    body = body[:max(0, room)]
    notes = []
    for r in note_rows:
        if len(chosen) + len(body) + len(notes) >= max_todos:
            break
        if not _todo_known(cells(r), [cells(x) for x in chosen + body + notes]):
            notes.append(r)
    return body + notes + chosen


def assemble_long_minutes(chunk_notes: List[str], max_summary: int = 8, max_decisions: int = 3, tail_lines: int = 4,
                          max_todos: int = 6, body_todos: Optional[List[str]] = None) -> str:
    """chunk_notes：每段的 compact_notes() 輸出 (依時間順序)。
    摘要：最後一段 (會議結尾常有總結) 先保留 tail_lines 條，其餘名額每段先取第 1 條、有剩再取第 2 條…，維持時間順序；
    待辦/決議由規則從全部重點擷取。body_todos：逐字稿正文裡明講的交辦 (transcript_todo_rows)，見 _long_todo_rows。"""
    per_all = [_note_lines(tidy_dates(to_taiwan_traditional(n))) for n in chunk_notes]
    all_lines = [l for p in per_all for l in p]
    # 決議先取最後一段 (結尾總結的原句，會議最後拍板的內容)，名額有剩再依時間順序補前面各段的 (11-03 夜間 60 分鐘實錄：
    # 前段先佔滿 3 個名額，總結裡的兩條決議被擠到摘要，摘要因此出現「公安方面決議是…待辦事…」原句)
    tail = per_all[-1] if per_all else []
    ordered = [l for l in tail if _DECISION_RE.search(l)] + \
              [l for l in tail if len(per_all) > 1 and not _DECISION_RE.search(l) and _recap_statement(l)] + \
              [l for p in per_all[:-1] for l in p if _DECISION_RE.search(l)]
    decisions: List[str] = []
    dup_of_decision = set()
    for l in ordered:          # 同一件事在總結與前段各寫一次 (「總結：決議先簽60臺試單」「訂單：決議先簽60臺試單」) 只留一條
        body = set(re.sub(r"[\s，,。、：:]", "", re.split(r"[：:]|方面", l, 1)[-1]))
        if any(len(body & set(re.sub(r"[\s，,。、：:]", "", re.split(r"[：:]|方面", d, 1)[-1]))) >= 0.8 * max(1, len(body))
               for d in decisions):
            dup_of_decision.add(l)
            continue
        decisions.append(l)
        if len(decisions) >= max_decisions:
            break
    # 選進「決議」的條目不再放進「摘要」(2026-10-29 228 30 分鐘 e2e：同一條同時出現在兩段，duplicate_lines 不過門檻)；
    # 名額留給其他議題。全部條目都是決議時才退回原本的選法。
    per = [[l for l in p if l not in decisions and l not in dup_of_decision] for p in per_all]
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
    todos = None
    if body_todos:
        try:
            todos = _long_todo_rows(per_all, body_todos, max_todos)
        except Exception:       # noqa: BLE001 — 正文交辦只是加分，出錯時退回各段筆記
            todos = None
    if todos is None:
        todos, body_todos = extract_todos(all_lines, max_rows=max_todos), None
    cells = lambda l: [x.strip() for x in l.strip().strip("|").split("|")]
    decisions = _drop_listed_todo_clause(decisions, todos,
                                         known=(lambda t: _todo_known(t, [cells(r) for r in todos])) if body_todos else None)
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


_TODO_COMMON_CHARS = set("的前日月週周在是和與及會要將把再並完成")


_EN_TOPIC_RE = re.compile(r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|next|last|final)\s+topic\s+is\s+(?:the\s+)?"
                          r"([a-z][a-z \-]{2,30}?)\s*(?=[.,;]|\s+(?:for|of|on|from)\b)", re.I)


def _recap_timeless_todo_en(line: str, topics=()):
    """英文總結「and Kevin owns the testing, customer order」：有人名＋動作、沒有期限 → (事項, 負責人, "-")。
    11-05 228 en_short：模型漏了 Kevin 那列 (回顧事實 0.83)，總結原句裡有、但沒期限所以 extract_todos 不收。
    topics：正文的議題名稱 (「The fourth topic is the customer order」)；STT 沒斷句時黏在動作後面的下一個議題從那裡切掉。"""
    if _EN_TIME_RE.search(line):
        return None
    mo = _EN_OWNER_RE.search(line)
    owner = mo.group(1) if mo else None
    if not owner or owner.split()[0].lower() in ("we", "they", "i", "you", "it", "this", "that", "the", "all", "everyone", "team") \
            or owner.split()[0].lower() in _EN_NOT_NAMES:
        return None
    action = re.split(r"[,;.]", mo.group(2) or mo.group(3) or "")[0].strip()
    for t in topics:
        action = re.split(r"\s+" + re.escape(t) + r"\b", action, 1, flags=re.I)[0].strip()
    return (action[:40], owner, "-") if len(action) >= 3 else None


def _recap_timeless_todo(line: str):
    """「待辦事項是李協理，周會報告」：有負責人、沒有期限的總結待辦 → (事項, 負責人, "-")；其他回傳 None。"""
    if not _CJK_RE.search(line or ""):
        return _recap_timeless_todo_en(line or "")
    parts = _TODO_MARK_RE.split(line, 1)
    if len(parts) != 2:
        return None
    clauses = _CLAUSE_SPLIT_RE.split(parts[1])
    seg = clauses[0].strip()
    if len(clauses) > 1 and _OWNER_RE.fullmatch(seg):
        seg = seg + clauses[1].strip()
    mo = _OWNER_RE.search(seg)
    if not mo or _TIME_RE.search(line):
        return None
    action = _ACTION_LEAD_RE.sub("", seg.replace(mo.group(0), "", 1).strip()).strip()
    return (action[:20], mo.group(1), "-") if len(action) >= 2 else None


def _todo_known(row, table) -> bool:
    """(事項, 負責人, 時間) 是否已在待辦表 (table 為各列的格子)：同一負責人 (負責人欄或寫在事項欄)，
    且期限相同、或事項有一個以上相同的字 (不算「的、前、完成」這類常見字)。
    例：「主持面 | 王經理 | 12日前」與「招募業務部面試 | 王經理 | 12月3日前」是同一件事；「回覆客戶 | 王經理」不是。"""
    item, owner, when = row[:3]
    core = lambda x: set(re.sub(r"[\s，,。、：:|\-]", "", x)) - _TODO_COMMON_CHARS
    for c in table:
        if len(c) < 3 or not owner or (owner not in c[1] and owner not in c[0]):
            continue
        if when not in _TODO_SKIP and re.sub(r"[\s前]", "", when) == re.sub(r"[\s前]", "", c[2]):
            return True
        if core(item) & (core(c[0].replace(owner, "")) or core(c[0])):
            return True
    return False


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
    topics = [m.group(1).strip() for m in _EN_TOPIC_RE.finditer(" ".join(parts[:-1]))] if not _CJK_RE.search(parts[-1]) else []
    lines = _note_lines(_recap_notes_en(parts[-1], topics=topics) if topics else recap_notes(parts[-1]))
    if not lines:
        return md
    m = _minutes_section(md, "待辦")
    rows = extract_todos(lines, max_rows=max_rows)
    timeless = [_recap_timeless_todo_en(l, topics) if topics else _recap_timeless_todo(l) for l in lines]
    rows += [f"| {t[0]} | {t[1]} | {t[2]} |" for t in timeless if t][:max(0, max_rows - len(rows))]
    cells = lambda l: [x.strip() for x in l.strip().strip("|").split("|")]
    if m and rows:
        body = m.group(2).rstrip("\n").splitlines()
        table = [l for l in body if l.strip().startswith("|") and not _is_sep(l.strip())]
        data = table[1:] if table else []
        placeholder = [l for l in data if l.count("|") >= 2 and l.split("|")[1].strip() in ("無", SECTION_MISSING_NOTE)]
        real = [l for l in data if l not in placeholder]
        add = []
        for r in rows:      # 同一負責人已有同一件事才不加 (原本是負責人出現過就不加，另一件事會被丟掉)
            if not _todo_known(cells(r), [cells(l) for l in real + add]):
                add.append(r)
        add = add[:max(0, max_rows - len(real))]
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
        known = lambda t: _todo_known(t, [cells(r) for r in todo_rows if not _is_sep(r)][1:])
        # 模型 (或照抄總結) 寫的決議也一樣：待辦那半句已在表上就拿掉
        trimmed_real = ["- " + x for x in _drop_listed_todo_clause([l[2:] for l in real], todo_rows, known=known)]
        changed_real = trimmed_real != real
        real = trimmed_real
        md_norm = re.sub(r"[\s，,。、：:]", "", md)
        add = []
        for line in lines:
            if "決議" in line:
                kept = _drop_listed_todo_clause([line], todo_rows, known=known)
                if not kept:        # 整條就是表上已有的待辦 (「待辦事項是王經理主持面試」)：不當決議
                    continue
                d = kept[0]
                core = _CLAUSE_SPLIT_RE.split(re.split(r"決議\s*(?:是|為|：|:)", d, 1)[-1])[0].strip()
                # 「決議是12日前完成報到待辦事項是王經理主持面」：STT 沒斷句時待辦黏在決議後面，比對只看決議本身
                core = _TODO_MARK_RE.split(core, 1)[0].strip()
            elif not _CJK_RE.search(line) and (md_ := _DECISION_RE.search(line)):     # 英文總結：「we agreed on …」
                d = line
                core = re.split(r"[,;]", line[md_.start():])[0].strip()
            else:
                continue
            if len(core) >= 4 and not _already_said(core, md_norm, [x[2:] for x in real] + add):
                add.append(d)
        add = add[:max(0, max_decisions - len(real))]
        if add or changed_real:
            rest = md[m.end():]
            body = "\n".join((real or ["- 無"] if not add else real) + [f"- {x}" for x in add])
            md = md[:m.start(2)] + body + ("\n\n" + rest.lstrip("\n") if rest.strip() else "\n")
    return md.rstrip() + "\n" if md.endswith("\n") else md.rstrip()



def drop_listed_todos_from_summary(md: str) -> str:
    """摘要條目裡的「…待辦事項是某人某日前…」：多半是結尾總結的原句 (「決議」被 STT 聽壞，認不出是決議，就被選進摘要)。
    待辦表已有那件事時只留前半句；整條就是待辦時整條拿掉，摘要至少留一條。表上沒有的照舊整句保留。
    11-05 60 份喇叭實錄紀要：22 份摘要有這種條目 (「隨時韌體修正版10月11日上線待辦事項《市李協理安排回測是》」)。"""
    sm, tm = _minutes_section(md or "", "摘要"), _minutes_section(md or "", "待辦")
    if not sm or not tm or not _TODO_MARK_RE.search(sm.group(2)):
        return md
    cells = lambda l: [x.strip() for x in l.strip().strip("|").split("|")]
    todo_rows = [l.strip() for l in tm.group(2).splitlines() if l.strip().startswith("|") and not _is_sep(l.strip())][1:]
    todo_rows = [r for r in todo_rows if r.count("|") >= 4 and cells(r)[0] not in ("無", SECTION_MISSING_NOTE)]
    if not todo_rows:
        return md
    known = lambda t: _todo_known(t, [cells(r) for r in todo_rows])
    body = sm.group(2).rstrip()
    out, changed = [], False
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("- ") and _TODO_MARK_RE.search(s):
            kept = _drop_listed_todo_clause([s[2:]], todo_rows, known=known)
            if kept != [s[2:]]:
                changed = True
                if kept and kept[0].strip(" -，,：:"):
                    out.append(line[:len(line) - len(line.lstrip())] + "- " + kept[0].strip())
                continue
        out.append(line)
    if not changed or not any(l.strip().startswith("- ") for l in out):
        return md
    return md[:sm.start(2)] + "\n".join(out) + sm.group(2)[len(body):] + md[sm.end(2):]


# STT 沒在下一個議題前斷句時，「…請劉課長調貨第十五個議題是客戶訂單」的議題編號 (還常聽錯、編號亂跳) 黏在條目尾巴
_GLUED_TOPIC_TAIL_RE = re.compile(r"[，,、\s]*第\s*[\d一二三四五六七八九十百]{1,4}\s*個?\s*[議意]題(?:\s*[是為]?\s*[\u4e00-\u9fff]{0,4})?\s*[。.]?\s*$")


def cut_glued_topic_numbers(md: str) -> str:
    """摘要／決議條目尾巴黏著的「第N個議題(是X)」拿掉 (前面至少留 6 個字)；條目開頭的「第28議題品質異常：」是議題名稱，不動。"""
    if not md or not re.search(r"[議意]題", md):
        return md
    out = []
    for sec in ("摘要", "決議"):
        m = _minutes_section(md, sec)
        if not m:
            continue
        lines = m.group(2).split("\n")
        for i, line in enumerate(lines):
            if line.lstrip().startswith("- "):
                cut = _GLUED_TOPIC_TAIL_RE.sub("", line)
                if cut != line and len(re.sub(r"[\s\-，,：:]", "", cut)) >= 6:
                    lines[i] = cut
        md = md[:m.start(2)] + "\n".join(lines) + md[m.end(2):]
    return md


_PLACEHOLDER_ITEM_RE = re.compile(r"^(?:[-*•]\s*)*\**\s*(?:無|none|n/?a|nothing|（無）|\(none\))\s*[.。]?\s*\**$", re.I)


def drop_placeholder_items(md: str) -> str:
    """摘要／決議段已有實際條目時，拿掉佔位的「- 無」「- None」「- - None」(結尾總結補進決議後，模型原本寫的 None 還留著)。
    11-05 228：60 份英文 e2e 紀要有 7 份決議段夾著 None，其中 2 份因此有重複條目 (en_long「- None」兩次)。"""
    for sec in ("摘要", "決議"):
        m = _minutes_section(md or "", sec)
        if not m:
            continue
        lines = m.group(2).split("\n")
        items = [l for l in lines if l.strip().startswith("- ")]
        holders = [l for l in items if _PLACEHOLDER_ITEM_RE.match(l.strip()[2:].strip())]
        if not holders or len(holders) == len(items):
            continue
        md = md[:m.start(2)] + "\n".join(l for l in lines if l not in holders) + md[m.end(2):]
    return md

# 逐字稿正文 (結尾總結之前) 裡明講的交辦 → 待辦表列。qwen3:1.7b 單次紀要的待辦表常只寫 1–2 列：
# 11-04 離線 37 份 1–15 分鐘 e2e 逐字稿，劇本有負責人的議題 3 分鐘 7 個、5 分鐘 13 個，紀要待辦表平均只有 1.4–1.5 列。
# 只認有明確交辦用語的負責人 (「請X」「由X」「交給X」「負責人是X」、子句開頭的「X負責／會／要」) 且同一子句或相鄰子句有期限
# (「上週三」這類過去的不算)；事項取負責人後面的動作，只有「負責／主持」時改取相鄰子句或前半句。
# 劇本核對：115 列中 104 列的負責人與期限都對得上同一個議題，其餘 11 列是 STT 聽錯的名字或日期 (刀課長、週一→週日)。
_BODY_OWNER = _NAME_CHAR + r"{1,3}?(?:經理|副理|協理|主任|專員|工程師|處長|課長|組長|主管)"
_BODY_CUE_RE = re.compile(r"(?:請|由|交給|交由)\s*(" + _BODY_OWNER + r")|負責人\s*(?:是|為)?\s*(" + _BODY_OWNER + r")"
                          r"|^(?:這件事|並|再)?\s*(" + _BODY_OWNER + r")\s*(?=負責|會|要|將|主持|安排|在)")
_BODY_TOPIC_CUT_RE = re.compile(r"第\s*\S{1,3}\s*個?\s*[議意]題")     # STT 沒斷句時，下一個議題的開頭不算進這句
_BODY_JUNK_RE = re.compile(r"[A-Za-z\-]*(?=[A-Za-z\-]*\d)(?=[\dA-Za-z\-]*[A-Za-z])[\dA-Za-z\-]{4,}")   # 字母數字混雜的碎片
_BODY_BARE_RE = re.compile(r"^(?:負責|主持|主吃|處理|安排|跟進|追蹤)?$")      # 「主吃」：STT 把主持聽壞
_BODY_LEAD_RE = re.compile(r"^(?:這件事|並且?|再|然後|在|於|要|會|將|請|由|負責|之前|以前|前|希望|把|先)+")


def _body_due(clause: str):
    """子句裡的期限。有「…前」的優先 (「下週會再安排四場面試由王經理主持希望在12月3日前完成報到」取 12月3日前，
    不取「下週」，11-04 夜間 3 分鐘實錄)；「上週三樓產線發生…」這類過去的不算。"""
    found = [m for m in _TIME_RE.finditer(clause) if clause[max(0, m.start() - 1):m.start()] not in ("上", "昨")]
    return next((m for m in found if re.match(r"\s*(?:之前|以前|前)", clause[m.end():])), found[0] if found else None)


def _body_action(text: str) -> str:
    """去掉期限 (連同「在…前」)、開頭的虛詞與結尾標點，只留動作。"""
    m = _body_due(text)
    if m:
        text = re.sub(r"(?:在|於)?\s*\0\s*(?:之前|以前|前)?", "", text[:m.start()] + "\0" + text[m.end():])
    text = re.sub(r"^(?:負責|主持|主吃)(?=希望|並|然後|在|於)", "", re.sub(r"\s+", "", text))
    text = re.sub(r"^(?:負)+(?!責)", "", text)        # 「負責」聽壞成「負負請交初稿」「負回覆客戶」
    # 下一個議題沒斷句、編號也聽壞 (「送到財務部第11個已提是公啊」「交初稿第11」)；句尾語氣詞 (「完成報到好」)
    # 「第」也漏掉、或編號多唸一個字時 (「回覆客戶七個一提是招募」「第六十一一個議題」，5 分鐘 e2e)
    text = re.split(r"第\s*[\d一二三四五六七八九十]{1,4}(?:\s*個?.{0,2}?[題提體]|\s*$)"
                    r"|(?<![\d一二三四五六七八九十])[\d一二三四五六七八九十]{1,3}\s*個\s*[一議意已][提題體]", text, 1)[0]
    text = re.sub(r"(?<=.{3})[好吧啊]$", "", text)
    return re.sub(r"[.。，,、]+$", "", _BODY_LEAD_RE.sub("", text))


def transcript_todo_rows(transcript: str, max_rows: int = 8) -> List[str]:
    """逐字稿正文中「有交辦用語的負責人＋期限」→「| 事項 | 負責人 | 時間 |」列 (中文逐字稿；結尾總結另由 merge_recap_into_minutes 處理)。"""
    return [f"| {item} | {owner} | {when} |" for item, owner, when, _ in _transcript_todos(transcript, max_rows)]


def _transcript_todos(transcript: str, max_rows: int = 12) -> List[tuple]:
    """(事項, 負責人, 時間, 出處子句) 的清單；出處用來比對模型寫的同一件事。"""
    text = (transcript or "").strip()
    if not text or detect_language(text) == "en":
        return []
    parts = split_with_recap(text)
    if len(parts) >= 2 and is_recap_chunk(parts[-1]):
        cut = text.rfind(parts[-1][:30])
        text = text[:cut] if cut > 0 else text
    clauses = [c.strip() for x in _CLAUSE_SPLIT_RE.split(text) for c in _BODY_TOPIC_CUT_RE.split(x) if c and c.strip()]
    rows, seen = [], set()
    for i, c in enumerate(clauses):
        mo = _BODY_CUE_RE.search(c)
        if not mo:
            continue
        owner = next(g for g in mo.groups() if g)
        before, after = c[:mo.start()], c[mo.end():]
        nxt = clauses[i + 1] if i + 1 < len(clauses) else ""
        prv = clauses[i - 1] if i > 0 else ""
        mt, item, src = _body_due(c), "", c
        if mt is None and nxt and _body_due(nxt) and not _BODY_CUE_RE.search(nxt) and len(nxt) <= 16:
            # 「這件事由王經理負責，週五前回覆客戶」「請李協理跟對方確認，週五回報」
            mt, a, src = _body_due(nxt), _body_action(after), nxt
            item = a if not _BODY_BARE_RE.match(a) else _body_action(nxt)
        elif mt is None and mo.group(2) and prv and _body_due(prv):
            # 「…並在6月14日前完成全員複訓，負責人是張工程師」
            mt, src = _body_due(prv), prv
            segs = [x for x in re.split(r"並|然後", prv) if _body_due(x)] or [prv]
            item = _body_action(segs[-1])
        elif mt is not None:
            a = _body_action(after) if mt.start() >= mo.end() else _body_action(re.sub(r"\s+", "", after))
            if _BODY_BARE_RE.match(a) and before:   # 「下週會再安排四場面試由王經理主持」
                m0 = _body_due(before)       # 「面試了12位候選人下週會再安排六場面試」：從時間後面取，前一句的尾巴不要
                lead = before[m0.end():] if m0 is not None and len(before[m0.end():].strip()) >= 2 else before
                segs = [x for x in re.split(r"並|然後", lead) if x.strip()] or [lead]
                a = _body_action(segs[-1])[-12:]
            item = a
            nd = _body_due(nxt) if nxt and len(nxt) <= 16 and not _BODY_CUE_RE.search(nxt) else None
            if nd is not None and not re.match(r"\s*(?:之前|以前|前)", src[mt.end():]) \
                    and re.match(r"\s*(?:之前|以前|前)", nxt[nd.end():]):
                mt, src = nd, nxt      # 「下週會再安排四場面試由王經理主持，希望在12月3日前完成報到」：期限取後一句帶「前」的
        item = _BODY_JUNK_RE.split(item, 1)[0]     # 「回覆客戶PLOG1T-E是工業…」：下一個議題聽壞、沒斷句
        if mt is None or len(item) < 2:
            continue
        if "把" in before and re.search(r"交[給由]\s*$", c[:mo.start(1) if mo.group(1) else mo.start()]):
            # 「請各部門在週一前把名單交給李協理」：李協理是收件的窗口，事項寫成整句才不會像是他要交名單
            item = _body_action(re.split(r"請|[.。]", before)[-1]) + "交給" + owner   # 「還剩8個名額請各部門把名單…」只留請後面
        key = (re.sub(r"\s", "", item), owner)
        if key in seen:
            continue
        seen.add(key)
        when = mt.group(1).replace(" ", "")
        if re.match(r"\s*(?:之前|以前|前)", src[mt.end():]) and not when.endswith("前"):
            when += "前"          # 與模型寫的「10月7日前」「週五前」一致
        rows.append((item[:20], owner, when, f"{prv}{c}{nxt}"))
        if len(rows) >= max_rows:
            break
    return rows


def _longest_common_run(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size



# 單次紀要 (短會議) 漏掉的議題：qwen3:1.7b 依議題順序寫摘要，名額 (輸出長度) 用完就停，後面的議題整個不見。
# 11-05 離線 3／5 分鐘 e2e 紀要各 33 份 (劇本 7／13 個議題)：3 分鐘第 6 個議題 (韌體) 29 份完全沒寫到；
# 5 分鐘第 8、10 個議題 (庫存、上線) 33／32 份沒寫到。逐字稿裡「第N個議題是…」的段落在紀要裡找不到時，用該段開頭的原句補一條摘要。
_TOPIC_MARK_RE = re.compile(r"第\s*[\d一二三四五六七八九十]{1,3}\s*個?\s*[議意]題\s*[是為：:，,]?\s*")
_TOPIC_FACT_RE = re.compile(r"\d{1,2}月\d{1,2}日|\d+(?:\.\d+)?(?:%|萬元?|億|臺|人|件|位|天|小時|輪|場|款|支|個)")
_ZH_DIGIT = {"一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_ZH_NUM_RE = re.compile(r"(?<![一二三四五六七八九十兩百千萬])([一二三四五六七八九兩]?十[一二三四五六七八九]?|[一二三四五六七八九兩])"
                        r"(?=(?:月|日|號|%|萬|臺|人|件|位|天|小時|輪|場|款|支|個))")


def _zh_small_num(t: str) -> int:
    if "十" not in t:
        return _ZH_DIGIT[t]
    a, b = t.split("十", 1)
    return (_ZH_DIGIT[a] if a else 1) * 10 + (_ZH_DIGIT[b] if b else 0)


def _fact_text(text: str) -> str:
    """比對用：去掉空白與標點、臺／台統一、「十二月三日／百分之九十／十五位」改成阿拉伯數字。"""
    t = re.sub(r"[\s，,。、：:；;|\-]", "", text or "").replace("台", "臺").replace("％", "%")
    t = re.sub(r"百分之([一二三四五六七八九]?十[一二三四五六七八九]?|[一二三四五六七八九]|\d+)",
               lambda m: f"{m.group(1) if m.group(1).isdigit() else _zh_small_num(m.group(1))}%", t)
    return _ZH_NUM_RE.sub(lambda m: str(_zh_small_num(m.group(1))), t)


# 段落也在「第九個是…」(STT 漏了「議題」) 結束，否則下一個議題的日期會被當成這一段的 (5 分鐘 e2e：庫存那段吃到教育訓練的 5月25日)
_TOPIC_CUT_RE = re.compile(r"第\s*[\d一二三四五六七八九十]{1,3}\s*個?\s*(?:[議意]題|是)")


def _topic_segments(body: str) -> List[str]:
    marks = list(_TOPIC_MARK_RE.finditer(body))
    segs = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        cut = _TOPIC_CUT_RE.search(body, m.end(), end)
        segs.append(body[m.end():cut.start() if cut else end].strip(" ，,。"))
    return [x for x in segs if len(x) >= 8]


def _topic_said(seg: str, md_facts: str, bullets: List[str], window: int = 90) -> bool:
    """這個議題紀要裡已經寫到了嗎：開頭 30 字與某條摘要／決議有 5 字以上相同，或段落開頭 window 字內的日期出現在紀要，
    或兩個以上的數量 (85%、12萬元、3天…) 出現在紀要。只看開頭：STT 漏掉下一個議題的編號時，後面會黏著別的議題。"""
    t = _fact_text(seg)
    if any(_longest_common_run(t[:30], _fact_text(b)) >= 5 for b in bullets):
        return True
    facts = {m.group(0) for m in _TOPIC_FACT_RE.finditer(t[:window])}
    if any("月" in f and f in md_facts for f in facts):
        return True
    return sum(f in md_facts for f in facts) >= 2


def _topic_bullet(seg: str, min_len: int = 28, max_len: int = 46) -> str:
    out = ""
    for c in _CLAUSE_SPLIT_RE.split(re.sub(r"\s+", " ", seg)):
        c = c.strip()
        if not c:
            continue
        nxt = f"{out}，{c}" if out else c
        if out and len(nxt) > max_len:
            break
        if len(nxt) > max_len:      # 第一個子句就太長 (STT 沒斷句)：截斷，但不留半個數字或日期 (「修正版預計 6」)
            nxt = re.sub(r"[\s\d.]*(?:\d\s*月\s*\d*)?$", "", nxt[:max_len]).rstrip(" ，,") or nxt[:max_len]
        out = nxt
        if len(out) >= min_len:
            break
    return out


def fill_missing_topics(md: str, transcript: str, max_bullets: int = 10) -> str:
    """單次紀要的摘要補上漏掉的議題 (見 _TOPIC_MARK_RE 上的說明)。只認中文逐字稿裡「第N個議題」的段落 (結尾總結不算)，
    已經寫到的議題 (_topic_said) 不補；補的條目是該段開頭的原句 (約 28–46 字)，接在摘要最後，摘要最多 max_bullets 條。
    11-05 離線重放 (3 分鐘 33 份／5 分鐘 33 份)：議題事實出現在紀要 197/231 → 227/231、293/429 → 394/429；
    摘要平均 4.8 → 6.2／8.5 條，回顧事實與重複條目不變；1 分鐘 41 份不變。"""
    sm = _minutes_section(md or "", "摘要")
    if not sm or not transcript or not _CJK_RE.search(transcript):
        return md
    parts = split_with_recap(transcript)
    body = transcript[:len(transcript) - len(parts[-1])] if len(parts) > 1 and is_recap_chunk(parts[-1]) else transcript
    segs = _topic_segments(body)
    if len(segs) < 2:
        return md
    text = sm.group(2).rstrip()
    lines = text.splitlines()
    real = [l.strip()[2:].strip() for l in lines if l.strip().startswith("- ") and l.strip()[2:].strip() not in ("無", SECTION_MISSING_NOTE)]
    dm = _minutes_section(md, "決議")
    decisions = [l.strip()[2:] for l in (dm.group(2).splitlines() if dm else []) if l.strip().startswith("- ")]
    md_facts = _fact_text(md)
    add: List[str] = []
    for seg in segs:
        if len(real) + len(add) >= max_bullets:
            break
        if _topic_said(seg, md_facts, real + add + decisions):
            continue
        b = _topic_bullet(seg)
        if len(b) >= 8:
            add.append(b)
            md_facts += _fact_text(b)
    if not add:
        return md
    if not real:        # 摘要只有「- 無」或缺段說明：換掉
        lines = [l for l in lines if not l.strip().startswith("- ")]
    body_lines = lines + [f"- {b}" for b in add]
    return md[:sm.start(2)] + "\n".join(body_lines) + sm.group(2)[len(text):] + md[sm.end(2):]

def merge_body_todos(md: str, transcript: str, max_rows: int = 10) -> str:
    """單次紀要 (短會議)：待辦表補上逐字稿正文裡明講、表上還沒有的交辦 (_transcript_todos)。
    - 同一負責人已有一列，且期限相同或事項有兩個以上相同的字：視為已列。
    - 模型寫的列負責人對不上逐字稿 (「-」)，且事項裡有連續三個字出現在交辦那句：補上負責人 (與期限)，不另加一列。
      (只看同一天期限或零散的相同字會補錯：「出差報告」與「許經理會在5月25日前提出改善報告」只有「報告」相同。)
      例：「| 送交預算表 | - | 10月7日前 |」+「預算表請劉課長在10月7日前送到財務部」→「| 送交預算表 | 劉課長 | 10月7日前 |」。
    - 表上的「無」佔位列換掉。"""
    if not md or not transcript:
        return md
    orig, md = md, ground_todo_table(md, transcript)        # 先知道模型寫的負責人哪些對不上逐字稿
    m = _minutes_section(md, "待辦")
    if not m:
        return orig
    body = m.group(2).rstrip("\n").splitlines()
    table = [l for l in body if l.strip().startswith("|") and not _is_sep(l.strip())]
    if not table:
        return orig
    cells = lambda l: [x.strip() for x in l.strip().strip("|").split("|")]
    norm = lambda x: re.sub(r"[\s，,。、：:前]", "", x)
    placeholder = [l for l in table[1:] if cells(l)[0] in ("無", SECTION_MISSING_NOTE)]
    real = [l for l in table[1:] if l not in placeholder and len(cells(l)) >= 3]
    rows = [cells(l) for l in real]
    table_text = norm(" ".join(" ".join(c) for c in rows))
    add, filled = [], {}
    for item, owner, when, ctx in _transcript_todos(transcript, max_rows=max_rows):
        if norm(item) in table_text:
            continue
        known = rows + [[a, b, c] for a, b, c, _ in add]
        ctx_n = norm(ctx).replace(norm(owner), "")         # 比對交辦那句時不算負責人的名字本身
        run = lambda x: _longest_common_run(norm(x).replace(norm(owner), ""), ctx_n)
        k = next((k for k, c in enumerate(rows) if c[1] in _TODO_SKIP and k not in filled
                  and _longest_common_run(norm(c[0]), norm(ctx)) >= 3), None)    # 負責人欄空著：名字聽壞的「王精理主持面」也算
        if k is not None:
            c = rows[k]
            filled[k] = [c[0], owner, c[2] if c[2] not in _TODO_SKIP else when] + c[3:]
            continue
        same = [c for c in known if c[1] not in _TODO_SKIP and (owner in c[1] or c[1] in owner)
                and (norm(c[2]) == norm(when) or len(set(norm(item)) & set(norm(c[0]))) >= 2
                     # 「| 送預算表 | 劉課長 | - |」與「預算表請劉課長在10月7日前送到財務部」：沒寫期限的列與交辦那句有三個字相連
                     or (c[2] in _TODO_SKIP and run(c[0]) >= 3))]
        if same:
            # 同一件事、模型那列沒寫期限 (「| 王經理主持面試 | 王經理 | - |」)：補上交辦那句的期限
            k = next((k for k, c in enumerate(rows) if c in same and c[2] in _TODO_SKIP and k not in filled), None)
            if k is not None:
                filled[k] = rows[k][:2] + [when] + rows[k][3:]
            continue
        add.append((item, owner, when, ctx))
    add = add[:max(0, max_rows - len(real))]
    if not add and not filled:
        return orig
    swap = {real[k]: "| " + " | ".join(v) + " |" for k, v in filled.items()}
    new_body = [swap.get(l, l) for l in body if l not in placeholder] + [f"| {a} | {b} | {c} |" for a, b, c, _ in add]
    rest = md[m.end():]
    out = md[:m.start(2)] + "\n".join(new_body) + ("\n\n" + rest.lstrip("\n") if rest.strip() else "\n")
    return out.rstrip() + "\n" if orig.endswith("\n") else out.rstrip()


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
