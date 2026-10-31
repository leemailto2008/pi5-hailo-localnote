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
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("## "):
            section, converted = s[3:].strip(), False
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
                out.append(f"- {vals[0]}：{'，'.join(vals[1:])}" if len(vals) > 1 else f"- {vals[0]}")
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
    return re.sub(r"\n{3,}", "\n\n", "\n".join(_fix_table_separators(_summary_table_to_bullets(out)))).strip()



# 待辦表的負責人／時間必須出自逐字稿 (2026-10-29 228／229 提示詞 A/B：9 份逐字稿 × 2 次，現行提示詞的 94 個待辦欄位中
# 18 個是逐字稿與講稿都沒有的，例如「工程師A」「記錄員」「陳經理」「今日」)；改提示詞會連帶少寫 70% 的正確欄位，所以改在後處理擋。
_TODO_SKIP = {"", "-", "無", "待定", "未定", "未指定", "全體", "n/a", "N/A"}


def _ground_norm(text: str) -> str:
    return to_taiwan_traditional(text or "").replace(" ", "").replace("　", "").replace("臺", "台").lower()


def ground_todo_table(md: str, transcript: str) -> str:
    """待辦表每列的負責人、時間欄若在逐字稿中找不到，改成「-」 (多人時只留找得到的人)；事項欄與其他段落不動。"""
    if not md or not transcript or "## 待辦" not in md:
        return md
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


_RECAP_RE = re.compile(r"(最後總結|總結一下|做個總結|做一下總結|總結來說|最後來總結|以下總結)")


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


def recap_notes(recap: str, max_lines: int = 6, max_len: int = 50) -> str:
    """長會議的結尾總結改用逐字稿原句：依逗號／句號切成子句，去掉客套話；遇到「X方面」就換行 (一行一個議題)，
    同一議題內再依序打包成 <= max_len 字的條列。

    2026-10-30 228 離線 A/B (13 份 10／15／30 分鐘長逐字稿，`stage/exp_recap.py`、`recap_label_eval.py`)：口頭總結本身已是
    「X 方面，決議是…，待辦事項是某人在某日前…」，交給 qwen3:1.7b 再改寫、每行又截在 40 字時，負責人與期限常被丟掉；
    改用原句後回顧事實 (逐字稿內) 0.61 → 0.81、事實保留 0.176 → 0.230，且少一次 LLM 呼叫。
    依「方面」分行比單純依長度打包，議題覆蓋 0.815 → 0.831、事實保留 0.223 → 0.230。
    """
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


def _pad_todo_rows(lines: List[str]) -> List[str]:
    """待辦表格列的格數少於表頭時補「-」(例：模型寫到「| 出差報告 |」就停，10-31 5 分鐘 e2e)。"""
    out, section, width = [], None, 0
    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            section, width = s[3:].strip(), 0
        if section == "待辦" and s.startswith("|") and s.endswith("|") and not _is_sep(s):
            cells = s.count("|") - 1
            if not width:
                width = cells
            elif cells < width:
                line = s + " - |" * (width - cells)
        out.append(line)
    return out


def ensure_sections(md: str) -> str:
    """紀要被 num_predict 截斷時，補上缺少的段落標題並明確標示未產生 (不捏造內容)；
    最後一列若是被截斷的半行表格也一併移除。三段順序調成摘要、待辦、決議，待辦表格太短的列補「-」。"""
    lines = (md or "").rstrip().splitlines()
    if lines and lines[-1].lstrip().startswith("|") and not lines[-1].rstrip().endswith("|"):
        lines.pop()
    if lines and lines[-1].strip() in ("##", "#"):
        lines.pop()
    text = "\n".join(lines).rstrip()
    for sec in ("## 摘要", "## 待辦", "## 決議"):
        if sec not in text:
            text += f"\n\n{sec}\n- {SECTION_MISSING_NOTE}"
    # 補上的缺段也要排回原位 (10-31 3 分鐘 e2e：模型先寫決議、待辦被截斷，補上的「## 待辦」原本會落在最後)
    return "\n".join(_pad_todo_rows(_reorder_sections(text.strip().splitlines()))).strip()


# 長會議紀要組裝 (2026-10-27)：qwen3:1.7b 無法跨 20+ 條重點做歸納 (實測會照抄、待辦表留空、決議擠掉)，
# 因此 map 之後不再交給 LLM reduce，而是以程式從各段重點組出三段式紀要：內容全部來自 LLM 的分段重點，不捏造。
# 姓名部分排除數量詞與動詞 (避免「增加三位工程師」「需楊工程師」被當成負責人)
# 「是」也排除：結尾總結「待辦事項是林主任…」的「項是林主任」不能整串當成負責人 (10-31 10 分鐘 e2e)
_NAME_CHAR = r"(?:(?![需請加由交給與和的位名個人是一二三四五六七八九十兩])[\u4e00-\u9fff])"
_OWNER_RE = re.compile(r"(?:負責人(?:是|為)?\s*)?(" + _NAME_CHAR + r"{1,3}(?:經理|副理|協理|主任|專員|工程師|公程師|處長|課長|組長|主管|老師))")
_TIME_RE = re.compile(r"(\d{1,2}\s*月\s*\d{1,2}\s*日|\d{1,2}[./-]\d{1,2}(?:日)?|\d{1,2}\s*日前|下?週[一二三四五六日]|月底|下週|明天|今天|年底)")
_DECISION_RE = re.compile(r"決定|決議|定案|確定|通過|同意|維持|改為|改成|暫停|取消")


def _note_lines(notes: str) -> List[str]:
    return [l[2:].strip() for l in (notes or "").splitlines() if l.startswith("- ") and l[2:].strip()]


_TODO_MARK_RE = re.compile(r"待辦(?:事項)?\s*(?:是|為|：|:)\s*")
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
                item = action if len(action) >= 2 else (_CLAUSE_SPLIT_RE.split(line)[0].strip() or line)
                return item[:20], mo.group(1), mt.group(1)
    mo, mt = _OWNER_RE.search(line), _TIME_RE.search(line)
    if not (mo and mt):
        return None
    return (_CLAUSE_SPLIT_RE.split(line)[0].strip()[:20] or line[:20]), mo.group(1), mt.group(1)


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
        rows.append(f"| {item} | {owner} | {when.replace(' ', '')} |")
        if len(rows) >= max_rows:
            break
    return rows


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
    out = ["## 摘要"] + [f"- {l}" for _, _, l in picked] if picked else ["## 摘要", "- 無"]
    out += ["", "## 待辦", "| 事項 | 負責人 | 時間 |", "|---|---|---|"] + (todos or ["| 無 | - | - |"])
    out += ["", "## 決議"] + ([f"- {l}" for l in decisions] or ["- 無"])
    return "\n".join(out)



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
