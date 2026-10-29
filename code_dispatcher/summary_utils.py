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


def clean_summary_markdown(md: str, max_items: Optional[Dict[str, int]] = None) -> str:
    """LLM 紀要後處理：轉台灣繁體、移除重複行 (小模型偶發重複迴圈直到 num_predict 用盡)，並限制摘要/決議點數。"""
    limits = max_items or {"摘要": 5, "決議": 3}
    out, seen, section, count = [], set(), None, 0
    for line in to_taiwan_traditional(md or "").splitlines():
        s = line.strip()
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
    return re.sub(r"\n{3,}", "\n\n", "\n".join(_fix_table_separators(out))).strip()



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


def ensure_sections(md: str) -> str:
    """紀要被 num_predict 截斷時，補上缺少的段落標題並明確標示未產生 (不捏造內容)；
    最後一列若是被截斷的半行表格也一併移除。"""
    lines = (md or "").rstrip().splitlines()
    if lines and lines[-1].lstrip().startswith("|") and not lines[-1].rstrip().endswith("|"):
        lines.pop()
    if lines and lines[-1].strip() in ("##", "#"):
        lines.pop()
    text = "\n".join(lines).rstrip()
    for sec in ("## 摘要", "## 待辦", "## 決議"):
        if sec not in text:
            text += f"\n\n{sec}\n- {SECTION_MISSING_NOTE}"
    return text.strip()


# 長會議紀要組裝 (2026-10-27)：qwen3:1.7b 無法跨 20+ 條重點做歸納 (實測會照抄、待辦表留空、決議擠掉)，
# 因此 map 之後不再交給 LLM reduce，而是以程式從各段重點組出三段式紀要：內容全部來自 LLM 的分段重點，不捏造。
# 姓名部分排除數量詞與動詞 (避免「增加三位工程師」「需楊工程師」被當成負責人)
_NAME_CHAR = r"(?:(?![需請加由交給與和的位名個人一二三四五六七八九十兩])[\u4e00-\u9fff])"
_OWNER_RE = re.compile(r"(?:負責人(?:是|為)?\s*)?(" + _NAME_CHAR + r"{1,3}(?:經理|副理|協理|主任|專員|工程師|公程師|處長|課長|組長|主管|老師))")
_TIME_RE = re.compile(r"(\d{1,2}\s*月\s*\d{1,2}\s*日|\d{1,2}[./-]\d{1,2}(?:日)?|\d{1,2}\s*日前|下?週[一二三四五六日]|月底|下週|明天|今天|年底)")
_DECISION_RE = re.compile(r"決定|決議|定案|確定|通過|同意|維持|改為|改成|暫停|取消")


def _note_lines(notes: str) -> List[str]:
    return [l[2:].strip() for l in (notes or "").splitlines() if l.startswith("- ") and l[2:].strip()]


def extract_todos(lines: List[str], max_rows: int = 6) -> List[str]:
    """同時出現負責人與期限的重點 → 待辦表格列「| 事項 | 負責人 | 時間 |」。"""
    rows, seen = [], set()
    for line in lines:
        mo, mt = _OWNER_RE.search(line), _TIME_RE.search(line)
        if not (mo and mt):
            continue
        item = re.split(r"[，；;]|,(?!\d)", line)[0].strip()[:20] or line[:20]   # 不切「1,200」這類數字
        if item in seen:
            continue
        seen.add(item)
        rows.append(f"| {item} | {mo.group(1)} | {mt.group(1).replace(' ', '')} |")
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
    per = [_note_lines(to_taiwan_traditional(n)) for n in chunk_notes]
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
    all_lines = [l for p in per for l in p]
    todos = extract_todos(all_lines, max_rows=max_todos)
    decisions = [l for l in all_lines if _DECISION_RE.search(l)][:max_decisions]
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
