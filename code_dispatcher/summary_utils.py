# code_dispatcher/summary_utils.py
"""會議紀要 (LLM 輸出) 後處理，純函式，供 meeting_minutes 使用並可離線單元測試。"""
import re
from typing import Dict, Optional


def clean_summary_markdown(md: str, max_items: Optional[Dict[str, int]] = None) -> str:
    """LLM 紀要後處理：移除重複行 (小模型偶發重複迴圈直到 num_predict 用盡)，並限制摘要/決議點數。"""
    limits = max_items or {"摘要": 5, "決議": 3}
    out, seen, section, count = [], set(), None, 0
    for line in (md or "").splitlines():
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
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
