"""LocalNote e2e 評分函式 (純函式，無硬體相依，可在 CI 離線測試)。"""
import re
import unicodedata

try:
    import opencc
    _CC = opencc.OpenCC("s2twp")
except Exception:  # CI 環境可能沒有 OpenCC
    _CC = None

PLACEHOLDERS = ("（核心重點", "若無具體待辦", "只寫真正定案", "不要重複規則", "每點 40 字以內", "依議題分點")
SECTIONS = ("## 摘要", "## 待辦", "## 決議")


def to_tw(text: str) -> str:
    if _CC and text:
        try:
            return _CC.convert(text)
        except Exception:
            pass
    return text or ""


def norm(text: str) -> str:
    """CER 用：轉台灣繁體，只留文字與數字。"""
    return "".join(c for c in to_tw(text) if unicodedata.category(c)[0] in ("L", "N"))


def _fact_norm(text: str) -> str:
    t = to_tw(text or "")
    return t.replace(" ", "").replace("　", "").replace("臺", "台").replace("％", "%").replace("，", ",")


def levenshtein(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except Exception:
        pass
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    return round(levenshtein(r, h) / max(1, len(r)), 4)


def fact_hits(text: str, facts: dict) -> dict:
    t = _fact_norm(text)
    return {k: any(_fact_norm(v) in t for v in vs) for k, vs in facts.items()}


def recall(hits: dict) -> float:
    return round(sum(hits.values()) / len(hits), 4) if hits else 1.0


def summary_body(markdown: str) -> str:
    """去掉 app 匯出 markdown 中附帶的「## 逐字稿」段，只留 LLM 紀要本身。"""
    md = markdown or ""
    i = md.find("## 逐字稿")
    return md[:i] if i >= 0 else md


def summary_checks(summary: str, fixture: dict) -> dict:
    s = summary or ""
    kws = fixture.get("topic_keywords") or []
    kw_hit = [k for k in kws if k in s]
    denom = min(len(kws), 5) or 1
    lines = [l.strip() for l in s.splitlines() if l.strip() and not l.strip().startswith(("## ", "|---", "| ---"))]
    rh = fact_hits(s, fixture.get("recap_facts") or {})
    return {
        "summary_chars": len(s),
        "sections_ok": all(x in s for x in SECTIONS),
        "placeholder_echo": any(p in s for p in PLACEHOLDERS),
        "duplicate_lines": len(lines) - len(set(lines)),
        "recap_recall": recall(rh),
        "recap_missing": [k for k, v in rh.items() if not v],
        "topic_coverage": round(min(1.0, len(kw_hit) / denom), 4),
        "topics_hit": kw_hit,
    }


def transcript_checks(transcript: str, fixture: dict) -> dict:
    ref = "".join(t for _, t in fixture["lines"])
    fh = fact_hits(transcript, fixture.get("facts") or {})
    return {
        "cer": cer(ref, transcript or ""),
        "ref_chars": len(norm(ref)),
        "hyp_chars": len(norm(transcript or "")),
        "fact_recall": recall(fh),
        "facts_missing": [k for k, v in fh.items() if not v],
    }


def evaluate(result: dict, th: dict) -> list:
    """回傳 [(check, value, limit, passed)]；result 為 e2e.run_case() 輸出。"""
    out = []

    def chk(name, value, limit, ok):
        out.append({"check": name, "value": value, "limit": limit, "passed": bool(ok)})

    t, s, r = result["transcript_eval"], result.get("summary_eval"), result.get("resources", {})
    chk("app_state_done", result.get("state"), "done", result.get("state") == "done")
    chk("transcript_cer", t["cer"], f"<= {th['transcript_cer_max']}", t["cer"] <= th["transcript_cer_max"])
    chk("transcript_fact_recall", t["fact_recall"], f">= {th['transcript_fact_recall_min']}", t["fact_recall"] >= th["transcript_fact_recall_min"])
    if th.get("summary_required", True):
        chk("summary_present", bool(s and s["summary_chars"] > 0), True, bool(s and s["summary_chars"] > 0))
        if s:
            chk("summary_sections", s["sections_ok"], True, s["sections_ok"])
            chk("summary_no_placeholder", not s["placeholder_echo"], True, not s["placeholder_echo"])
            chk("summary_no_duplicate_lines", s["duplicate_lines"], f"<= {th['summary_duplicate_lines_max']}", s["duplicate_lines"] <= th["summary_duplicate_lines_max"])
            chk("summary_recap_recall", s["recap_recall"], f">= {th['summary_recap_recall_min']}", s["recap_recall"] >= th["summary_recap_recall_min"])
            chk("summary_topic_coverage", s["topic_coverage"], f">= {th['summary_topic_coverage_min']}", s["topic_coverage"] >= th["summary_topic_coverage_min"])
        lat = result.get("summary_latency_s")
        lim = th["summary_latency_base_s"] + th["summary_latency_per_min_s"] * result.get("audio_s", 0) / 60.0
        chk("summary_latency_s", lat, f"<= {lim:.0f}", lat is not None and lat <= lim)
    n_npu, n_cpu = result.get("stt_npu_chunks", 0), result.get("stt_cpu_chunks", 0)
    ratio = round(n_npu / max(1, n_npu + n_cpu), 4)
    chk("stt_npu_ratio", ratio, f">= {th['stt_npu_ratio_min']}", ratio >= th["stt_npu_ratio_min"])
    chk("hailo_busy_errors", result.get("hailo_busy_errors", 0), f"<= {th['hailo_busy_errors_max']}", result.get("hailo_busy_errors", 0) <= th["hailo_busy_errors_max"])
    tail = (result.get("metrics") or {}).get("stt_tail_wait")
    chk("stt_tail_wait_s", tail, f"<= {th['stt_tail_wait_max_s']}", tail is not None and tail <= th["stt_tail_wait_max_s"])
    rec = r.get("rec_cpu_avg")
    chk("rec_cpu_avg_pct", rec, f"<= {th['rec_cpu_avg_max']}", rec is not None and rec <= th["rec_cpu_avg_max"])
    chk("cpu_temp_max_c", r.get("cpu_temp_max"), f"<= {th['cpu_temp_max']}", (r.get("cpu_temp_max") or 0) <= th["cpu_temp_max"])
    chk("ram_peak_mb", r.get("ram_peak_mb"), f"<= {th['ram_peak_mb_max']}", (r.get("ram_peak_mb") or 0) <= th["ram_peak_mb_max"])
    chk("swap_growth_mb", r.get("swap_growth_mb"), f"<= {th['swap_growth_mb_max']}", (r.get("swap_growth_mb") or 0) <= th["swap_growth_mb_max"])
    thr_ok = r.get("throttled_start") in (None, "throttled=0x0") and r.get("throttled_end") in (None, "throttled=0x0")
    chk("not_throttled", f"{r.get('throttled_start')}/{r.get('throttled_end')}", "throttled=0x0", thr_ok)
    chk("no_app_errors", len(result.get("errors") or []), "== 0", not result.get("errors"))
    return out


def thresholds_for(cfg: dict, minutes: float) -> dict:
    th = dict(cfg["default"])
    th.update((cfg.get("per_minutes") or {}).get(f"{minutes:g}", {}))
    return th
