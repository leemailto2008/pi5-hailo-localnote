import scoring


def test_cer():
    assert scoring.cer("今天天氣很好", "今天天氣很好") == 0
    assert scoring.cer("今天天氣很好", "今天天汽很好") == round(1 / 6, 4)
    assert scoring.cer("好，我們開始。", "好我們開始") == 0          # 標點不計


def test_fact_hits_aliases():
    facts = {"降價": ["8%", "百分之八"], "數量": ["200台", "兩百台"], "日期": ["11月15", "十一月十五"]}
    assert all(scoring.fact_hits("希望降價百分之8...不對是8％，拿到200臺，11月15日前", facts).values())
    assert scoring.recall(scoring.fact_hits("沒有任何事實", facts)) == 0


def test_summary_body_strips_transcript():
    md = "# 會議紀要\n## 摘要\n- a\n## 逐字稿\n全文全文"
    assert "全文" not in scoring.summary_body(md) and "## 摘要" in scoring.summary_body(md)


def test_summary_checks():
    fx = {"topic_keywords": ["訂單", "預算", "韌體"], "recap_facts": {"T1-負責人": ["王經理"], "T2-預算": ["120萬"]}}
    ok = "## 摘要\n- 訂單：王經理負責\n- 預算 120萬\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 回覆 | 王經理 | 週一 |\n## 決議\n- 無"
    c = scoring.summary_checks(ok, fx)
    assert c["sections_ok"] and not c["placeholder_echo"] and c["duplicate_lines"] == 0
    assert c["recap_recall"] == 1.0 and c["topic_coverage"] == round(2 / 3, 4)
    bad = "## 摘要\n1. （核心重點 1，每條不超過 30 字）\n- x\n- x"
    b = scoring.summary_checks(bad, fx)
    assert b["placeholder_echo"] and not b["sections_ok"] and b["duplicate_lines"] == 1


def _result(**over):
    r = {"state": "done", "audio_s": 180, "summary_latency_s": 60, "stt_npu_chunks": 20, "stt_cpu_chunks": 0, "hailo_busy_errors": 0,
         "metrics": {"stt_tail_wait": 1.0}, "errors": [],
         "transcript_eval": {"cer": 0.2, "fact_recall": 0.9},
         "summary_eval": {"summary_chars": 300, "sections_ok": True, "placeholder_echo": False, "duplicate_lines": 0,
                          "recap_recall": 0.8, "topic_coverage": 0.8},
         "resources": {"rec_cpu_avg": 3, "cpu_temp_max": 50, "ram_peak_mb": 1700, "swap_max_mb": 49, "swap_growth_mb": 0,
                       "throttled_start": "throttled=0x0", "throttled_end": "throttled=0x0"}}
    r.update(over)
    return r


TH = {"default": {"transcript_cer_max": 0.4, "transcript_fact_recall_min": 0.6, "summary_required": True, "summary_duplicate_lines_max": 0,
                  "summary_recap_recall_min": 0.5, "summary_topic_coverage_min": 0.5, "summary_latency_base_s": 120,
                  "summary_latency_per_min_s": 12, "stt_npu_ratio_min": 0.95, "hailo_busy_errors_max": 0, "stt_tail_wait_max_s": 10,
                  "rec_cpu_avg_max": 15, "cpu_temp_max": 75, "ram_peak_mb_max": 3500, "swap_growth_mb_max": 64},
      "per_minutes": {"30": {"stt_tail_wait_max_s": 15}}}


def test_evaluate_pass_and_fail():
    th = scoring.thresholds_for(TH, 3)
    assert all(c["passed"] for c in scoring.evaluate(_result(), th))
    failed = {c["check"] for c in scoring.evaluate(_result(transcript_eval={"cer": 0.6, "fact_recall": 0.9}, stt_cpu_chunks=5,
                                                               summary_latency_s=None), th) if not c["passed"]}
    assert failed == {"transcript_cer", "stt_npu_ratio", "summary_latency_s"}


def test_thresholds_override():
    assert scoring.thresholds_for(TH, 30)["stt_tail_wait_max_s"] == 15
    assert scoring.thresholds_for(TH, 5)["stt_tail_wait_max_s"] == 10
