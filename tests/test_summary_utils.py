import summary_utils  # noqa: E402
from summary_utils import (SECTION_MISSING_NOTE, assemble_long_minutes, clean_summary_markdown, compact_notes, ensure_sections,
                           extract_todos, keep_alive_plan, split_transcript, split_with_recap, summary_limits)


def test_removes_duplicate_rows_and_caps_bullets():
    md = ("## 摘要\n- a\n- b\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| x | y | z |\n| x | y | z |\n\n"
          "## 決議\n- d1\n- d2\n- d1\n- d3\n- d4\n- d5")
    out = clean_summary_markdown(md)
    assert out.count("| x | y | z |") == 1
    assert out.split("## 決議")[1].count("- d") == 3
    assert "|---|---|---|" in out


def test_repetition_loop_collapsed():
    md = "## 摘要\n- 重點一\n\n## 決議\n" + "\n".join(["- 代辦事項：王經理回覆客戶"] * 12)
    assert clean_summary_markdown(md).count("王經理回覆客戶") == 1


def test_empty_input():
    assert clean_summary_markdown("") == ""
    assert clean_summary_markdown(None) == ""


def _live_style(n_sent=120):
    # 與 app 即時逐字稿相同：句子以「，」串接、沒有換行
    return "，".join(f"第{i}句討論出貨數量為{i * 37}台並在十一月{i % 28 + 1}日前完成" for i in range(n_sent))


def test_split_transcript_respects_limit_and_keeps_text():
    text = _live_style()
    chunks = split_transcript(text, 1000)
    assert len(text) > 3000 and len(chunks) >= 4
    assert all(len(c) <= 1100 for c in chunks)          # 尾段併入時容許 +10%
    assert "".join(chunks) == text                       # 不增不減
    assert all(c.endswith("成") or c.endswith("，") or c == chunks[-1] for c in chunks)  # 切在子句邊界


def test_split_transcript_short_and_empty():
    assert split_transcript("  短短一句。 ", 1000) == ["短短一句。"]
    assert split_transcript("", 1000) == []


def test_split_transcript_hard_cuts_run_on_text():
    text = "無標點" * 900                                   # 2700 字、沒有任何標點
    chunks = split_transcript(text, 1000)
    assert [len(c) for c in chunks] == [1000, 1000, 700]
    assert "".join(chunks) == text


def test_split_transcript_prefers_sentence_end():
    a, b = "甲" * 600 + "。", "乙" * 600 + "。"
    assert split_transcript(a + b, 1000) == [a, b]


def test_split_transcript_balanced_no_tiny_tail():
    text = _live_style(160)
    chunks = split_transcript(text, 1000)
    assert len(chunks) == -(-len(text) // 1000)          # 不多切
    assert min(len(c) for c in chunks) > 0.6 * len(text) / len(chunks)


def test_split_transcript_merges_tiny_tail():
    text = "。".join(["測" * 95] * 63) + "。"              # 6048 字：6 段滿 + 很短的尾段
    chunks = split_transcript(text, 1000)
    assert "".join(chunks) == text
    assert min(len(c) for c in chunks) >= 150 and max(len(c) for c in chunks) <= 1100


def test_compact_notes_caps_lines_and_length():
    raw = ("1. 教育訓練9月2日簡報技巧課程，名額30人，已報名22人，還剩8名，請各部門在週四前交給吳副理\n"
           "- 公安加裝8支警示燈\n- 公安加裝8支警示燈\n- �\n- 預算\n* 第三條重點內容足夠長\n- 第四條重點內容足夠長\n- 第五條不該出現")
    out = compact_notes(raw).splitlines()
    assert len(out) == 4 and all(l.startswith("- ") and len(l) <= 42 for l in out)
    assert out[0].endswith("已報名22人") or out[0].endswith("還剩8名")          # 切在標點
    assert out[1] == "- 公安加裝8支警示燈"                                        # 去重、去亂碼與過短殘句
    assert "第五條" not in "\n".join(out)


def test_ensure_sections_marks_truncation_without_inventing():
    md = "## 摘要\n- a\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 供應商交期 | 黃專員 | 週五 |\n| 供應商"
    out = ensure_sections(md)
    assert "| 供應商\n" not in out + "\n" and out.count("| 供應商交期 |") == 1
    assert out.endswith("## 決議\n- " + SECTION_MISSING_NOTE)
    full = "## 摘要\n- a\n\n## 待辦\n| 無 | - | - |\n\n## 決議\n- 無"
    assert ensure_sections(full) == full


def test_extract_todos():
    rows = extract_todos(["公安加裝8支警示燈，12月17日前完成，負責人是邱經理", "預算表請於4月13日前送至財務部",
                          "華新電源晶片交期延到4.23，楊工程師週四回報", "公安加裝8支警示燈，再確認"])
    assert rows == ["| 公安加裝8支警示燈 | 邱經理 | 12月17日 |", "| 華新電源晶片交期延到4.23 | 楊工程師 | 4.23 |"]


def test_assemble_long_minutes_is_grounded_and_chronological():
    chunks = ["- 甲議題一，王經理週三回報\n- 甲議題二", "- 乙議題一，決定維持原價\n- 乙議題二", "- 丙總結一\n- 丙總結二"]
    md = assemble_long_minutes(chunks, max_summary=5, tail_lines=2)
    summ = md.split("## 待辦")[0].splitlines()[1:]
    # 最後一段 (結尾總結) 保留 2 條，其餘名額依時間順序輪流取
    assert [l for l in summ if l] == ["- 甲議題一，王經理週三回報", "- 甲議題二", "- 乙議題一，決定維持原價", "- 丙總結一", "- 丙總結二"]
    assert "| 甲議題一 | 王經理 | 週三 |" in md
    assert md.endswith("## 決議\n- 乙議題一，決定維持原價")
    assert all(sec in md for sec in ("## 摘要", "## 待辦", "## 決議"))
    empty = assemble_long_minutes([""])
    assert "| 無 | - | - |" in empty and empty.endswith("## 決議\n- 無")


def test_extract_todos_ignores_quantities_and_number_commas():
    rows = extract_todos(["明年預算350萬，需增加三位工程師，預算表2.25前送財務部",
                          "庫存1,200臺，下一批3月20日到貨，請林主任調貨",
                          "交期延至4.23，需楊工程師確認"])
    assert rows == ["| 庫存1,200臺 | 林主任 | 3月20日 |", "| 交期延至4.23 | 楊工程師 | 4.23 |"]


def test_compact_notes_drops_meta_lines_and_todo_time_formats():
    assert compact_notes("- 會議日期：7.25\n- 出席：全體同仁\n- 工安：加裝8支警示燈，8-15日前完成") == "- 工安：加裝8支警示燈，8-15日前完成"
    assert extract_todos(["工安：加裝8支警示燈，負責人是許經理，8-15日前完成"]) == ["| 工安：加裝8支警示燈 | 許經理 | 8-15日 |"]


def test_assemble_prefers_topic_diversity():
    chunks = ["- 教育訓練：報名22人\n- 預算：80萬", "- 教育訓練：報名28人\n- 客戶訂單：200台", "- 總結：工安加裝警示燈"]
    md = assemble_long_minutes(chunks, max_summary=4, tail_lines=1)
    summ = [l for l in md.split("## 待辦")[0].splitlines()[1:] if l]
    assert summ == ["- 教育訓練：報名22人", "- 預算：80萬", "- 客戶訂單：200台", "- 總結：工安加裝警示燈"]


def test_split_with_recap_isolates_closing_summary():
    body = _live_style(80)
    recap = "好，最後總結一下，展覽方面決議是參加台北展覽，待辦事項是李協理週二前交文宣初稿，今天就到這裡"
    chunks = split_with_recap(body + "，" + recap, 1000)
    assert chunks[-1] == recap[2:] and all(len(c) <= 1100 for c in chunks)
    assert "".join(chunks).replace("，好，", "") .endswith("今天就到這裡")
    no_recap = split_with_recap(body, 1000)
    assert no_recap == split_transcript(body, 1000)
    early = split_with_recap("總結一下前面" + body, 1000)        # 開頭的「總結」不算
    assert early == split_transcript("總結一下前面" + body, 1000)


def test_keep_alive_plan_windows():
    assert keep_alive_plan(1) == [None]
    assert keep_alive_plan(3) == [345, 270, None]                       # 120 + 75 × 剩餘段數，最後一段卸載
    p = keep_alive_plan(55)                                            # 4 小時會議
    assert len(p) == 55 and p.count(None) == 3 and p[-1] is None      # 3 個載入視窗
    assert all(ka is None or ka <= 1800 for ka in p)
    # 每個視窗內，從載入那段起算的預算都涵蓋到該視窗的最後一段
    starts = [0] + [i + 1 for i, ka in enumerate(p) if ka is None][:-1]
    for s0 in starts:
        n = next(i for i in range(s0, 55) if p[i] is None) - s0 + 1
        assert n == 1 or p[s0] >= 120 + 75 * n - 1


def test_summary_limits_scale_with_length():
    assert summary_limits(7) == {"max_summary": 8, "max_todos": 6, "max_decisions": 3}
    big = summary_limits(55)
    assert big == {"max_summary": 15, "max_todos": 11, "max_decisions": 6}
    assert summary_limits(500)["max_summary"] == 16


class _FakeCC:
    def convert(self, text):
        return text.translate(str.maketrans({"应": "應", "许": "許", "条": "條"}))


def test_simplified_chars_in_summary_become_traditional(monkeypatch):
    monkeypatch.setattr(summary_utils, "_CC", _FakeCC())
    md = "## 摘要\n- 供应商交期延后\n- 供應商交期延后\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 主持面試 | 许經理 | 本週 |"
    out = clean_summary_markdown(md)
    assert "应" not in out and "许" not in out and "許經理" in out
    assert out.count("供應商交期") == 1          # 轉換後才去重，簡繁兩行合併成一行


def test_long_minutes_notes_become_traditional(monkeypatch):
    monkeypatch.setattr(summary_utils, "_CC", _FakeCC())
    out = assemble_long_minutes(["- 招募：许經理主持面試\n- 決議：供应商下週回覆"])
    assert "許經理" in out and "供應商" in out and "应" not in out


def test_without_opencc_text_is_unchanged(monkeypatch):
    monkeypatch.setattr(summary_utils, "_CC", False)
    assert summary_utils.to_taiwan_traditional("供应商") == "供应商"
    assert summary_utils.to_taiwan_traditional("") == "" and summary_utils.to_taiwan_traditional(None) == ""


def test_missing_table_separator_is_restored():
    md = "## 待辦\n| 事項 | 負責人 | 時間 |\n| 回覆客戶 | 王經理 | 週五 |\n\n## 決議\n- 無"
    out = clean_summary_markdown(md)
    assert "| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 回覆客戶 | 王經理 | 週五 |" in out
    ok = "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 回覆客戶 | 王經理 | 週五 |"
    assert clean_summary_markdown(ok) == ok          # 已有分隔線就不動
    assert clean_summary_markdown("## 待辦\n| 事項 | 負責人 | 時間 |") == "## 待辦\n| 事項 | 負責人 | 時間 |"   # 只有表頭不補

