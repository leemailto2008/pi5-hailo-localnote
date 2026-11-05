import summary_utils  # noqa: E402
from summary_utils import (SECTION_MISSING_NOTE, assemble_long_minutes, clean_summary_markdown, compact_notes, ensure_sections, ground_todo_table,
                           fill_decisions_from_transcript, is_recap_chunk, recap_notes,
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
    # 最後一段 (結尾總結) 保留 2 條，其餘名額依時間順序輪流取；決議條目只放在「決議」，不重複列在摘要
    assert [l for l in summ if l] == ["- 甲議題一，王經理週三回報", "- 甲議題二", "- 乙議題二", "- 丙總結一", "- 丙總結二"]
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


def test_summary_table_becomes_bullets_and_empty_rows_are_dropped():
    md = clean_summary_markdown("## 摘要\n| 議題 | 關鍵數字 | 日期 | 金額 | 人名 |\n|---|---|---|---|---|\n"
                                "| 1. 供應商交期延後 | 10月10日 | 10月10日 | - | 李協理 |\n| 預算 | 200萬元 | - | - | - |\n\n"
                                "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 送預算表 | 劉課長 | 10月7日 |\n| | | |\n| | | |\n\n## 決議\n- 無")
    assert "## 摘要\n- 供應商交期延後：10月10日，李協理\n- 預算：200萬元\n" in md
    assert "| | |" not in md and "| 送預算表 | 劉課長 | 10月7日 |" in md and "|---|---|---|" in md
    rows = "".join(f"| {i}. 議題{i} | {i}0臺 | 3.28L | - | - |\n" for i in range(1, 8))
    md7 = clean_summary_markdown("## 摘要\n| 議題 | 關鍵數字 | 日期 | 金額 | 人名 |\n|---|---|---|---|---|\n" + rows)
    assert md7.count("\n- ") == 7 and "- 議題7：70臺，3.28L" in md7      # 表格轉條列不受 5 點限制、不去「3.28」的小數點
    dec = clean_summary_markdown("## 摘要\n- 甲\n\n## 決議\n|---|---|---|\n| 產線加開一站檢查 | - | - |\n| 11月28日前完成報到 | - | - |\n\n- 無")
    assert dec.endswith("## 決議\n- 產線加開一站檢查\n- 11月28日前完成報到")                 # 決議段的表格也改條列，多餘的「- 無」拿掉
    dup = clean_summary_markdown("## 摘要\n| 議題 | 數字 |\n|---|---|\n| 產線加開檢查 | - |\n| 產線加開檢查 | 無 |\n| 品質 | 18件 |")
    assert dup.count("- 產線加開檢查") == 1 and "- 品質：18件" in dup                       # 轉出來重複的條目只留一條
    stray = clean_summary_markdown("## 摘要\n- 甲\n\n## 決議\n|---|---|---|---|---|\n- 維持原價")
    assert stray.endswith("## 決議\n- 維持原價")


def test_ensure_sections_fixes_order_and_short_todo_rows():
    md = ensure_sections("## 摘要\n- 甲\n\n## 決議\n- 招募：8月26日前完成報到\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
                         "| 面試 | 劉課長 | 週一前 |\n| 出差報告 |")
    assert md == ("## 摘要\n- 甲\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 面試 | 劉課長 | 週一前 |\n"
                  "| 出差報告 | - | - |\n\n## 決議\n- 招募：8月26日前完成報到")
    same = "## 摘要\n- 甲\n\n## 待辦\n| 無 | - | - |\n\n## 決議\n- 無"
    assert ensure_sections(same) == same
    empty = ensure_sections("## 摘要\n- 甲\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n\n## 決議\n- 乙")
    assert "|---|---|---|\n| 無 | - | - |\n\n## 決議" in empty                      # 只有表頭：補「無」
    one = ensure_sections("## 摘要\n- 甲\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 送預算表 | 劉課長 | 10月7日 |\n\n## 決議\n- 乙")
    assert "| 無 |" not in one
    cut = ensure_sections("## 摘要\n- 甲\n\n## 決議\n- 乙\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 送預算")
    assert cut.startswith(f"## 摘要\n- 甲\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| {SECTION_MISSING_NOTE} | - | - |\n\n## 決議\n- 乙")   # 半行先丟、再排序；截斷造成的空表標示未產生
    miss = ensure_sections("## 摘要\n- 甲\n\n## 決議\n- 乙")
    assert miss == f"## 摘要\n- 甲\n\n## 待辦\n- {SECTION_MISSING_NOTE}\n\n## 決議\n- 乙"      # 補上的缺段排回中間


def test_take_stable_chunk_and_live_tidy():
    text = "，".join(f"第{i}句話說了一些內容" for i in range(400))      # 約 4400 字，只有逗號
    assert summary_utils.take_stable_chunk(text, 0, 1000, 1000) is not None
    end = summary_utils.take_stable_chunk(text, 0, 1000, 1000)
    assert 600 <= end <= 1000 and text[end - 1] == "，"                 # 切在子句邊界
    assert summary_utils.take_stable_chunk(text, len(text) - 1500, 1000, 1000) is None   # 後面不足 lag：先不切
    assert summary_utils.take_stable_chunk("甲" * 2500, 0, 1000, 1000) == 1000          # 沒有標點：硬切
    live = "好，，我們開始\ufffd今天的會議。。 第一個議題"
    assert summary_utils.tidy_live_transcript(live) == "好，我們開始今天的會議 第一個議題"    # 與原本錄音結束後的整理完全相同 (連續標點留最後一個)
    # 前段整理結果與全文整理結果一致 (預先整理的前提)
    full = summary_utils.tidy_live_transcript(text + "，，最後總結一下")
    assert full.startswith(summary_utils.tidy_live_transcript(text)[:end])


def test_map_notes_complete():
    three = "- 教育訓練：報名22人\n- 預算：80萬元\n- 招募：下週面試\n"
    assert not summary_utils.map_notes_complete(three)
    assert not summary_utils.map_notes_complete(three + "- 客戶訂單：先簽")              # 第 4 條還沒寫完
    assert summary_utils.map_notes_complete(three + "- 客戶訂單：先簽50臺\n")
    assert summary_utils.map_notes_complete(three + "- 客戶訂單：" + "甲" * 45)          # 第 4 條已超過 40 字
    assert summary_utils.map_notes_complete("## 重點\n- 會議日期：7.25\n" + three + "- 品質：加開一站\n")   # 標題、日期行不算
    assert not summary_utils.map_notes_complete("")


def test_dates_in_minutes_lose_inner_spaces():
    md = clean_summary_markdown("## 摘要\n- 預算 6月14 日前送財務部，9 月 10 日完成\n- 版本 3.5 日誌")
    assert "6月14日前" in md and "9月10日完成" in md and "3.5 日誌" in md
    long_md = assemble_long_minutes(["- 公映模具交期延至 3月15 日，林主任負責"])
    assert "3月15日" in long_md and "| 3月15日 |" in long_md


def test_extract_todos_reads_the_spoken_todo_clause():
    rows = extract_todos(["是訂單調. 定單方面，決議是先簽50臺示單，待辦事項是林主任週三前回復華新機店",
                          "招募方面，決議是11月28日前完成報到，待辦事項是許經理主持面試",
                          "公安方面，決議是7月4日前完成複訓，待辦事項是蔡組長週二前回報名單"])
    assert rows == ["| 回復華新機店 | 林主任 | 週三 |",
                    "| 主持面試 | 許經理 | 11月28日 |",       # 這句沒講時間：用決議的期限
                    "| 回報名單 | 蔡組長 | 週二 |"]           # 不拿決議的 7月4日


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


def test_todo_owner_and_time_must_come_from_transcript():
    transcript = "下週由許經理主持面試，11月28日前完成；王經理週五回覆客戶。"
    md = ("## 摘要\n- 招募：工程師A負責\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| 主持面試 | 許經理 | 11月28日前 |\n| 撰寫報告 | 工程師A | 今日 |\n| 回覆客戶 | 王經理、陳經理 | 週五 |\n\n## 決議\n- 無")
    out = ground_todo_table(md, transcript)
    assert "| 主持面試 | 許經理 | 11月28日前 |" in out              # 逐字稿有的保留
    assert "| 撰寫報告 | - | - |" in out                          # 捏造的負責人與時間改成 -
    assert "| 回覆客戶 | 王經理 | 週五 |" in out                    # 多人時只留逐字稿有的
    assert "- 招募：工程師A負責" in out                               # 待辦以外的段落不動
    assert "| 事項 | 負責人 | 時間 |\n|---|---|---|" in out


def test_todo_grounding_keeps_placeholders_and_handles_empty_input():
    md = "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 無 | - | - |"
    assert ground_todo_table(md, "今天沒有待辦") == md
    assert ground_todo_table("", "x") == "" and ground_todo_table(md, "") == md


_CLOSING = "好最後總結一下，招募方面，決議是11月28日前完成報到，待辦事項是許經理主持面試，品質方面，決議是產線加開一站檢查待辦事項是黃專員"


def test_empty_decisions_filled_from_spoken_decisions():
    md = "## 摘要\n- 招募進度\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 面試 | 許經理 | - |\n\n## 決議\n- 無"
    out = fill_decisions_from_transcript(md, _CLOSING)
    assert out.endswith("## 決議\n- 11月28日前完成報到\n- 產線加開一站檢查")    # 到「，」或「待辦」為止
    assert out.split("## 決議")[0] == md.split("## 決議")[0]                      # 其他段落不動
    mid = "## 摘要\n- a\n\n## 決議\n\n## 待辦\n- b"
    assert fill_decisions_from_transcript(mid, _CLOSING) == "## 摘要\n- a\n\n## 決議\n- 11月28日前完成報到\n- 產線加開一站檢查\n\n## 待辦\n- b"
    missing = f"## 摘要\n- a\n\n## 決議\n- {SECTION_MISSING_NOTE}"
    assert "- 11月28日前完成報到" in fill_decisions_from_transcript(missing, _CLOSING)


def test_empty_decisions_heading_at_end_is_filled():
    md = "## 摘要\n- a\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| x | y | z |\n\n## 決議"
    assert fill_decisions_from_transcript(md, _CLOSING).endswith("## 決議\n- 11月28日前完成報到\n- 產線加開一站檢查")


def test_spoken_decisions_left_alone_when_not_needed():
    md = "## 摘要\n- 產線加開一站檢查\n\n## 決議\n- 無"
    assert fill_decisions_from_transcript("## 決議\n- 維持原價", _CLOSING) == "## 決議\n- 維持原價"   # 已有決議
    assert fill_decisions_from_transcript(md, "今天沒有做出決定") == md                          # 逐字稿沒明講
    assert fill_decisions_from_transcript(md, _CLOSING).endswith("## 決議\n- 11月28日前完成報到")  # 已在摘要的不重複
    assert fill_decisions_from_transcript(md, "決議是" + "很長" * 20 + "。") == md               # 30 字內沒有斷點不取
    assert fill_decisions_from_transcript("## 摘要\n- a", _CLOSING) == "## 摘要\n- a"             # 沒有決議段


def test_recap_notes_keep_owners_and_dates_verbatim():
    recap = ("最後總結一下，韌體方面，決議是修正版10月11日上線，待辦事項是李協理安排回歸測試，"
             "庫存方面，決議是下一批3000臺11月5日到貨，待辦事項是劉課長調貨，今天就到這裡，謝謝大家！")
    out = recap_notes(recap)
    lines = out.splitlines()
    assert all(l.startswith("- ") and len(l) <= 52 for l in lines)
    joined = "".join(lines)
    for fact in ("10月11日", "李協理", "3000臺", "11月5日", "劉課長"):
        assert fact in joined                               # 負責人與期限都留下
    assert "今天就到這裡" not in joined and "謝謝大家" not in joined and "總結一下" not in joined
    assert len(lines) <= 6 and recap_notes("") == ""
    assert lines[0].startswith("- 韌體方面") and lines[1].startswith("- 庫存方面")   # 一行一個議題
    assert recap_notes("最後總結一下任體方面，修正版上線").startswith("- 任體方面")       # 黏在前面的總結用語去掉
    glued = recap_notes("最後總結一下，展覽方面，決議是參加新竹展覽，待辦事項是吳副理週四前交文宣出稿今天就到這裡 謝謝")
    assert "吳副理週四前交文宣出稿" in glued and "今天就到" not in glued and "謝謝" not in glued   # 只去客套話，不丟待辦


def test_is_recap_chunk():
    assert is_recap_chunk("最後總結一下，招募方面…") and is_recap_chunk("  總結一下，預算方面…")
    assert not is_recap_chunk("第一個議題是招募，最後總結一下")


def test_long_minutes_never_repeat_a_decision_in_the_summary():
    chunks = ["- 預算：明年預算200萬元\n- 訂單：決議先簽60臺試單", "- 總結：決議先簽60臺試單"]
    md = assemble_long_minutes(chunks, max_summary=5, tail_lines=2)
    lines = [l for l in md.splitlines() if l.startswith("- ")]
    assert len(lines) == len(set(lines))                     # e2e 的 duplicate_lines = 0
    assert md.count("決議先簽60臺試單") == 1 and "- 預算：明年預算200萬元" in md     # 總結與前段的同一條決議只留一條
    only = assemble_long_minutes(["- 決議：維持原價"])             # 全部都是決議時，摘要仍保留
    assert only.split("## 待辦")[0].count("維持原價") == 1



def test_long_minutes_decision_drops_a_todo_clause_already_in_the_table():
    notes = ("- 供應商方面，決議是替代料持續驗證，待辦事項是黃專員週三回報宏遠科技的交期，決議是參加新竹展覽\n"
             "- 預算方面，決議是維持原案，待辦事項是盡快送出")
    md = assemble_long_minutes([notes])
    assert "| 回報宏遠科技的交期 | 黃專員 | 週三 |" in md
    decisions = md.split("## 決議")[1]
    # 已在待辦表的半句拿掉，後面黏著的另一個決議保留
    assert "- 供應商方面，決議是替代料持續驗證，決議是參加新竹展覽" in decisions
    # 待辦表沒收到 (認不出負責人) 的照舊整句保留
    assert "- 預算方面，決議是維持原案，待辦事項是盡快送出" in decisions


def test_ensure_sections_merges_a_repeated_heading_and_builds_a_missing_todo_table():
    md = ("## 摘要\n- 出差安排：東京拜訪，6月15日報告\n\n"
          "## 決議\n- 決議：8月26日前完成報到\n- 待辦：劉課長主持面試，週一前回報\n\n"
          "## 決議\n- 出差新加坡待辦事項是李協理，週會報告，今天就到這裡，謝謝大家!")
    out = ensure_sections(md)
    assert out.count("## 決議") == 1 and SECTION_MISSING_NOTE not in out
    assert "| 主持面試 | 劉課長 | 週一 |" in out
    assert "- 待辦：劉課長" not in out                      # 已移進待辦表
    assert out.index("## 摘要") < out.index("## 待辦") < out.index("## 決議")
    assert "今天 |" not in out                              # 「今天就到這裡」不是期限


def test_ensure_sections_still_marks_a_missing_todo_without_owner_and_time():
    out = ensure_sections("## 摘要\n- 預算維持原案\n\n## 決議\n- 維持原案")
    assert f"## 待辦\n- {SECTION_MISSING_NOTE}" in out


def test_short_minutes_get_the_closing_recap_todos_and_decisions():
    transcript = ("第一個議題是出差，吳副理下個月要去東京，行程四天，預算抓十二萬元，機票已經訂好了。" * 4 +
                  "第二個議題是招募，業務部三個職缺，"
                  "希望在8月26日前完成報到。第三個議題是出差，李協理下個月要去新加坡。"
                  "好，最後總結一下，招募方面，決議是8月26日前完成報到，待辦事項是劉課長週一前主持面試，"
                  "出差方面，決議是李協理出差新加坡，待辦事項是李協理在5月14日的週會報告，今天就到這裡，謝謝大家")
    md = ("## 摘要\n- 出差：吳副理下個月去東京，行程四天\n\n"
          "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 無 | - | - |\n\n"
          "## 決議\n- 招募：8月26日前完成報到")
    out = summary_utils.merge_recap_into_minutes(md, transcript)
    todo = out.split("## 待辦")[1].split("## 決議")[0]
    assert "| 劉課長 |" in todo and "| 李協理 |" in todo and "| 無 |" not in todo
    decisions = out.split("## 決議")[1]
    assert decisions.count("8月26日") == 1                      # 已寫過的決議 (同一個日期) 不重複
    assert "新加坡" in decisions
    lines = [l for l in out.splitlines() if l.startswith("- ")]
    assert len(lines) == len(set(lines))
    # 沒有結尾總結的逐字稿：原樣不動
    assert summary_utils.merge_recap_into_minutes(md, "第一個議題是出差，吳副理下個月要去東京。") == md


def test_todo_clause_without_shi_after_shixiang():
    # STT 漏了「是」：「…開課待辦事項十個部門週四前把名單交給吳副理」
    assert summary_utils._todo_parts("決議是簡報技巧課程9月2日開課待辦事項十個部門週四前把名單交給吳副理") == \
        ("十個部門前把名單交給", "吳副理", "週四")
    assert summary_utils._todo_parts("待辦很多，大家辛苦了") is None


def test_split_todo_table_is_joined_and_closing_filler_dropped():
    md = ("## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 送預算表 | 王經理 | 10月7日 |\n\n|---|---|---|\n"
          "| 主持面試 | 王經理 | 12月3日 |\n\n## 決議\n- 決議是王經理主持面試，今天就到這裡\n- 今天就到這裡，謝謝大家")
    out = clean_summary_markdown(md)
    assert out.count("|---|---|---|") == 1 and "10月7日 |\n| 主持面試" in out
    assert "- 決議是王經理主持面試" in out and "今天就到這裡" not in out and "謝謝大家" not in out


def test_todo_without_marker_takes_the_action_after_the_owner():
    assert summary_utils._todo_parts("依依商方面 決意施替代料持續驗證像是理協理週五回報星光材料的交期") == \
        ("回報星光材料的交期", "理協理", "週五")
    assert summary_utils._todo_parts("公安：8月15日前完成，許經理負責") == ("公安：8月15日前完成", "許經理", "8月15日")


def test_recap_drops_a_bare_topic_label():
    out = summary_utils.recap_notes("最後總結一下，招募方面，決議是週五前完成報到，依應商方面,，展覽方面，決議是參加新竹展覽，今天就到這裡")
    assert "依應商方面" not in out and "招募方面" in out and "展覽方面" in out


def test_english_minutes_keep_their_language():
    assert summary_utils.detect_language("John will send the revised quote by Friday, and Mary confirms capacity.") == "en"
    assert summary_utils.detect_language("新的 ERP 系統下週上線，test 跑完再切換") == "zh"
    # 中英夾雜的中文會議：英文字母比中文字多，但英文都是名詞／動詞，沒有 the、is、we 這類功能詞
    mixed = "我們這個 release 的 schedule 要 delay，因為 QA 那邊的 regression test 還沒 pass，下週再 review 一下 coverage。"
    assert summary_utils.detect_language(mixed) == "zh"
    # 英文會議裡夾幾個中文字 (人名、地名) 仍是英文
    assert summary_utils.detect_language("Okay, let's start. Mr. 王 will send the quote to the customer in 新竹 by Friday, "
                                         "and we agreed to keep the price.") == "en"
    md = ("## Summary\n- Budget: 2M next year\n\n## Action Items\n| Item | Owner | Due |\n|---|---|---|\n"
          "| Send quote | John | Friday |\n\n## Decisions\n- None")
    internal = summary_utils.to_internal_sections(md)
    assert "## 摘要" in internal and "| 事項 | 負責人 | 時間 |" in internal and "- 無" in internal
    shown = summary_utils.localize_minutes(ensure_sections(clean_summary_markdown(internal)), "en")
    assert shown.index("## Summary") < shown.index("## Action Items") < shown.index("## Decisions")
    assert "| Item | Owner | Due |" in shown and "- None" in shown and "摘要" not in shown
    assert summary_utils.localize_minutes(internal, "zh") == internal


def test_english_todos_and_decisions_in_long_minutes():
    assert summary_utils._todo_parts("Quote: John will send the revised quote by Friday") == ("send the revised quote", "John", "Friday")
    assert summary_utils._todo_parts("Owner: Mary Chen, due Nov 5, confirm sample capacity")[1:] == ("Mary Chen", "Nov 5")
    assert summary_utils._todo_parts("We will review it next week") is None
    md = summary_utils.localize_minutes(assemble_long_minutes(
        ["- Budget: decided to keep 2M for next year", "- Orders: John will send the quote by Friday"]), "en")
    assert "| send the quote | John | Friday |" in md and "- Budget: decided to keep 2M for next year" in md.split("## Decisions")[1]



def test_empty_owner_and_due_are_taken_from_the_item():
    md = ("## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| Reply to Lotus Tech by Wednesday | John | - |\n"
          "| Confirm firmware testing by Kevin | - | - |\n| 王經理主持面試 | - | - |\n| 送預算表 | 劉課長 | 10月7日 |")
    tr = "Kevin will arrange the testing. John will reply to Lotus Tech by Wednesday. 王經理主持面試，劉課長10月7日前送預算表"
    out = ground_todo_table(md, tr)
    assert "| Reply to Lotus Tech | John | Wednesday |" in out
    assert "| Confirm firmware testing | Kevin | - |" in out
    assert "| 王經理主持面試 | 王經理 | - |" in out and "| 送預算表 | 劉課長 | 10月7日 |" in out
    # 從事項取出的值也要在逐字稿裡找得到
    assert "| Call Mike | - | Monday |" in ground_todo_table(
        "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| Call Mike by Monday | - | - |", "Someone should call by Monday.")


def test_english_closing_recap_is_split_off_and_kept_verbatim():
    body = "The first topic is training. There is a course on September 2 with 30 seats. " * 12
    recap = ("Okay, to wrap up. Training: names go to Mary by Thursday. Firmware: we decided to release the fix on "
             "March 15th, and Kevin owns the testing. Customer order: we agreed on a 30-unit trial order, and John will "
             "reply to Lotus Tech by Wednesday. That's all for today, thank you everyone.")
    parts = summary_utils.split_with_recap(body + recap, 3200)
    assert summary_utils.is_recap_chunk(parts[-1])
    notes = summary_utils.recap_notes(parts[-1])
    assert notes.splitlines()[0] == "- Training: names go to Mary by Thursday" and "thank you" not in notes.lower()
    md = summary_utils.localize_minutes(assemble_long_minutes(["- Training: course on September 2", notes]), "en")
    assert "| owns the testing | Kevin | March 15th |" in md and "| reply to Lotus Tech | John | Wednesday |" in md
    assert "we decided to release the fix" in md.split("## Decisions")[1]
    # 「signed up」不是決議
    assert not summary_utils._DECISION_RE.search("22 people signed up, 8 seats left")



def test_short_english_minutes_get_the_recap_actions_and_decisions():
    tr = ("The first topic is the customer order. Lotus Tech wants 200 units by February 28. " * 6 +
          "Okay, to wrap up. Customer order: we agreed on a 30-unit trial order at the new price. And John will reply "
          "to Lotus Tech by Wednesday. That's all for today, thank you everyone.")
    md = ("## 摘要\n- Customer order: 200 units requested by Lotus Tech\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| 無 | - | - |\n\n## 決議\n- 無")
    out = summary_utils.localize_minutes(summary_utils.merge_recap_into_minutes(md, tr), "en")
    assert "| reply to Lotus Tech | John | Wednesday |" in out and "And John" not in out
    assert "we agreed on a 30-unit trial order" in out.split("## Decisions")[1]


def test_placeholder_bullet_dropped_when_the_section_has_items():
    md = summary_utils.to_internal_sections("## Summary\n- Budget 800k\n\n## Decisions\n- Firmware fix on March 15\n- None")
    out = clean_summary_markdown(md)
    assert "- 無" not in out and "- Firmware fix on March 15" in out
    assert "## 決議\n- 無" in clean_summary_markdown("## 摘要\n- 預算 80 萬\n\n## 決議\n- 無")      # 只有「無」時保留


def test_years_the_meeting_never_said_are_dropped():
    tr = "There is a course on September 2. Lotus Tech wants 200 units by February 28. The 2026 budget is 800k. 明年9月2日開課"
    md = ("- Training on September 2, 2023, 30 seats\n- Order by Feb 28th, 2024\n- Budget for 2026: 800k\n"
          "- Launch March 15 2026\n- 2023年9月2日開課，2026年3月1日上線")
    out = summary_utils.drop_unspoken_years(md, tr)
    assert "September 2, 30 seats" in out and "Feb 28th\n" in out
    assert "Budget for 2026" in out and "March 15 2026" in out          # 逐字稿有 2026：保留
    assert "9月2日開課，2026年3月1日上線" in out and "2023年" not in out


def test_recap_decision_glued_to_its_todo_is_not_added_twice():
    # 1d1eb6f 3 分鐘 e2e：STT 沒斷句，「決議是12日前完成報到待辦事項是王經理主持面」整句被當成新決議補進去
    transcript = ("第七個議題是招募，業務部目前二個職缺，下週會再安排四場面試，由王經理主持，希望在12月3日前完成報到。" * 3 +
                  "好，最後總結，招募方面 決議是12日前完成報到待辦事項是王經理主持面，今天就到這裡，謝謝大家")
    md = ("## 摘要\n- 招募：二個職缺，四場面試\n\n"
          "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 招募面試 | 王經理 | 12月3日前 |\n\n"
          "## 決議\n- 12月3日前完成報到\n- 王經理主持面試")
    out = summary_utils.merge_recap_into_minutes(md, transcript)
    assert "待辦事項是" not in out.split("## 決議")[1]


def test_english_todo_owner_filled_when_the_transcript_names_them_as_owner():
    tr = ("we decided to release the fix on March 15th, and Kevin owns the testing. Please send your names to Mary by "
          "Thursday. John will reply to Lotus Tech by Wednesday.")
    md = ("## Summary\n- Firmware release March 15\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| Confirm firmware testing with Kevin | - | March 15 |\n| Send training names to Mary | - | Thursday |\n"
          "| Reply to Lotus Tech | John | Wednesday |\n\n## 決議\n- Release on March 15")
    out = summary_utils.ground_todo_table(md, tr)
    assert "| Confirm firmware testing with Kevin | Kevin | March 15 |" in out
    assert "| Send training names to Mary | - | Thursday |" in out        # 收件人不是負責人
    assert "| Reply to Lotus Tech | John | Wednesday |" in out


def test_recap_todo_clause_without_deadline_is_dropped_when_already_in_the_table():
    # 90ddccb 3 分鐘 e2e：「預算方面，決議是…200萬元，待辦事項是劉課長送預算表」，待辦子句沒有期限，原本整句照抄進決議
    transcript = ("第四個議題是預算，明年平保部的預算是200萬元，預算表請劉課長在10月7日前送到財務部。" * 3 +
                  "好，最後總結，預算方面，決議是平保部明年預算200萬元，待辦事項是劉課長送預算表，今天就到這裡，謝謝大家")
    md = ("## 摘要\n- 預算200萬元\n\n"
          "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 劉課長送預算表 | 劉課長 | 10月7日前 |\n\n"
          "## 決議\n- 預算表10月7日前送至財務部")
    decisions = summary_utils.merge_recap_into_minutes(md, transcript).split("## 決議")[1]
    assert "平保部明年預算200萬元" in decisions and "待辦事項" not in decisions


def test_ownerless_duplicate_todo_row_is_dropped():
    md = ("## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 回報星光材料的交期 | 理協理 | 週五 |\n"
          "| 回報星光材料的交期 | 李協理 | 週五 |\n| 送預算表 | - | 10月7日前 |\n\n## 決議\n- 無")
    out = summary_utils.ground_todo_table(md, "請李協理週五回報星光材料的交期，預算表10月7日前送出")
    assert out.count("回報星光材料的交期") == 1 and "| 回報星光材料的交期 | 李協理 | 週五 |" in out
    assert "| 送預算表 | - | 10月7日前 |" in out          # 沒有重複的照舊保留


def test_table_cells_in_decision_bullets_are_stripped():
    md = ("## 摘要\n- 討論：天竹同學推薦加入團隊 | - | - |\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| 確認是否需要重新整理紀錄 | - | - |\n\n## 決議\n- 確認是否需要重新整理紀錄 | - | - |")
    out = summary_utils.clean_summary_markdown(md)
    assert "- 討論：天竹同學推薦加入團隊\n" in out + "\n"
    decisions = out.split("## 決議")[1]
    assert "|" not in decisions and "重新整理紀錄" not in decisions and "- 無" in decisions
    # 不是待辦表事項的決議：只拿掉表格格子
    md2 = md.replace("- 確認是否需要重新整理紀錄 | - | - |", "- 預算維持原案 | 王經理 | - |")
    assert "- 預算維持原案（王經理）" in summary_utils.clean_summary_markdown(md2).split("## 決議")[1]


def test_split_translation_sections():
    text = "【Summary】\n• 培訓課程9月2日\n• 工安複訓\n\n【Action Items】\n• 送名單給瑪莉 (週四)\n\n【Decisions】\n• 3月15日發布"
    secs = summary_utils.split_translation_sections(text)
    assert [h for h, _ in secs] == ["【Summary】", "【Action Items】", "【Decisions】"]
    assert secs[0][1] == "• 培訓課程9月2日\n• 工安複訓" and secs[2][1] == "• 3月15日發布"
    assert summary_utils.split_translation_sections("沒有標題的一段文字") == [(None, "沒有標題的一段文字")]


def test_today_is_a_deadline_only_when_used_as_one():
    got = summary_utils._todo_parts("招募方面，待辦事項是王經理主持面試今天這到")
    assert got is None or got[2] != "今天"
    got = summary_utils._todo_parts("待辦事項是王經理今天下班前回覆客戶")
    assert got and got[1] == "王經理" and got[2] == "今天"


def test_todo_marker_with_stray_punctuation():
    got = summary_utils._todo_parts("工安方面，決議是7月4日前完成全員複訓，待辦事項,是蔡組長,負責工安複訓")
    assert got and got[1] == "蔡組長" and got[2] == "7月4日" and "複訓" in got[0]


def test_recap_splits_a_topic_glued_to_the_previous_todo():
    # 11-03 夜間 30 分鐘喇叭實錄：「…回報宏宇科技的交期展覽方面,決議是參加新竹展覽」沒有斷句
    recap = ("最後總結，供應商方面,決議是替代料持續驗證，待辦事項是黃專員週三回報宏宇科技的交期展覽方面,決議是參加新竹展覽，"
             "待辦事項,是吳副理週五前交出文宣稿，今天就到這裡,謝謝")
    lines = summary_utils._note_lines(summary_utils.recap_notes(recap))
    assert lines[0].startswith("供應商方面") and lines[0].endswith("交期")
    assert lines[1].startswith("展覽方面，決議是參加新竹展覽")
    assert summary_utils._todo_parts(lines[0])[1:] == ("黃專員", "週三")
    assert summary_utils._split_topic_start("教育訓練方面，決議是開課") == ["教育訓練方面，決議是開課"]


def test_stt_residue_is_not_part_of_the_owner_name():
    # 11-03 夜間 10 分鐘實錄：「待辦事項是李協理安排」聽成「待會兒時向市李協理安排」
    assert summary_utils._OWNER_RE.search("待會兒時向市李協理安排").group(1) == "李協理"
    assert summary_utils._OWNER_RE.search("交給王小明經理").group(1) == "王小明經理"


def test_long_minutes_take_the_recap_decisions_first_and_valid_months():
    chunks = ["- 品質：決議加開一站檢查\n- 訂單：決議維持新報價\n- 庫存：決議下一批三千臺", "- 出差：決議東京行程四天",
              "- 工安方面，決議是7月4日前完成全員複訓，待辦事項是蔡組長負責\n- 招募方面，決議是3月19日前完成報到"]
    md = assemble_long_minutes(chunks, max_summary=6, max_decisions=3)
    dec = md.split("## 決議")[1]
    assert "7月4日前完成全員複訓" in dec and "3月19日前完成報到" in dec           # 總結裡的決議不會被前段擠掉
    assert "7月4日前完成全員複訓" not in md.split("## 待辦")[0]                   # 也就不會跑到摘要
    assert summary_utils._TIME_RE.findall("日1月13月19日完成報到，12月3日") == ["3月19日", "12月3日"]


def test_recap_line_splits_at_a_second_decision_without_topic():
    rec = "最後總結，公安方面，決議是七月四日，完成全員複訓，待辦事項，是蔡組長負責，決議是3月19日完成報到，待辦事項是林主任主持面試，今天就到這裡"
    lines = summary_utils._note_lines(summary_utils.recap_notes(rec))
    assert lines[0].startswith("公安方面") and "3月19日" not in lines[0]
    assert lines[1].startswith("決議是3月19日完成報到")


def test_recap_topic_statement_with_garbled_decision_word_goes_to_decisions():
    chunks = ["- 品質：上個月客訴18件\n- 預算：明年預算200萬元", "- 訓練方面爭一是簡報技巧9月2日開課，待辦事項十個部門週四前交名單"]
    md = assemble_long_minutes(chunks, max_summary=6, max_decisions=3)
    assert "簡報技巧9月2日開課" in md.split("## 決議")[1] and "簡報技巧" not in md.split("## 待辦")[0]


def test_empty_todo_section_gets_a_table_so_the_recap_todos_can_be_added():
    # 11-04 3 分鐘 e2e：模型寫了「## 待辦」就直接接「## 決議」，結尾總結的待辦因此補不進去，還黏在決議裡
    md = ensure_sections("## 摘要\n- 招募：二個職缺\n\n## 待辦\n\n## 決議\n- 12月3日前完成報到")
    assert "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 無 | - | - |" in md
    transcript = ("第七個議題是招募，業務部目前二個職缺，下週會再安排四場面試，由王經理主持，希望在12月3日前完成報到。" * 3 +
                  "好，最後總結，招募方面 決議是12日前完成報到待辦事項是王經理主持面試，今天就到這裡，謝謝大家")
    out = summary_utils.merge_recap_into_minutes(md, transcript)
    assert "| 王經理 |" in out.split("## 待辦")[1].split("## 決議")[0]
    assert ensure_sections("## 摘要\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 無 | - | - |\n\n## 決議").count("- 無") == 2



def test_transcript_todo_rows_reads_explicit_assignments():
    tr = ("第一個議題是供應商，新光材料的電源晶片交期延到10月10日，請李協理跟對方確認，週五回報，"
          "第三個議題是工安，上週三樓產線發生一起小事故，已經加裝四支警示燈，並在6月14日前完成全員複訓，負責人是張工程師，"
          "預算表請劉課長在10月7日前送到財務部，同意，這件事由王經理負責，週五前回覆客戶，"
          "好，最後總結一下，預算方面，待辦事項是劉課長10月7日前送預算表，今天就到這裡，謝謝大家")
    assert summary_utils.transcript_todo_rows(tr) == [
        "| 跟對方確認 | 李協理 | 週五 |",
        "| 完成全員複訓 | 張工程師 | 6月14日前 |",
        "| 送到財務部 | 劉課長 | 10月7日前 |",
        "| 回覆客戶 | 王經理 | 週五前 |",
    ]


def test_transcript_todo_rows_skips_past_dates_and_english():
    assert summary_utils.transcript_todo_rows("上週三樓產線發生事故，由王經理說明經過，大家都同意") == []
    assert summary_utils.transcript_todo_rows("Kevin will send the report by Friday, Susan owns the testing.") == []


def test_merge_body_todos_adds_missing_rows_and_fills_owner():
    md = ("## 摘要\n- 預算 200 萬元\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| 送交預算表 | 劉客 | 10月7日前 |\n| 回覆客戶 | 王經理 | 週五前 |\n\n## 決議\n- 先簽 60 臺試單\n")
    tr = ("預算表請劉課長在10月7日前送到財務部，這件事由王經理負責，週五前回覆客戶，"
          "已經加裝四支警示燈，並在6月14日前完成全員複訓，負責人是張工程師")
    out = summary_utils.merge_body_todos(md, tr)
    table = out.split("## 待辦")[1].split("## 決議")[0]
    assert "| 送交預算表 | 劉課長 | 10月7日前 |" in table        # 模型的負責人對不上逐字稿 → 用交辦那句的人
    assert table.count("王經理") == 1                          # 已經有的不重複
    assert "| 完成全員複訓 | 張工程師 | 6月14日前 |" in table
    assert out.endswith("## 決議\n- 先簽 60 臺試單\n")


def test_merge_body_todos_replaces_placeholder_row():
    md = "## 摘要\n- x\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 無 | - | - |\n\n## 決議\n- 無\n"
    out = summary_utils.merge_body_todos(md, "請黃專員在2月15日前提出改善報告")
    assert "| 無 |" not in out and "| 提出改善報告 | 黃專員 | 2月15日前 |" in out


def test_transcript_todo_rows_hand_over_keeps_whole_clause():
    tr = "名額20人，請各部門在週一前把名單交給李協理，第十個議題是上線，這件事交給吳副理，週三前回覆客戶"
    assert summary_utils.transcript_todo_rows(tr) == ["| 各部門把名單交給李協理 | 李協理 | 週一前 |",
                                                     "| 回覆客戶 | 吳副理 | 週三前 |"]


def test_clean_summary_drops_prompt_label_echo():
    md = ("## 摘要\n- 供應商星光材料電源晶片交期延至10月10日，關鍵數字：10月10日，金額：無，人名：曦代料\n"
          "- 新版ERP系統正式切換定於2月3日，預計時間：2月3日，負責人：李協理\n"
          "- 訂單先簽60臺試單，人名：王經理\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 無 | - | - |\n\n## 決議\n- 無")
    out = clean_summary_markdown(md)
    assert "- 供應商星光材料電源晶片交期延至10月10日\n" in out
    assert "- 新版ERP系統正式切換定於2月3日，負責人：李協理\n" in out
    assert "- 訂單先簽60臺試單，王經理\n" in out      # 是「姓＋職稱」且前面沒寫過：留名字，欄位名稱拿掉


def test_merge_recap_moves_known_todo_out_of_decisions():
    md = ("## 摘要\n- 招募業務部，面試15人\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| 招募業務部面試 | 王經理 | 12月3日前 |\n\n## 決議\n- 招募方面 決議是12日前完成報到待辦事項是王經理主持面\n")
    tr = ("第七個議題是招募，業務部目前2個職缺，下週會再安排四場面試，由王經理主持，" * 8 +
          "好 最後總結，出差方面，決議是李協理出差新加坡，待辦事項是李協理，週會報告，"
          "訂單方面，決議是先簽60臺試單，待辦事項是王經理週五前回覆客戶，今天就到這裡，謝謝大家")
    out = summary_utils.merge_recap_into_minutes(md, tr)
    table = out.split("## 待辦")[1].split("## 決議")[0]
    dec = out.split("## 決議")[1]
    assert "- 招募方面 決議是12日前完成報到\n" in dec                 # 同一件事已在表上：待辦半句拿掉
    assert "| 週會報告 | 李協理 | - |" in table                       # 沒有期限的總結待辦也進表
    assert "| 回覆客戶 | 王經理 | 週五 |" in table                     # 同一負責人的另一件事照樣加
    assert "待辦事項" not in dec


def test_drop_listed_todo_clause_removes_owner_only_clause_with_its_action():
    rows = ["| 事項 | 負責人 | 時間 |", "|---|---|---|", "| 訂單掉貨 | 劉課長 | 1月5日 |"]
    out = summary_utils._drop_listed_todo_clause(["存方面，決議是下一批3000臺1月5日到，待辦事項是劉課長，訂單掉貨"], rows)
    assert out == ["存方面，決議是下一批3000臺1月5日到"]


def test_past_weekday_is_not_a_deadline():
    assert summary_utils.extract_todos(["公安：上週三事故，許經理負責"]) == []
    assert summary_utils.extract_todos(["許經理週三前回報"]) == ["| 許經理週三前回報 | 許經理 | 週三 |"]


def test_clean_summary_drops_jinri_closing_filler():
    md = "## 摘要\n- 預算200萬元\n- 今日會議結束，謝謝\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 無 | - | - |\n\n## 決議\n- 招募8月26日前完成報到\n- 今日就到這裡"
    out = clean_summary_markdown(md)
    assert "今日" not in out and "- 招募8月26日前完成報到" in out and "- 預算200萬元" in out


def test_closing_filler_removed_from_todo_item_and_todo_only_decision_dropped():
    md = ("## 摘要\n- 招募面試15人\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| 王經理主持面試，今天就到這裡 | 王經理 | - |\n\n## 決議\n- 招募12月3日前完成報到")
    assert "| 王經理主持面試 | 王經理 | - |" in clean_summary_markdown(md)
    rows = ["| 事項 | 負責人 | 時間 |", "|---|---|---|", "| 主持面試 | 劉課長 | 8月26日 |"]
    known = lambda t: summary_utils._todo_known(t, [[x.strip() for x in r.strip("|").split("|")] for r in rows[2:]])
    assert summary_utils._drop_listed_todo_clause(["待辦：劉課長主持面試", "待辦：王經理回覆客戶"], rows, known=known) == \
        ["待辦：王經理回覆客戶"]


def test_assemble_long_minutes_prefers_recap_and_body_todos():
    notes = ["- 公安：上週二事故未受傷，許經理負責\n- 存貨：庫存1500臺，張工程師1月3日盤點",
             "- 品質異常：溫控模組客訴，劉課長9月15日報告",
             "- 訂單方面，決議是先簽五十臺試單，待辦事項是林主任週三前回覆華新"]
    body = ["| 完成全員複訓 | 許經理 | 8月15日前 |", "| 提出改善報告 | 劉課長 | 9月15日前 |",
            "| 回覆客戶 | 黃專員 | 週三前 |"]
    out = summary_utils.assemble_long_minutes(notes, max_todos=4, body_todos=body)
    table = out.split("## 待辦")[1].split("## 決議")[0]
    assert "| 回覆華新 | 林主任 | 週三 |" in table                  # 結尾總結的待辦一定在
    assert "| 完成全員複訓 | 許經理 | 8月15日前 |" in table and "| 提出改善報告 | 劉課長 | 9月15日前 |" in table
    assert table.count("劉課長") == 1                                 # 筆記裡同一件事不重複
    assert "- 訂單方面，決議是先簽五十臺試單\n" in out + "\n"          # 待辦半句已在表上
    assert summary_utils.assemble_long_minutes(notes, max_todos=4) == summary_utils.assemble_long_minutes(notes, max_todos=4, body_todos=[])


def test_decision_label_prefix_removed():
    out = clean_summary_markdown("## 摘要\n- 招募面試\n\n## 決議\n- 決議：8月26日前完成報到\n- 決定是先簽60臺")
    assert "- 8月26日前完成報到" in out and "- 決定是先簽60臺" in out


def test_hand_over_item_starts_after_please():
    rows = summary_utils.transcript_todo_rows("目前已經報名8人，還剩8個名額請各部門在週一前把名單交給許經理，第十個議題是上線")
    assert rows == ["| 各部門把名單交給許經理 | 許經理 | 週一前 |"]


def test_merge_recap_todo_only_recap_line_does_not_crash():
    # 11-04 夜間 3 分鐘實錄：總結裡「待辦事項是王經理主持面試」整條就是表上已有的待辦 → IndexError (處理失敗)
    md = ("## 摘要\n- 招募已面試15位候選人\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| 主持面試 | 王經理 | 12月3日 |\n\n## 決議\n- 無\n")
    tr = ("第七個議題是招募，下週會再安排四場面試，由王經理主持，" * 8 +
          "好 最後總結，招募方面，待辦事項是王經理主持面試，今天就到這裡，謝謝")
    out = summary_utils.merge_recap_into_minutes(md, tr)
    assert "| 主持面試 | 王經理 | 12月3日 |" in out


def test_truncated_table_separator_is_restored():
    raw = "## 摘要\n- 預算200萬元\n\n## 決議\n- 先簽60臺試單\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---"
    out = ensure_sections(clean_summary_markdown(raw))
    todo = out.split("## 待辦")[1].split("## ")[0]
    assert "| 事項 | 負責人 | 時間 |\n|---|---|---|\n|" in todo


def test_replacement_char_removed():
    out = clean_summary_markdown("## 摘要\n- �品保部明年預算200萬元\n\n## 決議\n- 無")
    assert "�" not in out and "- 品保部明年預算200萬元" in out


def test_body_due_prefers_explicit_deadline_and_cuts_junk():
    assert summary_utils.transcript_todo_rows("下週會再安排四場面試由王經理主持希望在12月3日前完成報到，第八個議題是") == \
        ["| 完成報到 | 王經理 | 12月3日前 |"]
    assert summary_utils.transcript_todo_rows("同意 這件事由王經理負責週五前回覆客戶PLOG1T-E是工業杈到器的便體更新，目前") == \
        ["| 回覆客戶 | 王經理 | 週五前 |"]


def test_ground_todo_drops_empty_item_rows():
    md = "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n|  | 王經理 | - |  |\n| 主持面試 | 王經理 | 12月3日 |"
    out = ground_todo_table(md, "下週由王經理主持面試，12月3日前完成報到")
    assert "|  |" not in out and "| 主持面試 | 王經理 | 12月3日 |" in out


def test_chinese_numeral_dates_are_deadlines():
    assert summary_utils.transcript_todo_rows("預算表請李協理在七月一日前送到財務部，第五個議題") == \
        ["| 送到財務部 | 李協理 | 七月一日前 |"]
    assert summary_utils.extract_todos(["存方面，決議是下一批十一月五日到，待辦事項是劉課長調整訂單"]) == \
        ["| 調整訂單 | 劉課長 | 十一月五日 |"]


def test_header_echo_rows_and_topic_label_items():
    md = ("## 摘要\n- x\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 事項 | 李協理 | 週五回報 |\n"
          "| 事項 | 劉課長 | 10月7日前送預算 |\n| 事項 | 王經理 | - |\n\n## 決議\n- 招募方面，決議是12月3日前完成報到，今天做到這裡")
    out = clean_summary_markdown(md)
    assert "| 回報 | 李協理 | 週五 |" in out and "| 送預算 | 劉課長 | 10月7日前 |" in out
    assert "| 事項 | 王經理 |" not in out and "今天做到這裡" not in out
    assert summary_utils._todo_parts("招募方面，決議是12月3日前完成報到，待辦事項王經理主持面試") is not None
    assert summary_utils._todo_parts("招募方面，決議是12月3日前完成報到，王經理主持面試") == ("主持面試", "王經理", "12月3日")


def test_en_due_filled_from_owner_sentence():
    md = "## 待辦\n| Item | Owner | Due |\n|---|---|---|\n| Reply to Lotus Tech | John | - |\n| Sign 30-unit trial order | John | - |"
    tr = "Everyone agreed. John will reply to the customer by Wednesday. Kevin owns the testing."
    out = ground_todo_table(md, tr)
    assert "| Reply to Lotus Tech | John | Wednesday |" in out
    assert "| Sign 30-unit trial order | John | - |" in out          # 動詞對不上：不補


def test_en_owner_typo_and_weekday_abbreviation():
    md = ("## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| Firmware testing | Kedin | March 15 |\n"
          "| Trial order of 30 units | John | Wed |\n| Confirm testing | Kevn | - |\n| Budget sheet | Susan | Fri |")
    out = ground_todo_table(md, "Kevin owns the testing. John will reply to Lotus Tech by Wednesday. We release on March 15. Susan sends it.")
    assert "| Firmware testing | Kevin | March 15 |" in out and "| Confirm testing | Kevin | - |" in out
    assert "| Trial order of 30 units | John | Wednesday |" in out
    assert "| Budget sheet | Susan | - |" in out          # 逐字稿沒有 Friday：不寫全名，對照後清掉


def test_body_deadline_from_following_clause_and_fill_dateless_row():
    tr = "業務部已經面試了12位候選人下週會再安排六場面試由王經理主持，希望在3月7日前完成報到，第八個議題是庫存"
    assert summary_utils.transcript_todo_rows(tr) == ["| 安排六場面試 | 王經理 | 3月7日前 |"]
    md = ("## 摘要\n- x\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 王經理主持面試 | 王經理 | - |\n\n## 決議\n- 無\n")
    out = summary_utils.merge_body_todos(md, tr)
    assert "| 王經理主持面試 | 王經理 | 3月7日前 |" in out and out.count("王經理 |") == 1


def test_dateless_row_gets_date_from_assignment_sharing_three_chars():
    md = "## 摘要\n- x\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 送預算表 | 劉課長 | - |\n\n## 決議\n- 無\n"
    tr = "人力的部分要再增加二位工程師，預算表請劉課長在10月7日前送到財務部，同意這件事由王經理負責週五前回覆客戶"
    out = summary_utils.merge_body_todos(md, tr)
    assert "| 送預算表 | 劉課長 | 10月7日前 |" in out and "送到財務部" not in out
    md2 = "## 摘要\n- x\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 王經理主持面試 | 王經理 | - |\n\n## 決議\n- 無\n"
    out2 = summary_utils.merge_body_todos(md2, "同意這件事由王經理負責週五前回覆客戶")
    assert "| 王經理主持面試 | 王經理 | - |" in out2 and "| 回覆客戶 | 王經理 | 週五前 |" in out2   # 只因為名字相同不算同一件事


def test_en_owner_accent_and_stt_typo():
    md = "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| Test firmware | Kévin | March 15 |\n| Test 100 units | Kevn | March 15 |"
    out = ground_todo_table(md, "Kevn will arrange the testing on March 15. Kevin owns the testing.")
    assert "| Test firmware | Kevin | March 15 |" in out and "| Test 100 units | Kevin | March 15 |" in out


def test_body_items_cut_garbled_topic_numbers_and_particles():
    assert summary_utils.transcript_todo_rows("預算表請王經理在6月14日前送到財務部第11個已提是公啊") == ["| 送到財務部 | 王經理 | 6月14日前 |"]
    assert summary_utils.transcript_todo_rows("下週會再安排三場面試，由林主任主吃，希望在9月11日前完成報到好，最後") == \
        ["| 完成報到 | 林主任 | 9月11日前 |"]
    assert summary_utils.transcript_todo_rows("文宣請黃專員負責，週三前交初稿第11") == ["| 交初稿 | 黃專員 | 週三前 |"]


def test_todo_action_drops_leading_shi_and_brackets():
    assert summary_utils._todo_parts("存方面，決議是下一批11月5日到，待辦事項是劉課長是訂單調貨")[0] == "訂單調貨"
    assert summary_utils._todo_parts("韌體方面，待辦事項是李協理《安排回歸測試》10月11日前")[0].startswith("安排回歸測試")


def test_summary_drops_todo_clause_already_in_table():
    md = ("## 摘要\n- 韌體修正版10月11日上線待辦事項是李協理安排回測\n- 待辦事項是林主任週三前回覆華新\n"
          "- 待辦事項是吳副理週五前交文宣\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| 安排回測 | 李協理 | 10月11日 |\n| 回覆華新 | 林主任 | 週三 |\n\n## 決議\n- 無\n")
    out = summary_utils.drop_listed_todos_from_summary(md)
    assert "- 韌體修正版10月11日上線\n" in out and "回覆華新 | 林主任" in out and "- 待辦事項是林主任" not in out
    assert "- 待辦事項是吳副理週五前交文宣\n\n## 待辦" in out        # 表上沒有的照舊保留
    assert summary_utils.drop_listed_todos_from_summary(out) == out
    only = "## 摘要\n- 待辦事項是林主任週三前回覆華新\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 回覆華新 | 林主任 | 週三 |\n"
    assert summary_utils.drop_listed_todos_from_summary(only) == only     # 摘要至少留一條


def test_cut_glued_topic_number_at_bullet_end():
    md = ("## 摘要\n- 客戶訂單：臺南訂單提前，請劉課長調貨第十五個議題\n- 第28議題品質異常：客訴六件\n- 第3個議題\n\n"
          "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 交初稿 | 第3個議題 | - |\n\n## 決議\n- 黃專員通知使用者第120個議題是出差安排\n")
    out = summary_utils.cut_glued_topic_numbers(md)
    assert "- 客戶訂單：臺南訂單提前，請劉課長調貨\n" in out and "- 第28議題品質異常：客訴六件\n" in out
    assert "- 第3個議題\n" in out and "| 第3個議題 |" in out and out.endswith("- 黃專員通知使用者\n")
    assert summary_utils.cut_glued_topic_numbers(out) == out


def test_body_todo_takes_mixed_numeral_deadline():
    assert summary_utils.transcript_todo_rows("下週會再安排三場面試由許經理主持，希望在7月十六日前完成報到") == \
        ["| 安排三場面試 | 許經理 | 7月十六日前 |"]


def test_fill_missing_topics_adds_late_topic_from_transcript():
    tr = ("第一個議題是供應商，新光材料的電源晶片交期延到10月10日，比原本晚了七天。"
          "第二個議題是新版ERP系統上線，資料轉換已經完成85%，正式切換訂在2月3日晚上，預計停機三小時。"
          "第三個議題是工業閘道器的韌體更新，目前回報的問題是藍牙斷線，修正版預計六月二十五日上線，測試的部分請黃專員安排。"
          "好，最後總結一下，供應商方面，決議是替代料持續驗證，今天就到這裡，謝謝大家")
    md = ("## 摘要\n- 供應商新光材料電源晶片交期延至10月10日\n- 新版ERP系統資料轉換完成85%，2月3日晚切換\n\n"
          "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 無 | - | - |\n\n## 決議\n- 無\n")
    out = summary_utils.fill_missing_topics(md, tr)
    assert "- 工業閘道器的韌體更新，目前回報的問題是藍牙斷線" in out and out.count("新光材料") == 1 and out.count("ERP") == 1
    assert "最後總結" not in out and "\n\n## 待辦" in out
    assert summary_utils.fill_missing_topics(out, tr) == out
    assert summary_utils.fill_missing_topics(md, "第一個議題是供應商，交期延到10月10日") == md      # 只有一個議題不補
    said = md.replace("- 新版ERP", "- 韌體：藍牙斷線，6月25日上線\n- 新版ERP")
    assert summary_utils.fill_missing_topics(said, tr) == said     # 「六月二十五日」與「6月25日」算同一個日期


def test_topic_segment_stops_at_topic_number_without_yiti():
    segs = summary_utils._topic_segments("第八個議題是庫存，目前存1200臺，下一批500臺2月7日到，第九個是教育訓練，5月25日有一場課程")
    assert segs == ["庫存，目前存1200臺，下一批500臺2月7日到"]


def test_long_minutes_treat_recap_line_with_todo_as_decision():
    notes = ["- 訂單：客戶要120臺樣品\n- 品質：客訴18件",
             "- 隨時韌體修正版10月11日上線待辦事項是李協理安排回歸測試10月11日前\n- 待辦事項是林主任週三前回覆華新"]
    md = summary_utils.assemble_long_minutes(notes, max_summary=4, max_decisions=3)
    dec = md.split("## 決議")[1]
    assert "- 隨時韌體修正版10月11日上線" in dec and "隨時韌體" not in md.split("## 待辦")[0]
    assert "待辦事項" not in summary_utils.drop_listed_todos_from_summary(md).split("## 待辦")[0]


def test_same_owner_same_task_rows_collapse():
    md = ("## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 劉課長送預算表 | 劉課長 | 10月7日前 |\n| 主持面試 | 劉課長 | 8月26日 |\n"
          "| 送預算表 | 劉課長 | - |\n| 面試：劉課長主持 | 劉課長 | 8月26日前 |\n| 回覆客戶 | 王經理 | 週五前 |\n| 主持面試 | 王經理 | 12月3日前 |\n")
    out = summary_utils._drop_same_task_rows(md)
    assert "| 送預算表 | 劉課長 | - |" not in out and "面試：劉課長主持" not in out
    assert "| 劉課長送預算表 | 劉課長 | 10月7日前 |" in out and "| 主持面試 | 劉課長 | 8月26日 |" in out
    assert "| 主持面試 | 王經理 | 12月3日前 |" in out and "| 回覆客戶 | 王經理 | 週五前 |" in out
    assert summary_utils._drop_same_task_rows(out) == out


def test_body_item_cut_at_topic_number_without_di():
    assert summary_utils.transcript_todo_rows("等試用結果再談，這件事由黃專員負責,週一前回覆客戶七個議題是招募，業務部目") == \
        ["| 回覆客戶 | 黃專員 | 週一前 |"]


def test_en_recap_owner_without_deadline_gets_a_row():
    tr = ("The first topic is training. There is a presentation skills course on September 2nd with 30 seats. "
          "22 people have signed up so far, so there are 8 seats left. The second topic is safety. Nobody was hurt. "
          "We have installed eight warning lights, and everyone must finish the safety refresher by August 15th. "
          "The third topic is the firmware update for the power supply. The fixed version is planned for March 15th. "
          "The reported problem is Bluetooth disconnection, and we will run it on 100 units for three days. "
          "Lotus Tech wants 200 sample units of the power supply by February 28th and asked for a 6 percent discount. "
          "The fourth topic is the customer order. Everyone agreed, John will reply to the customer by Wednesday. "
          "OK, to wrap up training. Names go to Mary by Thursday. firmware. We decided to release the fix on March 15th. "
          "and Kevin owns the testing customer order. We agreed on a 30-unit trial order, And John will reply to Lotus Tech by Wednesday. "
          "That's all for today. Thank you everyone!")
    md = ("## 摘要\n- Firmware fix on March 15\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n"
          "| reply to Lotus Tech | John | Wednesday |\n\n## 決議\n- 無\n")
    out = summary_utils.merge_recap_into_minutes(md, tr)
    assert "| owns the testing | Kevin | - |" in out and out.count("| John |") == 1


def test_drop_placeholder_items_only_when_section_has_real_items():
    md = ("## 摘要\n- 無\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 無 | - | - |\n\n"
          "## 決議\n- Firmware fix released March 15\n- 無\n- - None\n- We agreed on a 30-unit trial order\n")
    out = summary_utils.drop_placeholder_items(md)
    assert out.startswith("## 摘要\n- 無\n") and "| 無 | - | - |" in out
    assert out.endswith("## 決議\n- Firmware fix released March 15\n- We agreed on a 30-unit trial order\n")
    assert summary_utils.drop_placeholder_items(out) == out


def test_en_recap_run_on_sentence_split_at_topics():
    recap = ("OK, to wrap up training names go to mary by thursday, firmware we decided to release the fix on march 15th "
             "after 10 rounds of regression tests, and Kevin owns the testing customer order. "
             "We agreed on a 30-unit trial order at the new price. That's all for today.")
    notes = summary_utils._recap_notes_en(recap, topics=["training", "firmware update", "customer order"])
    assert notes.splitlines()[:2] == ["- training names go to mary by thursday",
                                      "- firmware we decided to release the fix on march 15th after 10 rounds of regression tests, "
                                      "and Kevin owns the testing"]
    assert summary_utils._recap_notes_en("We agreed on the customer order with a longer explanation than sixty characters here.",
                                         topics=["customer order"]).endswith("customer order with a longer explanation than sixty characters here")
    long = summary_utils._recap_notes_en("We decided " + "very " * 40 + "carefully.", max_len=50)
    assert long.startswith("- We decided very") and long.endswith(" very") and len(long) <= 52


def test_label_only_todo_rows_dropped_and_placeholder_kept():
    md = "## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 決議 | - | - |\n| 招募方面 | 王經理 | - |\n\n## 決議\n- x\n"
    out = summary_utils._drop_empty_item_rows(md)
    assert "| 無 | - | - |" in out and "招募方面" not in out and "| 決議 |" not in out
    md2 = md.replace("| 決議 | - | - |\n", "| 招募方面 | 王經理 | 12月3日前 |\n| 主持面試 | 王經理 | 週五 |\n")
    out2 = summary_utils._drop_empty_item_rows(md2)
    assert "| 招募方面 | 王經理 | 12月3日前 |" in out2 and "| 主持面試 | 王經理 | 週五 |" in out2 and "| 無 |" not in out2


def test_recap_and_body_todos_fill_a_todo_section_the_model_never_wrote():
    note = summary_utils.SECTION_MISSING_NOTE
    md = f"## 摘要\n- 預算：品保部明年預算200萬元\n\n## 待辦\n- {note}\n\n## 決議\n- {note}\n"
    tr = ("第一個議題是預算，品保部明年預算是200萬元，預算表請劉課長在10月7日前送到財務部。第二個議題是招募，業務部目前缺二個職缺，"
          "下週會再安排四場面試，由王經理主持，希望在12月3日前完成報到。好，最後總結一下，預算方面，決議是品保部明年預算兩百萬元，"
          "待辦事項是劉課長10月7日前送預算表，今天就到這裡，謝謝大家")
    out = summary_utils.merge_body_todos(summary_utils.merge_recap_into_minutes(md, tr), tr)
    todo = out.split("## 待辦")[1].split("## 決議")[0]
    assert "| 事項 | 負責人 | 時間 |" in todo and "劉課長" in todo and "王經理" in todo and note not in todo
    assert summary_utils.merge_body_todos(md, "今天沒有交辦事項。") == md     # 沒有可補的列：保留「未產生」說明


def test_doubled_or_decorated_headings_are_recognised():
    raw = ("## ## Summary\n- Training on Sept 2\n\n## ## Action Items\n| Item | Owner | Due |\n|---|---|---|\n"
           "| Send names | Mary | Thursday |\n\n### **Decisions**\n- Fix released March 15\n")
    out = summary_utils.ensure_sections(clean_summary_markdown(summary_utils.to_internal_sections(raw)))
    assert out.count("## 摘要") == 1 and out.count("## 待辦") == 1 and out.count("## 決議") == 1
    assert summary_utils.SECTION_MISSING_NOTE not in out and "- Training on Sept 2" in out
    assert summary_utils.to_internal_sections("### 摘要\n- a\n## 待辦事項：\n| x |") == "## 摘要\n- a\n## 待辦\n| x |"


def test_extra_recap_section_merges_into_summary():
    raw = ("## Summary\n- a\n\n## Action Items\n| Item | Owner | Due |\n|---|---|---|\n| x | John | Monday |\n\n"
           "## Decisions\n- d\n\n## Recap\n- Warehouse: Migration 90%\n")
    out = summary_utils.ensure_sections(clean_summary_markdown(summary_utils.to_internal_sections(raw)))
    assert out.count("## 摘要") == 1 and "Recap" not in out and out.split("## 待辦")[0].endswith("- Warehouse: Migration 90%\n\n")
