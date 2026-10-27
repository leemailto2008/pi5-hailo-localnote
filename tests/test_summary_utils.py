from summary_utils import clean_summary_markdown


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
