"""last_meeting：只保留一份、整份覆蓋；檔案不存在或壞掉時回傳 None，不影響服務啟動。"""
import json

import last_meeting


def _result(job, text="好，我們開始"):
    return {"job": job, "title": "會議紀要", "started_at": "2026-10-30T15:00:00", "duration": 90,
            "summary": ["a"], "todos": [], "decisions": [], "transcript": text, "markdown": "# x"}


def test_save_then_load_keeps_only_the_latest(tmp_path):
    p = str(tmp_path / "data" / "last_meeting.json")
    assert last_meeting.save_last_meeting(_result("m_1"), {"metrics": {"link_arrival_pct": 89.3}}, path=p)
    assert last_meeting.save_last_meeting(_result("m_2", "第二場"), {"metrics": {}}, path=p)
    got = last_meeting.load_last_meeting(path=p)
    assert got["result"]["job"] == "m_2" and got["result"]["transcript"] == "第二場"
    assert got["saved_at"] and got["status"] == {"metrics": {}}
    assert [f.name for f in (tmp_path / "data").iterdir()] == ["last_meeting.json"]    # 沒有殘留暫存檔


def test_missing_or_broken_file_is_none(tmp_path):
    p = tmp_path / "last_meeting.json"
    assert last_meeting.load_last_meeting(path=str(p)) is None
    p.write_text("{壞掉", encoding="utf-8")
    assert last_meeting.load_last_meeting(path=str(p)) is None
    p.write_text(json.dumps({"result": {"job": ""}}), encoding="utf-8")
    assert last_meeting.load_last_meeting(path=str(p)) is None


def test_save_failure_does_not_raise(tmp_path):
    blocker = tmp_path / "f"
    blocker.write_text("x")                      # 目錄位置被檔案佔住 → makedirs 失敗
    assert last_meeting.save_last_meeting(_result("m_1"), {}, path=str(blocker / "last.json")) is False


def test_env_overrides_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PI5_LAST_MEETING_PATH", str(tmp_path / "x.json"))
    assert last_meeting.last_meeting_path() == str(tmp_path / "x.json")
