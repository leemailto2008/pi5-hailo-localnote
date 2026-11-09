"""run_suite 的案例清單組裝：數字長度與英文會議各自對應到正確的 fixture / WAV，英文門檻長度取 fixture 的 minutes。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools", "e2e"))

import run_suite  # noqa: E402


def _write_fixture(folder, name, minutes):
    path = os.path.join(folder, f"meeting_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"minutes": minutes}, f)
    return path


def test_numeric_minutes_map_to_suffixed_fixture_and_wav(tmp_path):
    specs = run_suite.case_specs([1, 3.5], [], str(tmp_path), "/data")
    assert [s["label"] for s in specs] == ["1min", "3.5min"]
    assert specs[0]["fixture"] == os.path.join(str(tmp_path), "meeting_1min.json")
    assert specs[1]["wav"] == os.path.join("/data", "meeting_3.5min.wav")
    assert specs[0]["title"] == "1 分"


def test_english_case_takes_threshold_minutes_from_fixture(tmp_path):
    _write_fixture(str(tmp_path), "en_short", 2.1)
    specs = run_suite.case_specs([], ["en_short"], str(tmp_path), "/data")
    assert specs == [{
        "label": "en_short",
        "title": "英文 en_short",
        "minutes": 2.1,
        "fixture": os.path.join(str(tmp_path), "meeting_en_short.json"),
        "wav": os.path.join("/data", "meeting_en_short.wav"),
    }]


def test_numeric_then_english_keep_order(tmp_path):
    _write_fixture(str(tmp_path), "en_long", 5.65)
    specs = run_suite.case_specs([1], ["en_long"], str(tmp_path), "/data")
    assert [s["label"] for s in specs] == ["1min", "en_long"]
    assert specs[1]["minutes"] == 5.65
