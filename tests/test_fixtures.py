"""測試會議 fixtures 的一致性：每個事實都真的被說出來、總結事實都在結尾段、長度符合規格。"""
import glob
import json
import os

import pytest

import meeting_gen
import scoring

FX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "e2e", "fixtures")
FIXTURES = sorted(glob.glob(os.path.join(FX_DIR, "meeting_*min.json")))


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def test_all_lengths_present():
    names = {os.path.basename(p) for p in FIXTURES}
    for m in meeting_gen.SHORT_MINUTES + meeting_gen.LONG_MINUTES:
        assert f"meeting_{m}min.json" in names


def test_long_series_share_topic_prefix():
    """長會議共用 seed：較短者的議題段落必須是較長者的前綴 (TTS 快取可共用的前提)。"""
    long = [_load(os.path.join(FX_DIR, f"meeting_{m}min.json")) for m in meeting_gen.LONG_MINUTES]
    for a, b in zip(long, long[1:]):
        sa, sb = a["segments"][1:-1], b["segments"][1:-1]
        assert sb[:len(sa)] == sa


def test_api_limit_covers_longest_fixture():
    import re
    main_py = open(os.path.join(os.path.dirname(FX_DIR), "..", "..", "code_dispatcher", "main.py"), encoding="utf-8").read()
    le = int(re.search(r"seconds: int = Field\(default=90, ge=10, le=(\d+)", main_py).group(1))
    for p in FIXTURES:
        fx = _load(p)
        if "audio" in fx:
            assert int(fx["audio"]["duration_s"] + 4) <= le   # e2e.py 以 seconds=int(dur + 4) 呼叫 /api/record


@pytest.mark.parametrize("path", FIXTURES, ids=os.path.basename)
def test_facts_are_spoken(path):
    fx = _load(path)
    text = "".join(t for _, t in fx["lines"])
    missing = [k for k, ok in scoring.fact_hits(text, fx["facts"]).items() if not ok]
    assert not missing, missing


@pytest.mark.parametrize("path", FIXTURES, ids=os.path.basename)
def test_recap_facts_in_closing(path):
    fx = _load(path)
    closing = "".join(l["text"] for s in fx["segments"] if s["name"] == "closing" for l in s["lines"])
    assert set(fx["recap_facts"]) <= set(fx["facts"])
    missing = [k for k, ok in scoring.fact_hits(closing, fx["recap_facts"]).items() if not ok]
    assert not missing, missing


@pytest.mark.parametrize("path", FIXTURES, ids=os.path.basename)
def test_structure_and_duration(path):
    fx = _load(path)
    flat = [[l["spk"], l["text"]] for s in fx["segments"] for l in s["lines"]]
    assert flat == fx["lines"]
    assert {spk for spk, _ in fx["lines"]} <= set(fx["speakers"])
    if "audio" in fx:
        dur, target = fx["audio"]["duration_s"], min(fx["minutes"] * 60, meeting_gen.MAX_AUDIO_S)
        tol = 0.30 if fx["minutes"] <= 1 else 0.12   # 與 gen_test_audio.py 一致
        assert abs(dur - target) <= tol * target
        assert dur <= meeting_gen.MAX_AUDIO_S
        assert len(fx["timeline"]) == len(fx["lines"])
        assert all(a["end_s"] <= b["start_s"] for a, b in zip(fx["timeline"], fx["timeline"][1:]))


@pytest.mark.parametrize("n,colloquial,formal", [(2, "二", "二"), (10, "十", "十"), (15, "十五", "十五"), (105, "一百零五", "一百零五"),
                                                 (200, "兩百", "二百"), (1500, "一千五百", "一千五百"), (2000, "兩千", "二千"),
                                                 (12000, "一萬兩千", "一萬二千")])
def test_zh_num(n, colloquial, formal):
    assert meeting_gen.zh_num(n) == colloquial and meeting_gen.zh_num(n, colloquial=False) == formal
