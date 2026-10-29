"""hailo_whisper.engine 的解碼參數 (repetition_penalty、自動語言偵測、靜音略過、日韓文重跑)，用假的 Speech2Text，不需要 Hailo 硬體。"""
import enum
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hailo_whisper import engine as eng   # noqa: E402


class _FakeS2T:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate_all_text(self, **kw):
        self.calls.append(kw)
        return self.replies.pop(0)


@pytest.fixture()
def fake_engine(monkeypatch):
    genai = types.ModuleType("hailo_platform.genai")
    genai.Speech2TextTask = enum.Enum("Speech2TextTask", {"TRANSCRIBE": "transcribe"})
    monkeypatch.setitem(sys.modules, "hailo_platform", types.ModuleType("hailo_platform"))
    monkeypatch.setitem(sys.modules, "hailo_platform.genai", genai)

    def make(replies):
        e = object.__new__(eng.HailoWhisperEngine)
        e.hef_path, e.group_id, e._vdevice, e._initialized = "x.hef", None, None, True
        e._s2t = _FakeS2T(replies)
        return e
    return make


def _speech(sec=2.0):
    t = np.arange(int(16000 * sec)) / 16000.0
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_defaults_from_ab_test():
    assert eng.WHISPER_LANGUAGE == os.environ.get("PI5_WHISPER_LANGUAGE", "auto")
    assert eng.WHISPER_REPETITION_PENALTY == float(os.environ.get("PI5_WHISPER_REPETITION_PENALTY", "1.1"))


def test_auto_language_and_repetition_penalty_are_passed(fake_engine):
    e = fake_engine(["We need two more engineers."])
    r = e.transcribe(_speech(), language="auto", repetition_penalty=1.1)
    kw = e._s2t.calls[0]
    assert kw["language"] is None and kw["repetition_penalty"] == 1.1
    assert r["text"] == "We need two more engineers." and r["language"] == "auto"


def test_penalty_1_is_not_passed_for_old_hailort(fake_engine):
    e = fake_engine(["好"])
    e.transcribe(_speech(), language="zh", repetition_penalty=1.0)
    assert "repetition_penalty" not in e._s2t.calls[0] and e._s2t.calls[0]["language"] == "zh"


def test_kana_output_is_retried_with_zh(fake_engine):
    e = fake_engine(["これはテストです", "這是測試"])
    r = e.transcribe(_speech(), language="auto", repetition_penalty=1.1)
    assert [c["language"] for c in e._s2t.calls] == [None, "zh"]
    assert r["text"] == "這是測試" and r["language"] == "zh"


def test_forced_language_is_never_retried(fake_engine):
    e = fake_engine(["これはテストです"])
    e.transcribe(_speech(), language="ja", repetition_penalty=1.1)
    assert len(e._s2t.calls) == 1


def test_near_silence_skips_the_npu(fake_engine):
    e = fake_engine([])
    r = e.transcribe(np.zeros(16000 * 8, np.float32), language="auto")
    assert r["text"] == "" and e._s2t.calls == []
    quiet = (np.random.default_rng(1).standard_normal(16000 * 6) * 10 ** (-62 / 20)).astype(np.float32)   # 錄音尾端底噪
    assert e.transcribe(quiet, language="auto")["text"] == "" and e._s2t.calls == []


def test_quiet_speech_still_goes_to_the_npu(fake_engine):
    e = fake_engine(["好"])
    voice = _speech(1.0) * 10 ** (-40 / 20) / 0.2 * np.sqrt(2)    # 約 -40 dBFS 的說話段
    e.transcribe(np.concatenate([voice, np.zeros(16000 * 3, np.float32)]), language="auto")
    assert len(e._s2t.calls) == 1


@pytest.mark.parametrize("text,retry", [("こんにちは", True), ("안녕하세요", True), ("我們下週 review PR", False), ("", False)])
def test_needs_zh_retry(text, retry):
    assert eng.needs_zh_retry(text) is retry
