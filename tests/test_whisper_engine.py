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

    monkeypatch.setattr(eng, "speech_seconds", lambda audio, sr=16000: 1.0)   # 預設 VAD 判定有語音；VAD 測試再覆寫

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


def test_vad_without_speech_skips_the_npu(fake_engine, monkeypatch):
    e = fake_engine([])
    monkeypatch.setattr(eng, "speech_seconds", lambda audio, sr=16000: 0.0)
    noise = (np.random.default_rng(2).standard_normal(16000 * 8) * 10 ** (-49 / 20)).astype(np.float32)   # 228 實測底噪
    assert e.transcribe(noise, language="auto")["text"] == "" and e._s2t.calls == []


def test_vad_unavailable_still_sends_the_slice(fake_engine, monkeypatch):
    e = fake_engine(["好"])
    monkeypatch.setattr(eng, "speech_seconds", lambda audio, sr=16000: None)
    assert e.transcribe(_speech(), language="auto")["text"] == "好" and len(e._s2t.calls) == 1


def _fake_vad_module(monkeypatch, spans):
    vad = types.ModuleType("faster_whisper.vad")
    vad.VadOptions = lambda **kw: kw
    vad.get_speech_timestamps = lambda audio, vad_options=None, sampling_rate=16000: spans
    monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(eng, "_VAD_OPTIONS", None)


def test_speech_seconds_sums_vad_spans(monkeypatch):
    _fake_vad_module(monkeypatch, [{"start": 0, "end": 8000}, {"start": 16000, "end": 40000}])
    assert eng.speech_seconds(np.zeros(48000, np.float32)) == pytest.approx(2.0)
    _fake_vad_module(monkeypatch, [])
    assert eng.speech_seconds(np.zeros(48000, np.float32)) == 0.0


def test_speech_seconds_can_be_disabled(monkeypatch):
    _fake_vad_module(monkeypatch, [])
    monkeypatch.setattr(eng, "WHISPER_VAD", False)
    assert eng.speech_seconds(np.zeros(48000, np.float32)) is None


def test_speech_seconds_without_faster_whisper(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", None)    # import 會失敗
    assert eng.speech_seconds(np.zeros(48000, np.float32)) is None


@pytest.mark.parametrize("raw,fixed", [
    ("招募方面，決意是11月28日前完成報到，代辦事像是許經理主持面試", "招募方面，決議是11月28日前完成報到，待辦事項是許經理主持面試"),
    ("品質方面，捷一是產線加開一站檢查，待辦事相是黃專員", "品質方面，決議是產線加開一站檢查，待辦事項是黃專員"),
    ("代版事向是劉課長送預算表", "待辦事項是劉課長送預算表"),
    ("決議是產線加開一站檢查，戴版是向是黃專員", "決議是產線加開一站檢查，待辦事項是黃專員"),
    ("待辦事是許經理主持面試", "待辦事項是許經理主持面試"),
    ("待辦事想是王經理回覆客戶", "待辦事項是王經理回覆客戶"),
    ("代辦是向國稅局申請的", "代辦是向國稅局申請的"),    # 後面沒接「是」：正常用法不動
    ("待半時像是黃專員", "待辦事項是黃專員"),
    ("他的決意很堅定", "他的決意很堅定"),            # 不是「決議是」的用法不動
    ("請代辦簽證", "請代辦簽證"),
    ("大半事項都完成了", "大半事項都完成了"),
    ("", ""),
])
def test_fix_meeting_terms(raw, fixed):
    assert eng.fix_meeting_terms(raw) == fixed


def test_meeting_terms_fixed_in_npu_output(fake_engine):
    e = fake_engine(["最後總結一下，預算方面決意是維持原案"])
    assert e.transcribe(_speech(), language="auto")["text"].endswith("預算方面決議是維持原案")

