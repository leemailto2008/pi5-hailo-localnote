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
    ("開課代半事向十個部門週四前交名單，待幹是像是黃專員", "開課待辦事項十個部門週四前交名單，待辦事項是黃專員"),
    ("期待幹部們的表現", "期待幹部們的表現"),
    ("黃專員會在2.15日前提出改善報告", "黃專員會在2月15日前提出改善報告"),      # 日期寫法統一
    ("正式切換定於 10 月 11 日 晚上", "正式切換定於 10月11日 晚上"),
    ("預計12.8日到貨，9.16 日前完成", "預計12月8日到貨，9月16日前完成"),
    ("良率是 95.5%，還要3.5天", "良率是 95.5%，還要3.5天"),                   # 不是「X.Y日」不動
    ("版本13.5日誌", "版本13.5日誌"),                                          # 13 月不存在
    ("1.12.5日", "1.12.5日"),
    ("在 6. 04. 日前完成，1.20. 日開會，2. 03 日切換", "在 6月4日前完成，1月20日開會，2月3日切換"),   # 點旁有空白、多一個點、補零                                                   # 前面接小數點不動
    ("他的決意很堅定", "他的決意很堅定"),            # 不是「決議是」的用法不動
    ("請代辦簽證", "請代辦簽證"),
    ("大半事項都完成了", "大半事項都完成了"),
    ("第二個議題是品值異常，品值方面，決議是加開一站", "第二個議題是品質異常，品質方面，決議是加開一站"),   # 品質
    ("產品值得信賴，商品值錢，產品值多少", "產品值得信賴，商品值錢，產品值多少"),                       # 跨詞的正常組合不動
    ("第六個議題是招木，朝目方面，決議是，朝幕方面，招牧業務部", "第六個議題是招募，招募方面，決議是，招募方面，招募業務部"),
    ("朝目標前進，這一招目前還有效，王朝幕後", "朝目標前進，這一招目前還有效，王朝幕後"),             # 招募：正常用法不動
    ("第一個議題是招牆資訊部，招床方面", "第一個議題是招募資訊部，招募方面"),
    ("這一招牆上也貼了，那一招床邊放著", "這一招牆上也貼了，那一招床邊放著"),
    ("第二個一提是品質異常，第三個 一題 是工安，第四個意提市預算", "第二個議題是品質異常，第三個 議題 是工安，第四個議題市預算"),
    ("第一題是選擇題，第二個議題是招募", "第一題是選擇題，第二個議題是招募"),        # 沒有「個」、或本來就對：不動
    # 職稱的同音錯字 (負責人要靠職稱認出來)
    ("負責人是張公成師，楊工城市負責通知，請洋公城失掉貨，楊公承事負責", "負責人是張工程師，楊工程師負責通知，請洋工程師掉貨，楊工程師負責"),
    ("楊工程時負責通知所有使用者", "楊工程師負責通知所有使用者"),
    ("做這個工程時，要注意安全，工程時間延長", "做這個工程時，要注意安全，工程時間延長"),       # 「工程時」是正常用法
    ("辦公城市，化工城市，員工城市生活", "辦公城市，化工城市，員工城市生活"),
    ("名單交給留客長，請菜族長確認，劉客掌安排，蔡旬長負責", "名單交給留課長，請菜組長確認，劉課長安排，蔡組長負責"),
    ("顧客長期合作，乘客長途跋涉，家族長輩，中旬長達十天，客掌握了，來客掌聲鼓勵", "顧客長期合作，乘客長途跋涉，家族長輩，中旬長達十天，客掌握了，來客掌聲鼓勵"),
    ("請理協利安排，林協禮主持，劉協李負責，王協裡回報", "請理協理安排，林協理主持，劉協理負責，王協理回報"),
    ("互協利益，協力廠商，協利率", "互協利益，協力廠商，協利率"),
    ("無負理負責，吳負禮主持", "無副理負責，吳副理主持"),
    ("正負理論值，負利率，擔負理想，勝負理由", "正負理論值，負利率，擔負理想，勝負理由"),
    # 韌體／良率／驗證
    ("第三個議題是電源供應器的任體更新，任體當機，任體方面，決議是", "第三個議題是電源供應器的韌體更新，韌體當機，韌體方面，決議是"),
    ("認體更新，韁體修正版", "韌體更新，韌體修正版"),
    ("擔任體育老師，他擔任體操教練", "擔任體育老師，他擔任體操教練"),       # 「任體」後面不是韌體相關詞：不動
    ("目前量率是 90%，糧率 95%，涼率是百分之九十", "目前良率是 90%，良率 95%，良率是百分之九十"),
    ("流量率是5%，產量率 80%，含量率 3%，量率很重要", "流量率是5%，產量率 80%，含量率 3%，量率很重要"),
    ("替代料已經在驗譁，目前在驗症", "替代料已經在驗證，目前在驗證"),
    ("檢驗症狀", "檢驗症狀"),
    # 「方面／總結一下」後面的決議同音錯字
    ("供應商方面，學藝是替代料持續驗證，展覽方面,冶一是參加新竹展", "供應商方面，決議是替代料持續驗證，展覽方面,決議是參加新竹展"),
    ("最後總結一下，舊一是簡報技巧課程開課", "最後總結一下，決議是簡報技巧課程開課"),
    ("他從小學藝是為了興趣", "他從小學藝是為了興趣"),                 # 前面不是「方面／總結一下」：不動
    ("", ""),
])
def test_fix_meeting_terms(raw, fixed):
    assert eng.fix_meeting_terms(raw) == fixed


def test_meeting_terms_fixed_in_npu_output(fake_engine):
    e = fake_engine(["最後總結一下，預算方面決意是維持原案"])
    assert e.transcribe(_speech(), language="auto")["text"].endswith("預算方面決議是維持原案")

