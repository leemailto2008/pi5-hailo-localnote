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
    ("第一個一是招募，第23個讀題是工安，第286個涼體是供應商，第18個計是預算，第二的一集是新版ERP，第 27 個議是客戶訂單",
     "第一個議題是招募，第23個議題是工安，第286個議題是供應商，第18個議題是預算，第二的議題是新版ERP，第 27 個議題是客戶訂單"),
    ("第二個問題是品質，第一個計畫是擴產，第三個意見是延後，第一個人是王經理，第一個是預算",
     "第二個問題是品質，第一個計畫是擴產，第三個意見是延後，第一個人是王經理，第一個是預算"),      # 正常用語不動
    # 職稱的同音錯字 (負責人要靠職稱認出來)
    ("負責人是張公成師，楊工城市負責通知，請洋公城失掉貨，楊公承事負責", "負責人是張工程師，楊工程師負責通知，請楊工程師調貨，楊工程師負責"),
    ("楊工程時負責通知所有使用者", "楊工程師負責通知所有使用者"),
    ("做這個工程時，要注意安全，工程時間延長", "做這個工程時，要注意安全，工程時間延長"),       # 「工程時」是正常用法
    ("辦公城市，化工城市，員工城市生活", "辦公城市，化工城市，員工城市生活"),
    ("名單交給留客長，請菜族長確認，劉客掌安排，蔡旬長負責", "名單交給劉課長，請菜組長確認，劉課長安排，蔡組長負責"),
    ("顧客長期合作，乘客長途跋涉，家族長輩，中旬長達十天，客掌握了，來客掌聲鼓勵", "顧客長期合作，乘客長途跋涉，家族長輩，中旬長達十天，客掌握了，來客掌聲鼓勵"),
    ("請理協利安排，林協禮主持，劉協李負責，王協裡回報", "請李協理安排，林協理主持，劉協理負責，王協理回報"),
    ("互協利益，協力廠商，協利率", "互協利益，協力廠商，協利率"),
    ("無負理負責，吳負禮主持", "吳副理負責，吳副理主持"),
    ("正負理論值，負利率，擔負理想，勝負理由", "正負理論值，負利率，擔負理想，勝負理由"),
    ("名單交給吳父裡，出差安排，五富里下個月，要請吳付禮掉貨，文宣請吳父理負責", "名單交給吳副理，出差安排，五副理下個月，要請吳副理調貨，文宣請吳副理負責"),
    ("他住在富里，請在富里下車，交給富里鄉公所，付禮金", "他住在富里，請在富里下車，交給富里鄉公所，付禮金"),     # 地名、正常用語不動
    # 招募／訂單句型
    ("已經面世了12位候選人，下週會再安排三場面是,有林主任主持，希望在4月10日完成報道",
     "已經面試了12位候選人，下週會再安排三場面試,由林主任主持，希望在4月10日完成報到"),
    ("還剩8個名兒，請各部門，只有 5 個名，還有十二個名.", "還剩8個名額，請各部門，只有 5 個名額，還有十二個名額."),
    ("我建議先簽30臺的市單，價格維持新爆價，先簽 50 臺事單", "我建議先簽30臺的試單，價格維持新報價，先簽 50 臺試單"),
    ("並在8月15日前完成全原附訊，負責人是許經理", "並在8月15日前完成全員複訓，負責人是許經理"),
    ("新產品已經面世了，這方面是重點，兩場面是很熱鬧", "新產品已經面世了，這方面是重點，兩場面是很熱鬧"),   # 正常用法不動
    ("記者完成報導後，取三個名，50臺是單價最低，疫情新爆發", "記者完成報導後，取三個名，50臺是單價最低，疫情新爆發"),
    # 夾在中文中間的英文碎片
    ("業務部目前 three 個職缺，大約可以 six 週，替代料持續's 驗證，至少要做三輪回 its 測試",
     "業務部目前三個職缺，大約可以六週，替代料持續驗證，至少要做三輪回歸測試"),
    ("新的 ERP 系統，這個 bug 要修，pin 腳要改，test 跑完，ok 嗎", "新的 ERP 系統，這個 bug 要修，pin 腳要改，test 跑完，ok 嗎"),   # 正常英文保留
    ("version is 3.2", "version is 3.2"),                                    # 前後不是中文：不動
    # 展覽／測試／產品名
    ("5月12日在新竹有一場展覚，攀胃費30萬元，攀位費15萬元", "5月12日在新竹有一場展覽，攤位費30萬元，攤位費15萬元"),
    ("至少要做三輪回測，十輪回歷測式，5輪回試，測式的部分", "至少要做三輪回歸測試，十輪回歸測試，5輪回歸測試，測試的部分"),
    ("電源供印器的韌體，電源印器目前存800臺，文控模組", "電源供應器的韌體，電源供應器目前存800臺，溫控模組"),
    ("品質弈長上個月溫通模組的客訴，品質一，上個月灣控模組，天性模組，文寇模組", "品質異常上個月溫控模組的客訴，品質異常，上個月溫控模組，天線模組，溫控模組"),
    ("品質一流，天性善良，品質第一，", "品質一流，天性善良，品質第一，"),
    ("預測式模型，股票回測，文控中心，文控流程", "預測式模型，股票回測，文控中心，文控流程"),            # 正常用法不動
    ("待辦事項是王精理主持面是，請張金李跟對方確認，交給林經歷，", "待辦事項是王經理主持面試，請張經理跟對方確認，交給林經理，"),
    ("請大家精力集中，他的經歷很豐富，主持面是前要準備", "請大家精力集中，他的經歷很豐富，主持面是前要準備"),   # 正常用法不動
    ("So this is, the plan: two, three, and one more. In the end it is on track.",
     "So this is, the plan: two, three, and one more. In the end it is on track."),          # 英文會議不動
    ("Okay，let's start today's weekly meeting. The second topic is safety，last week there was an accident。",
     "Okay, let's start today's weekly meeting. The second topic is safety, last week there was an accident."),
    ("新的 ERP 系統，下週上線。", "新的 ERP 系統，下週上線。"),
    ("我們這個 release 的 schedule 要 delay，因為 QA 那邊的 regression test 還沒 pass。",
     "我們這個 release 的 schedule 要 delay，因為 QA 那邊的 regression test 還沒 pass。"),         # 中英夾雜的中文：全形標點不動                                   # 中文會議的全形標點不動
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
    ("他從小學藝是為了興趣", "他從小學藝是為了興趣"),
    ("最後總結，訓練方面,隨意是簡報技巧課程9月2日開課，他很隨意是個好人", "最後總結，訓練方面,決議是簡報技巧課程9月2日開課，他很隨意是個好人"),                 # 前面不是「方面／總結一下」：不動
    # 負責人的姓氏同音錯字、人數、工安、供應商
    ("預算表請留課長在10月7日前送到財物部，目前粒率是百分之九十，情立協理跟對方確認",
     "預算表請劉課長在10月7日前送到財務部，目前良率是百分之九十，請李協理跟對方確認"),
    ("請零主任安排，交給領主任，情取經理跟對方，負責人是菜組長，有洋工程師主持，請皇專員跟對方，情緒經理會在",
     "請林主任安排，交給林主任，請許經理跟對方，負責人是蔡組長，由楊工程師主持，請黃專員跟對方，請許經理會在"),
    ("請保留課長的位子，處理經理會在，帶領工程師，這件事情經理會處理，錄取工程師，繼續工程師的", 
     "請保留課長的位子，處理經理會在，帶領工程師，這件事情經理會處理，錄取工程師，繼續工程師的"),   # 成詞的不動
    ("預算表清理協理在6月14日前，測試的部分清理協理安排，安排四場面試,有無副理主持", "預算表請李協理在6月14日前，測試的部分請李協理安排，安排四場面試,由吳副理主持"),
    ("清理工程師的桌面，確認有無副理在場", "清理工程師的桌面，確認有無副理在場"),
    ("人力的部分要再增加二維工程師，再加四圍工程師，增2為工程師", "人力的部分要再增加二位工程師，再加四位工程師，增2位工程師"),
    ("第三個議題是公安上週三樓產線發生事故，公安方面，決議是", "第三個議題是公安上週三樓產線發生事故，公安方面，決議是"),   # 公安／工安：維持現狀
    ("第一個議題是應商星光材料，依應商方面，侶應商，供應商", "第一個議題是供應商星光材料，供應商方面，供應商，供應商"),
    ("王經理負責週五前回付客戶，週二前回不客，週三前回副客第四個議題，週一前回覆客戶，溫控模組的客宿有九件，電表的客速有12件",
     "王經理負責週五前回覆客戶，週二前回覆客戶，週三前回覆客戶第四個議題，週一前回覆客戶，溫控模組的客訴有九件，電表的客訴有12件"),
    ("待辦事項是劉客 送預算表，名單交給留客，請留客跟對方確認，請留客帳安排，像是理協理週五回報",
     "待辦事項是劉課長送預算表，名單交給劉課長，請劉課長跟對方確認，請劉課長安排，像是李協理週五回報"),
    ("完成全員複訓待辦事，是蔡主長負責，待辦事項是楊公城司主持面試，3月19日完成報到大半事像是林主任主持面試",
     "完成全員複訓待辦事項，是蔡組長負責，待辦事項是楊工程師主持面試，3月19日完成報到待辦事項是林主任主持面試"),
    ("大半事項都完成了，公主長得漂亮，主長官", "大半事項都完成了，公主長得漂亮，主長官"),
    ("待辦事項是留課長是訂單調貨，保留課長是", "待辦事項是劉課長是訂單調貨，保留課長是"),
    ("我們要想辦法留客，交給顧客，請遊客在門口，是游客負責", "我們要想辦法留客，交給顧客，請遊客在門口，是游客負責"),
    ("回客運站，客宿一晚，回不去客廳", "回客運站，客宿一晚，回不去客廳"),
    ("沒有問題在全面更新，下週會在安排四場面試，等是用結果在談，先在50臺裝置上市跑3天，三場面試有許經理主持，這件是由張工程師負責",
     "沒有問題再全面更新，下週會再安排四場面試，等試用結果再談，先在50臺裝置上試跑3天，三場面試由許經理主持，這件事由張工程師負責"),
    ("預算表送到財物部，明年平保部的預算，產現已經加開一站，資安課城名額，五位同人支援，請個部門在週四前，先千八十臺的試單，智慧店表的刻速有18件",
     "預算表送到財務部，明年品保部的預算，產線已經加開一站，資安課程名額，五位同仁支援，請各部門在週四前，先簽八十臺的試單，智慧電表的客訴有18件"),
    ("目前2個直已經面試，價格為持新報價，預寄展出8款，一場自安課程，工業炸到期的客訴，精密的外科模具，負負人是張工程師，替代了持續驗證，第四個議題示預算",
     "目前2個職缺已經面試，價格維持新報價，預計展出8款，一場資安課程，工業閘道器的客訴，精密的外殼模具，負責人是張工程師，替代料持續驗證，第四個議題是預算"),
    ("品質一長上個月，地下停車廠，加裝八指警示燈，85%正是切換，文生請李協理，晶片交其延到10月，黃專源會在2月，對單價有一件希望降價，決議是新千五十臺，面試遊許經理主持，已經面識了八位候選人",
     "品質異常上個月，地下停車場，加裝八支警示燈，85%正式切換，文宣請李協理，晶片交期延到10月，黃專員會在2月，對單價有意見希望降價，決議是先簽五十臺，面試由許經理主持，已經面試了八位候選人"),
    ("負負得正，課成本，有一件事情", "負負得正，課成本，有一件事情"),
    ("正在安排會議，會在4月5日前，新產品上市，有人主持會議，保管財物，刻速度", "正在安排會議，會在4月5日前，新產品上市，有人主持會議，保管財物，刻速度"),
    ("回應商家，對應商品，反應商機，適應商業", "回應商家，對應商品，反應商機，適應商業"),
    ("", ""),
])
def test_fix_meeting_terms(raw, fixed):
    assert eng.fix_meeting_terms(raw) == fixed


def test_meeting_terms_fixed_in_npu_output(fake_engine):
    e = fake_engine(["最後總結一下，預算方面決意是維持原案"])
    assert e.transcribe(_speech(), language="auto")["text"].endswith("預算方面決議是維持原案")


def test_fix_meeting_terms_is_idempotent():
    # 切片各修一次、整份逐字稿接起來再修一次，結果要和只修一次相同
    for raw in ("業務部目前，two of 職已經面世了15位候選人", "最後總結一下，舊一是簡報技巧課程，交給吳父裡，楊公成師負責",
                "a small accident no one ，已經加裝四支", "情緒經理跟對方，依應商，增加二維工程師，公安上週",
                "由黃專員負責週三前回 er 客，第 73 個議題", "請留客-負，待辦事項是劉客 送預算表", "要請王經理調貨，第428個 er 體是電源供應器"):
        once = eng.fix_meeting_terms(raw)
        assert eng.fix_meeting_terms(once) == once

