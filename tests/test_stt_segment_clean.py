"""voice_pipeline.clean_stt_segment()：Hailo NPU 切片輸出的解碼迴圈清理 (2026-11-02，228 10929 個切片中 23 個)。"""
import sys
import types

import pytest

sys.modules.setdefault("config", types.SimpleNamespace(WHISPER_MODEL_SIZE="small", WHISPER_DEVICE="cpu", WHISPER_COMPUTE_TYPE="int8",
                                                       DEFAULT_TTS_VOICE="zh-TW-HsiaoChenNeural", DATA_DIR="/tmp"))
import voice_pipeline as vp  # noqa: E402


@pytest.mark.parametrize("raw,clean", [
    ("上線前先在二十臺裝置上是跑三天沒有問題再全面更新ZZZZZZZZ", "上線前先在二十臺裝置上是跑三天沒有問題再全面更新"),
    ("林主任會在 10月21日前提出改善報告ZZZZZZZZZZZZZzZzZzZ", "林主任會在 10月21日前提出改善報告"),
    ("聖珀日，三日中，三日，三日，三日，三日，三日,", "聖珀日，三日中，三日，三日,"),
    ("I'm a little bit of a little bit of a little bit of a little bit of a", ""),     # 沒有中文的英文迴圈：整段丟掉
    # 正常內容不動
    ("測試的部分請御經理安排至少要做10輪回 測", "測試的部分請御經理安排至少要做10輪回 測"),
    ("名額30人目前已經報名22人還剩8個名", "名額30人目前已經報名22人還剩8個名"),
    ("對對對，好好好，我們先這樣", "對對對，好好好，我們先這樣"),
    ("We need two more engineers to finish the migration", "We need two more engineers to finish the migration"),
    ("no no no, that is not what I said", "no no no, that is not what I said"),
    ("yes yes yes I agree", "yes I agree"),             # 單字重複：壓成一次 (沿用原本的片語規則)，不會整段丟掉
    ("AAA 級的供應商", "AAA 級的供應商"),
    ("", ""),
])
def test_clean_stt_segment(raw, clean):
    assert vp.clean_stt_segment(raw) == clean
