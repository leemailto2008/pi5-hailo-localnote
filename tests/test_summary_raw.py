"""MeetingJobManager._generate_summary_llm 會把 LLM 原文（未後處理）存進 status.summary_raw，供 e2e 存檔。

只驗證欄位內容與預設值；畫面用的 summary_stream 與後處理結果不在本測試範圍內。
"""
import asyncio
import sys
import types

import pytest

sys.modules.setdefault("config", types.SimpleNamespace(WHISPER_MODEL_SIZE="small", WHISPER_DEVICE="cpu", WHISPER_COMPUTE_TYPE="int8",
                                                       DEFAULT_TTS_VOICE="zh-TW-HsiaoChenNeural", DATA_DIR="/tmp"))
import meeting_minutes as mm  # noqa: E402


class _Backend:
    label = "fake-backend"


def _fake_stream(chunks):
    async def gen(prompt, **kwargs):
        for c in chunks:
            yield c, False, _Backend()
        yield "", True, _Backend()
    return gen


def _run_llm(chunks, monkeypatch):
    monkeypatch.setattr(mm, "aiter_chat_stream", _fake_stream(chunks))
    job = mm.MeetingJobManager.__new__(mm.MeetingJobManager)   # 跳過 __init__，避免硬體與背景任務
    job.status = mm.JobStatus()
    job._cancellation_token = None
    out = asyncio.run(job._generate_summary_llm("王經理：請李協理週五前回報供應商交期。", job_id=None))
    return job, out


def test_summary_raw_default_is_none():
    assert mm.JobStatus().summary_raw is None


def test_summary_raw_keeps_unprocessed_llm_output(monkeypatch):
    chunks = ["## 摘要\n", "- 供應商交期延至10月10日\n\n", "## 待辦\n",
              "| 事項 | 負責人 | 時間 |\n", "|---|---|---|\n", "| 回報交期 | 李協理 | 週五 |\n"]
    job, out = _run_llm(chunks, monkeypatch)
    assert job.status.summary_raw == "".join(chunks)
    assert out and out != job.status.summary_raw   # 後處理後的紀要與原文不同，證明 raw 未被改寫
