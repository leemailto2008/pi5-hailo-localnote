import asyncio
import json

import httpx
import pytest

import llm_backend as lb


def _backend(name, is_hailo):
    return lb.LLMBackend(name=name, base_url=f"http://{name}.test", model=f"{name}-model", is_hailo=is_hailo)


def _run(gen):
    async def collect():
        return [(tok, b.name) async for tok, _done, b in gen]
    return asyncio.run(collect())


@pytest.fixture
def fake_net(monkeypatch):
    """hailo 前 fail_first 次連線失敗，之後正常串流；cpu 一律正常。記錄每次請求打到哪個後端。"""
    state = {"hits": [], "fail_first": 1, "sleeps": []}
    hailo, cpu = _backend("hailo", True), _backend("cpu", False)

    def handler(request):
        host = request.url.host.split(".")[0]
        state["hits"].append(host)
        if host == "hailo" and state["hits"].count("hailo") <= state["fail_first"]:
            raise httpx.ConnectError("restarting", request=request)
        body = "\n".join(json.dumps({"message": {"content": t}, "done": d}) for t, d in ((f"{host}-a", False), (f"{host}-b", True)))
        return httpx.Response(200, text=body)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(lb.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(lb, "backends_in_order", lambda *a, **k: [hailo, cpu])
    monkeypatch.setattr(lb, "HAILO_BACKEND", hailo)
    monkeypatch.setattr(lb, "_free_npu_for_llm", lambda: None)

    async def fake_sleep(sec):
        state["sleeps"].append(sec)
    monkeypatch.setattr(lb.asyncio, "sleep", fake_sleep)
    return state


def test_stream_retries_hailo_before_cpu(fake_net):
    out = _run(lb.aiter_chat_stream("hi", hailo_retries=2, hailo_retry_wait=5.0))
    assert [t for t, _ in out if t] == ["hailo-a", "hailo-b"] and fake_net["hits"] == ["hailo", "hailo"]
    assert fake_net["sleeps"] == [5.0]


def test_stream_without_retries_falls_back_to_cpu(fake_net):
    out = _run(lb.aiter_chat_stream("hi"))
    assert [t for t, _ in out if t] == ["cpu-a", "cpu-b"] and fake_net["hits"] == ["hailo", "cpu"]


def test_stream_hailo_only_never_uses_cpu(fake_net):
    fake_net["fail_first"] = 9
    with pytest.raises(lb.LLMError):
        _run(lb.aiter_chat_stream("hi", hailo_retries=2, hailo_only=True))
    assert fake_net["hits"] == ["hailo"] * 3 and "cpu" not in fake_net["hits"]
