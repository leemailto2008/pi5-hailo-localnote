"""NPU 共用模式 (npu_share / llm_backend / stt_worker) 的單元測試，不需要 Hailo 硬體。"""
import os
import threading
import time

import pytest

import llm_backend as lb
import npu_share as ns
import stt_worker as sw


def _fake_proc(root, pid, comm, env):
    d = root / str(pid)
    d.mkdir()
    (d / "comm").write_bytes(comm.encode() + b"\n")
    (d / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in env.items()) + b"\0")


def test_scan_finds_group_of_hailo_ollama(tmp_path):
    _fake_proc(tmp_path, 10, "bash", {"HAILO_OLLAMA_VDEVICE_GROUP_ID": "WRONG"})
    _fake_proc(tmp_path, 20, "hailo-ollama", {"OLLAMA_HOST": "127.0.0.1:11435", "HAILO_OLLAMA_VDEVICE_GROUP_ID": "HAILO_OLLAMA_SHARED"})
    (tmp_path / "self").mkdir()
    assert ns._scan_hailo_ollama_group(str(tmp_path)) == "HAILO_OLLAMA_SHARED"


def test_scan_without_group_or_process(tmp_path):
    assert ns._scan_hailo_ollama_group(str(tmp_path)) is None
    _fake_proc(tmp_path, 20, "hailo-ollama", {"OLLAMA_HOST": "127.0.0.1:11435"})
    assert ns._scan_hailo_ollama_group(str(tmp_path)) is None
    assert ns._scan_hailo_ollama_group(str(tmp_path / "missing")) is None


def test_shared_group_can_be_disabled(monkeypatch):
    monkeypatch.setattr(ns, "_scan_hailo_ollama_group", lambda proc_root="/proc": "G")
    assert ns.shared_group(force_refresh=True) == "G"
    monkeypatch.setenv("PI5_NPU_SHARE", "0")
    assert ns.shared_group(force_refresh=True) is None


@pytest.mark.parametrize("raw,mb", [("3661/7221", 3661), ("3661 / 7221 MB", 3661), (" 157/7221", 157), ("", None), (None, None), ("n/a", None)])
def test_parse_ram_used_mb(raw, mb):
    assert ns.parse_ram_used_mb(raw) == mb


@pytest.fixture()
def reset_env(monkeypatch):
    calls = []
    monkeypatch.setattr(ns, "shared_group", lambda force_refresh=False: "G")
    monkeypatch.setattr(ns, "reset_group", lambda reason, prewarm=None: calls.append(reason) or True)
    monkeypatch.setattr(ns, "_reset_state", {"running": False, "last_time": 0.0, "last": None})
    return calls


def test_maybe_reset_conditions(reset_env, monkeypatch):
    hi = ns.RESET_MB + 100
    assert ns.maybe_reset(ns.RESET_MB - 1, npu_busy=False, llm_idle_sec=None, background=False) is False   # 正常值
    assert ns.maybe_reset(None, npu_busy=False, llm_idle_sec=None, background=False) is False
    assert ns.maybe_reset(hi, npu_busy=True, llm_idle_sec=None, background=False) is False                # 錄音／STT 中
    assert ns.maybe_reset(hi, npu_busy=False, llm_idle_sec=0.0, background=False) is False                # LLM 請求進行中
    assert ns.maybe_reset(hi, npu_busy=False, llm_idle_sec=5.0, background=False) is False                # LLM 剛用完
    assert reset_env == []
    assert ns.maybe_reset(hi, npu_busy=False, llm_idle_sec=600.0, background=False) is True
    assert len(reset_env) == 1 and str(hi) in reset_env[0]
    assert ns.maybe_reset(hi, npu_busy=False, llm_idle_sec=600.0, background=False) is False              # 10 分鐘內只做一次
    monkeypatch.setattr(ns, "shared_group", lambda force_refresh=False: None)
    monkeypatch.setattr(ns, "_reset_state", {"running": False, "last_time": 0.0, "last": None})
    assert ns.maybe_reset(hi, npu_busy=False, llm_idle_sec=None, background=False) is False               # 獨占模式不重置
    assert len(reset_env) == 1


def test_payload_keep_alive_shared_vs_exclusive(monkeypatch):
    msgs = lb.build_messages("hi")
    monkeypatch.setattr(lb, "shared_group", lambda force_refresh=False: None)
    assert lb.HAILO_BACKEND.build_chat_payload(msgs, stream=False)["keep_alive"] == lb.HAILO_KEEP_ALIVE
    assert lb.HAILO_BACKEND.build_chat_payload(msgs, stream=False, hailo_keep_alive=345)["keep_alive"] == 345
    monkeypatch.setattr(lb, "shared_group", lambda force_refresh=False: "G")
    for ka in (None, 0, 345):   # 共用模式一律常駐：卸載會留下約 2.9 GB 不歸還的裝置記憶體
        assert lb.HAILO_BACKEND.build_chat_payload(msgs, stream=False, hailo_keep_alive=ka)["keep_alive"] == ns.SHARED_KEEP_ALIVE
    assert lb.CPU_BACKEND.build_chat_payload(msgs, stream=False)["keep_alive"] == 0


def test_hailo_activity_tracking():
    with lb._hailo_activity(True):
        assert lb.hailo_idle_seconds() == 0.0
    idle = lb.hailo_idle_seconds()
    assert idle is not None and 0.0 <= idle < 5.0
    with lb._hailo_activity(False):          # CPU 後端不計入
        assert lb.hailo_idle_seconds() != 0.0 or lb._hailo_inflight == 0


def test_release_exclusive_only_skips_when_shared():
    w = sw.SttWorker()
    w._engine_group = "G"
    done = threading.Event()
    with w._lock:                            # 正在跑 STT：LLM 前的 release 不可等鎖
        t = threading.Thread(target=lambda: (w.release(exclusive_only=True), done.set()))
        t.start(); t.join(2.0)
    assert done.is_set()
    assert w._proc is None


def test_paused_stops_child_and_blocks_calls():
    w = sw.SttWorker()
    try:
        w._ensure()
        assert w._call("ping", None, 30)["ok"] is True
        with w.paused():
            assert w._proc is None
            blocked = threading.Event()
            t = threading.Thread(target=lambda: (w.release(), blocked.set()))
            t.start(); time.sleep(0.3)
            assert not blocked.is_set()      # 重置期間其他呼叫排隊
        t.join(5.0)
        assert blocked.is_set()
        assert w._call("ping", None, 30)["engine_open"] is False   # 之後自動重啟子行程
        assert w._call("release", {"exclusive_only": True}, 30)["ok"] is True
    finally:
        w.shutdown()
