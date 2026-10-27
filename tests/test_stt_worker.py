"""stt_worker 子行程協定測試 (不需要 Hailo 硬體：CI 上 NPU 不可用，子行程應回 hailo_unavailable 而不是卡住或崩潰)。"""
import numpy as np
import pytest

import stt_worker as sw


@pytest.fixture()
def worker():
    w = sw.SttWorker()
    yield w
    w.shutdown()


def _hailo_present() -> bool:
    try:
        import hailo_platform  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(_hailo_present(), reason="有 Hailo 的機器上會真的開 NPU，改由 HIL 測試涵蓋")
def test_transcribe_without_npu_reports_unavailable(worker):
    with pytest.raises(RuntimeError, match="hailo_unavailable|ModuleNotFoundError|ImportError"):
        worker.transcribe(np.zeros(16000, dtype=np.float32))
    assert worker._proc is not None and worker._proc.poll() is None      # 子行程仍在待命
    assert worker.is_busy() is False


def test_ping_release_and_restart_after_crash(worker):
    worker._ensure()
    assert worker._call("ping", None, 30)["ok"] is True
    worker.release()                                                       # 沒開 NPU 時也要能正常回應
    worker._proc.kill(); worker._proc.wait()
    assert worker._call("ping", None, 30)["engine_open"] is False         # 自動重啟


def test_release_without_process_is_noop():
    w = sw.SttWorker()
    w.release()
    assert w._proc is None


def test_frame_roundtrip(tmp_path):
    import os
    r, wfd = os.pipe()
    with os.fdopen(wfd, "wb") as f:
        sw._write_frame(f, ("transcribe", (np.arange(5, dtype=np.float32), "zh")))
    cmd, (arr, lang) = sw._read_frame(r, None)
    os.close(r)
    assert cmd == "transcribe" and lang == "zh" and arr.tolist() == [0, 1, 2, 3, 4]
