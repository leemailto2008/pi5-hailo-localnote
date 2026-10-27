# -*- coding: utf-8 -*-
"""
stt_worker.py — Hailo-10H Whisper 語音轉文字 (STT) 獨立子行程 (2026-10-27，方案 B)。

為什麼要獨立行程：
  py-spy 實測錄音中 uvicorn 行程約 80% 時間有 STT 執行緒持有 GIL：
  (1) 每段都重新建立 VDevice + Speech2Text (載入 Whisper HEF)；(2) HailoRT 的 generate_all_text 執行期間不釋放 GIL。
  結果 /api/status 最長 18 s 無回應、網頁卡頓。GIL 是每個行程各一把，推論移到子行程後 uvicorn 不再被卡住。

設計：
  - 推論仍在 Hailo-10H NPU 上執行；子行程只負責持有 VDevice 與 Speech2Text。
  - 子行程常駐待命 (不佔 NPU)。第一次收到音訊才開 NPU 並載入 Whisper，整段錄音共用同一個引擎。
  - 閒置 IDLE_RELEASE_SEC 秒，或主行程要用 Hailo LLM 前呼叫 release()，就釋放 NPU 給 hailo-ollama
    (Hailo-10H 的 VDevice 一次只能給一個行程)。
  - NPU 開不起來時回傳 error，由 voice_pipeline 走原本的 CPU 備援流程 (子行程內不做 CPU 推論)。
  - 子行程異常結束時，下一次呼叫自動重啟。

協定：stdin/stdout 上的 [4 bytes big-endian 長度][pickle] 訊框。子行程的 print/log 全部導到 stderr (進 journal)。
"""

import os
import pickle
import select
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

IDLE_RELEASE_SEC = float(os.environ.get("PI5_STT_IDLE_RELEASE_SEC", "10"))
_HERE = os.path.dirname(os.path.abspath(__file__))


def _write_frame(f, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(struct.pack(">I", len(data)) + data)
    f.flush()


def _read_exact(fd: int, n: int, deadline: Optional[float]) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("stt worker reply timeout")
            r, _, _ = select.select([fd], [], [], remaining)
            if not r:
                raise TimeoutError("stt worker reply timeout")
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            raise EOFError("stt worker closed the pipe")
        buf += chunk
    return bytes(buf)


def _read_frame(fd: int, deadline: Optional[float] = None) -> Any:
    (size,) = struct.unpack(">I", _read_exact(fd, 4, deadline))
    return pickle.loads(_read_exact(fd, size, deadline))


# --------------------------------------------------------------------------- 子行程


def _child_main() -> None:
    proto_out = os.fdopen(os.dup(1), "wb")   # 協定專用的 stdout
    os.dup2(2, 1)                            # 其餘 print 一律寫到 stderr，避免汙染協定
    sys.stdout = sys.stderr
    in_fd = sys.stdin.fileno()
    for p in (_HERE, os.path.dirname(_HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from hailo_whisper.engine import HailoWhisperEngine

    engine = None
    last_use = time.monotonic()

    def release() -> None:
        nonlocal engine
        if engine is None:
            return
        for obj in (getattr(engine, "_s2t", None), getattr(engine, "_vdevice", None)):
            try:
                if obj is not None:
                    obj.release()
            except Exception as e:
                print(f"[SttWorker] release warning: {e}", flush=True)
        engine = None
        print("[SttWorker] Hailo NPU released", flush=True)

    print(f"[SttWorker] ready (pid {os.getpid()})", flush=True)
    while True:
        timeout = IDLE_RELEASE_SEC if engine is not None else None
        r, _, _ = select.select([in_fd], [], [], timeout)
        if not r:
            if engine is not None and time.monotonic() - last_use >= IDLE_RELEASE_SEC:
                release()
            continue
        try:
            cmd, payload = _read_frame(in_fd)
        except EOFError:
            break
        if cmd == "exit":
            release()
            _write_frame(proto_out, {"ok": True})
            break
        if cmd == "release":
            release()
            _write_frame(proto_out, {"ok": True})
            continue
        if cmd == "ping":
            _write_frame(proto_out, {"ok": True, "engine_open": engine is not None})
            continue
        if cmd == "transcribe":
            audio, language = payload
            try:
                if engine is None:
                    t0 = time.perf_counter()
                    engine = HailoWhisperEngine()
                    if not engine.is_hardware_accelerated():
                        release()
                        _write_frame(proto_out, {"ok": False, "error": "hailo_unavailable"})
                        continue
                    print(f"[SttWorker] Hailo NPU opened, Whisper loaded ({(time.perf_counter() - t0) * 1000:.0f} ms)", flush=True)
                res = engine.transcribe(audio, language=language, initial_prompt=None, temperature=0.0)
                _write_frame(proto_out, {"ok": True, "result": res})
            except Exception as e:
                release()
                _write_frame(proto_out, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            last_use = time.monotonic()
            continue
        _write_frame(proto_out, {"ok": False, "error": f"unknown command {cmd!r}"})


# --------------------------------------------------------------------------- 主行程端


class SttWorker:
    """主行程端代理：所有呼叫互斥 (一次一段音訊)，子行程掛掉會自動重啟。"""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._busy = False

    def is_busy(self) -> bool:
        return self._busy

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__), "--child"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, cwd=_HERE)
        return self._proc

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
        self._proc = None

    def _call(self, cmd: str, payload: Any, timeout: float) -> Dict[str, Any]:
        proc = self._ensure()
        try:
            _write_frame(proc.stdin, (cmd, payload))
            return _read_frame(proc.stdout.fileno(), time.monotonic() + timeout)
        except (TimeoutError, EOFError, BrokenPipeError, OSError) as e:
            self._kill()   # 狀態不明 → 砍掉，下次重啟 (子行程結束時 VDevice 隨之釋放)
            raise RuntimeError(f"stt worker {cmd} failed: {e}") from e

    def transcribe(self, audio: Any, language: Optional[str] = None) -> Dict[str, Any]:
        """回傳 HailoWhisperEngine.transcribe() 的 dict；NPU 不可用或失敗時 raise RuntimeError。"""
        try:
            import numpy as np
            if isinstance(audio, np.ndarray):
                timeout = 60.0 + 1.5 * len(audio) / 16000.0
            elif isinstance(audio, bytes):
                timeout = 60.0 + 1.5 * len(audio) / 32000.0
            else:
                timeout = 4 * 3600.0 + 600.0   # 檔案路徑 (整段母帶，最長 4 小時) 長度未知
        except Exception:
            timeout = 4 * 3600.0 + 600.0
        with self._lock:
            self._busy = True
            try:
                reply = self._call("transcribe", (audio, language), timeout)
            finally:
                self._busy = False
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or "stt worker error")
        return reply["result"]

    def release(self, timeout: float = 15.0) -> None:
        """釋放 NPU (子行程保留待命)。沒有子行程時什麼都不做。"""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return
            try:
                self._call("release", None, timeout)
            except RuntimeError as e:
                print(f"[SttWorker] release: {e}")

    def shutdown(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._call("exit", None, 10.0)
                except RuntimeError:
                    pass
            self._kill()


stt_worker = SttWorker()


if __name__ == "__main__" and "--child" in sys.argv:
    _child_main()
