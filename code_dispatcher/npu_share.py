# -*- coding: utf-8 -*-
"""
npu_share.py — Hailo-10H 共用模式 (Whisper 與 hailo-ollama 同時常駐，2026-10-29)。

hailo-ollama (hailo-gen-ai-model-zoo >= 5.3.0) 以環境變數 HAILO_OLLAMA_VDEVICE_GROUP_ID 啟動時，
其他 HailoRT 行程建立 VDevice 時帶相同 group_id 就能共用 NPU。本模組：
  1) 從執行中 hailo-ollama 行程的環境變數判斷是否為共用模式 (唯一設定來源：hailo-ollama 的 systemd drop-in)；
  2) 閒置時若 NPU RAM 異常偏高，執行「群組重置」把裝置記憶體全部收回。

raspi543 實測 (HailoRT 5.3.0、hailo-ollama 0.5.1)：
  - 兩者常駐：約 3.7 GB / 7.2 GB，Whisper 30 輪 + LLM 30 次、同時推論都持平。
  - 群組內任一方釋放記憶體 (LLM 卸載、hailo-ollama 重啟、stt_worker 被 kill) 時，只要另一方還活著，
    那份裝置記憶體就不歸還：LLM 卸載/重啟後殘留約 2.9 GB (最多一份)，再載入 → 約 6.5 GB；
    每次 Whisper 行程被 kill 殘留約 0.4 GB。
  - 群組內所有行程都結束時才全部歸還 → 重置 = 先結束 stt_worker 子行程，再重啟 hailo-ollama (實測 6470 → 302 MB)。
所以共用模式下 LLM 一律常駐 (不送 keep_alive=0)，並由 NPU 遙測在閒置時檢查、必要時重置。

環境變數：
  PI5_NPU_SHARE=0                 強制停用共用模式 (退回獨占模式：STT 與 LLM 輪流使用 NPU)
  PI5_HAILO_SHARED_KEEP_ALIVE     共用模式下 hailo-ollama 的 keep_alive 秒數 (預設 604800 = 7 天)
  PI5_NPU_SHARED_RESET_MB         閒置時 NPU RAM 超過此值 (MB) 就重置 (預設 4500；兩者常駐的正常值約 3700，
                                  超過代表有殘留：LLM 卸載約 2.9 GB、每次 STT 子行程結束約 0.4 GB)
"""

import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("Pi5NpuShare")

GROUP_ENV = "HAILO_OLLAMA_VDEVICE_GROUP_ID"
SHARED_KEEP_ALIVE = int(os.environ.get("PI5_HAILO_SHARED_KEEP_ALIVE", str(7 * 24 * 3600)))
RESET_MB = int(os.environ.get("PI5_NPU_SHARED_RESET_MB", "4500"))
RESET_MIN_INTERVAL_SEC = 600.0
LLM_IDLE_SEC = 30.0
HAILO_OLLAMA_SERVICE = "hailo-ollama.service"

_GROUP_TTL_SEC = 30.0
_group_cache: Dict[str, Any] = {"time": 0.0, "value": None}
_group_lock = threading.Lock()


def _scan_hailo_ollama_group(proc_root: str = "/proc") -> Optional[str]:
    """回傳執行中 hailo-ollama 行程的 HAILO_OLLAMA_VDEVICE_GROUP_ID；沒有行程或沒設定時回傳 None。"""
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return None
    key = (GROUP_ENV + "=").encode()
    for pid in pids:
        try:
            with open(os.path.join(proc_root, pid, "comm"), "rb") as f:
                if f.read().strip() != b"hailo-ollama":
                    continue
            with open(os.path.join(proc_root, pid, "environ"), "rb") as f:
                env = f.read().split(b"\0")
        except OSError:
            continue
        for kv in env:
            if kv.startswith(key):
                value = kv[len(key):].decode("utf-8", "replace").strip()
                return value or None
        return None
    return None


def shared_group(force_refresh: bool = False) -> Optional[str]:
    """共用模式時回傳 VDevice group_id，否則 None (結果快取 30 秒)。"""
    if os.environ.get("PI5_NPU_SHARE", "1").strip() == "0":
        return None
    now = time.monotonic()
    with _group_lock:
        if force_refresh or now - _group_cache["time"] > _GROUP_TTL_SEC:
            _group_cache["value"] = _scan_hailo_ollama_group()
            _group_cache["time"] = now
        return _group_cache["value"]


def parse_ram_used_mb(ram_usage: Any) -> Optional[int]:
    """hailortcli monitor 的 RAM 欄位 (例如 '3661/7221' 或 '3661 / 7221 MB') → 已用 MB。"""
    if ram_usage is None:
        return None
    m = re.match(r"\s*(\d+)", str(ram_usage))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- 群組重置

_reset_lock = threading.Lock()
_reset_state: Dict[str, Any] = {"running": False, "last_time": 0.0, "last": None}


def reset_status() -> Optional[Dict[str, Any]]:
    return _reset_state["last"]


def _restart_hailo_ollama() -> None:
    subprocess.run(["sudo", "-n", "systemctl", "restart", HAILO_OLLAMA_SERVICE], check=True, timeout=90,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def reset_group(reason: str, prewarm: Optional[Callable[[], Any]] = None) -> bool:
    """先結束 stt_worker 子行程 (期間 STT 呼叫會排隊等候)，再重啟 hailo-ollama，最後重新載入 LLM。"""
    from stt_worker import stt_worker
    t0 = time.time()
    ok, err = True, None
    print(f"[NpuShare] 群組重置開始：{reason}", flush=True)
    try:
        with stt_worker.paused():
            _restart_hailo_ollama()
            for _ in range(30):                      # 等新的 hailo-ollama 行程出現，STT 子行程才會以共用模式重開
                if shared_group(force_refresh=True):
                    break
                time.sleep(0.5)
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    if ok and prewarm is not None:
        try:
            prewarm()
        except Exception as e:
            logger.warning(f"[NpuShare] prewarm after reset failed: {e}")
    _reset_state["last"] = {"time": round(t0, 1), "reason": reason, "ok": ok, "error": err,
                            "duration_s": round(time.time() - t0, 1)}
    print(f"[NpuShare] 群組重置{'完成' if ok else '失敗'} ({time.time() - t0:.1f} s){'' if ok else ' ' + str(err)}", flush=True)
    return ok


def maybe_reset(ram_used_mb: Optional[int], npu_busy: bool, llm_idle_sec: Optional[float],
                prewarm: Optional[Callable[[], Any]] = None, background: bool = True) -> bool:
    """NPU 遙測在閒置時呼叫：共用模式且 NPU RAM >= RESET_MB 時啟動群組重置 (10 分鐘內最多一次)。回傳是否啟動。"""
    if ram_used_mb is None or ram_used_mb < RESET_MB or npu_busy:
        return False
    if llm_idle_sec is not None and llm_idle_sec < LLM_IDLE_SEC:
        return False
    if not shared_group():
        return False
    with _reset_lock:
        now = time.monotonic()
        if _reset_state["running"] or (_reset_state["last_time"] and now - _reset_state["last_time"] < RESET_MIN_INTERVAL_SEC):
            return False
        _reset_state["running"] = True
        _reset_state["last_time"] = now

    def run() -> None:
        try:
            reset_group(f"閒置時 NPU RAM {ram_used_mb} MB >= {RESET_MB} MB", prewarm)
        finally:
            _reset_state["running"] = False

    if background:
        threading.Thread(target=run, daemon=True, name="npu-group-reset").start()
    else:
        run()
    return True


def is_resetting() -> bool:
    return bool(_reset_state["running"])
