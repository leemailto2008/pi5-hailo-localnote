# f:\12_prj_raspi5\code_dispatcher\main.py
"""
Raspberry Pi 5 Code Dispatcher & Task Gateway (FastAPI)
Exposes REST endpoints for pushing code analysis tasks to Redis Queue.
"""

import os
import sys
import subprocess
import logging
import re
import time
import json
import asyncio
import socket
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime

SYSTEM_HOSTNAME = socket.gethostname()
DEFAULT_HOTSPOT_SSID = f"{SYSTEM_HOSTNAME}_AI" if SYSTEM_HOSTNAME not in ("raspberrypi", "localhost") else "raspi543_AI"

logger = logging.getLogger("pi5-dispatcher")

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

try:
    from redis import Redis
    from rq import Queue
    from rq.job import Job
except ImportError:
    class DummyRedis:
        def ping(self):
            return False

    class DummyRegistry:
        count = 0

    class DummyQueue:
        def enqueue(self, *args, **kwargs):
            raise RuntimeError("Redis/RQ is not installed on this environment.")
        def __len__(self):
            return 0
        started_job_registry = DummyRegistry()
        finished_job_registry = DummyRegistry()
        failed_job_registry = DummyRegistry()

    Redis = lambda **kw: DummyRedis()
    Queue = lambda *args, **kw: DummyQueue()
    Job = None


# Ensure PipeWire / PulseAudio session is accessible for systemd service
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
os.environ.setdefault("PULSE_SERVER", "unix:/run/user/1000/pulse/native")

# Ensure code_dispatcher directory has absolute highest priority in sys.path
_code_dispatcher_dir = os.path.dirname(os.path.abspath(__file__))
if _code_dispatcher_dir in sys.path:
    sys.path.remove(_code_dispatcher_dir)
sys.path.insert(0, _code_dispatcher_dir)

agent_candidate_dirs = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "telegram_agent")),
    "/home/pi/pi5-llm-arena/telegram_agent",
    os.path.expanduser("~/pi5-llm-arena/telegram_agent"),
    r"f:\12_prj_raspi5\telegram_agent"
]
for d in agent_candidate_dirs:
    if os.path.exists(d) and d not in sys.path:
        sys.path.append(d)

from tasks import analyze_code_task

QUEUE_NAME = "code_review_queue"
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379

app = FastAPI(
    title="Pi 5 LLM Code Dispatcher API",
    description="High-performance asynchronous code review and inference queue powered by FastAPI & Redis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
def _shutdown_stt_worker():
    """關閉 STT 子行程 (會先釋放 Hailo NPU)。"""
    try:
        from stt_worker import stt_worker
        stt_worker.shutdown()
    except Exception as e:
        print(f"[Shutdown] stt worker: {e}")


@app.on_event("startup")
async def startup_event():
    import asyncio
    try:
        from voice_pipeline import get_whisper_model
        print("[Startup] Pre-warming faster-whisper model in background...")
        asyncio.create_task(asyncio.to_thread(get_whisper_model))
    except Exception as e:
        print(f"[Startup] Whisper preload notice: {e}")

    try:
        def prewarm_ollama():
            from llm_backend import prewarm as llm_prewarm
            print("[Startup] Pre-warming LLM backend (Hailo-10H qwen3:1.7b preferred, CPU Ollama fallback)...")
            backend = llm_prewarm()
            if backend is not None:
                print(f"[Startup] LLM backend ready: {backend.label}")
            else:
                print("[Startup] LLM prewarm failed on every backend; will retry lazily on first request.")
        asyncio.create_task(asyncio.to_thread(prewarm_ollama))
    except Exception as e:
        print(f"[Startup] Ollama preload notice: {e}")

    try:
        from meeting_minutes import minutes_manager
        asyncio.create_task(minutes_manager.offline_detector.start())
        print("[Startup] OfflineDetector background watchdog started.")
    except Exception as e:
        print(f"[Startup] OfflineDetector startup notice: {e}")

    try:
        asyncio.create_task(ws_telemetry_loop())
        print("[Startup] WebSocket telemetry background broadcaster started.")
    except Exception as e:
        print(f"[Startup] WebSocket telemetry task error: {e}")

    try:
        asyncio.create_task(bt_auto_reconnect_loop())
        print("[Startup] Bluetooth auto-reconnect background daemon started.")
    except Exception as e:
        print(f"[Startup] Bluetooth auto-reconnect task error: {e}")

    try:
        async def start_port_8000_proxy():
            async def handle_client(reader, writer):
                try:
                    remote_reader, remote_writer = await asyncio.open_connection("127.0.0.1", 80)
                    async def forward(r, w):
                        try:
                            while True:
                                data = await r.read(16384)
                                if not data:
                                    break
                                w.write(data)
                                await w.drain()
                        except Exception:
                            pass
                        finally:
                            try:
                                w.close()
                                await w.wait_closed()
                            except Exception:
                                pass
                    await asyncio.gather(forward(reader, remote_writer), forward(remote_reader, writer))
                except Exception:
                    try:
                        writer.close()
                    except Exception:
                        pass

            try:
                server = await asyncio.start_server(handle_client, "0.0.0.0", 8000)
                print("[Startup] Port 8000 -> Port 80 TCP proxy started successfully on 0.0.0.0:8000.")
            except Exception as e:
                print(f"[Startup] Port 8000 proxy notice: {e}")

        asyncio.create_task(start_port_8000_proxy())
    except Exception as e:
        print(f"[Startup] Error launching port 8000 proxy: {e}")

redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
task_queue = Queue(QUEUE_NAME, connection=redis_conn)


def _queue_len() -> int:
    """redis 套件已安裝但 redis-server 未執行時 len(task_queue) 會丟 ConnectionError；/health 不應因此回 500。"""
    try:
        return len(task_queue)
    except Exception:
        return 0

HTML_CLIENT_UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_client.html")
HTML_UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
HTML_DBG01_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_dbg01.html")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


class CodeTaskRequest(BaseModel):
    code: Optional[str] = Field(default="", description="Source code snippet or file content to review")
    prompt: Optional[str] = Field(
        default="Perform a thorough code review, identifying potential bugs, security vulnerabilities, edge cases, and performance optimizations.",
        description="Custom instruction or review focus"
    )
    model: Optional[str] = Field(default="qwen2.5-coder:7b", description="Ollama model to use")
    language: Optional[str] = Field(default="python", description="Programming language of the code")
    filename: Optional[str] = Field(default="snippet.py", description="Source filename")
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=1500, ge=50, le=4096, description="Max generated tokens")


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def get_web_ui():
    """Serve the client-facing LocalNote Meeting Minutes Dashboard."""
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    if os.path.exists(HTML_CLIENT_UI_PATH):
        with open(HTML_CLIENT_UI_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers=headers)
    if os.path.exists(HTML_UI_PATH):
        with open(HTML_UI_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers=headers)
    return HTMLResponse(content="<h1>LocalNote 聽記盒</h1><p>ui_client.html not found.</p>", headers=headers)


@app.get("/dbg01", response_class=HTMLResponse, tags=["Web UI"])
@app.get("/dgb01", response_class=HTMLResponse, include_in_schema=False)
@app.get("/arena", response_class=HTMLResponse, tags=["Web UI"])
def get_dbg01_ui():
    """Serve the original Code Review & LLM Arena interface at /dbg01 and /arena."""
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    if os.path.exists(HTML_DBG01_PATH):
        with open(HTML_DBG01_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers=headers)
    return HTMLResponse(content="<h1>Pi 5 LLM Arena (DBG01)</h1><p>ui_dbg01.html not found.</p>", headers=headers)


@app.get("/dbg02", response_class=HTMLResponse, tags=["Web UI"])
@app.get("/dgb02", response_class=HTMLResponse, include_in_schema=False)
def get_dbg02_ui():
    """Serve the Engineering Bluetooth Gateway & Voice Application Studio at /dbg02."""
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    if os.path.exists(HTML_UI_PATH):
        with open(HTML_UI_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), headers=headers)
    return HTMLResponse(content="<h1>Pi 5 Studio (DBG02)</h1><p>ui.html not found.</p>", headers=headers)


@app.get("/bluetooth", response_class=HTMLResponse, tags=["Web UI"])
@app.get("/bt", response_class=HTMLResponse, include_in_schema=False)
def get_bluetooth_ui():
    """Serve the LocalNote client interface directly on the Bluetooth Management view."""
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    if os.path.exists(HTML_CLIENT_UI_PATH):
        with open(HTML_CLIENT_UI_PATH, "r", encoding="utf-8") as f:
            html = f.read()
            html = html.replace('id="viewBt" hidden', 'id="viewBt"').replace('id="viewMain"', 'id="viewMain" hidden')
            return HTMLResponse(content=html, headers=headers)
    return get_web_ui()


# -------------------------------------------------------------
# Task 10: Progressive Web App (PWA) Manifest & Service Worker
# -------------------------------------------------------------
@app.get("/manifest.json", include_in_schema=False)
def get_pwa_manifest():
    """Serve PWA Web App Manifest."""
    manifest_path = os.path.join(STATIC_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/manifest+json")
    return Response(content="{}", media_type="application/manifest+json")


@app.get("/sw.js", include_in_schema=False)
@app.get("/service-worker.js", include_in_schema=False)
def get_pwa_service_worker():
    """Serve PWA Service Worker with root scope permissions and no-cache policy."""
    sw_path = os.path.join(STATIC_DIR, "sw.js")
    headers = {
        "Service-Worker-Allowed": "/",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    if os.path.exists(sw_path):
        return FileResponse(sw_path, media_type="application/javascript", headers=headers)
    return Response(content="// Service worker not found", media_type="application/javascript", headers=headers)
class TaskEnqueueResponse(BaseModel):
    task_id: str
    status: str
    queue_name: str
    position_in_queue: int
    model: str
    filename: str
    submitted_at: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.get("/health", tags=["Monitoring"])
def get_system_health() -> Dict[str, Any]:
    """Retrieve Pi 5 hardware telemetry and services health."""
    redis_ok = False
    try:
        redis_ok = redis_conn.ping()
    except Exception:
        pass

    ollama_ok = False
    active_models: List[str] = []
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            ollama_ok = True
            active_models = [m.get("name") for m in resp.json().get("models", [])]
    except Exception:
        pass

    temp_c = 0.0
    try:
        res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2)
        temp_c = float(res.stdout.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        pass

    free_ram_mb = 0
    total_ram_mb = 0
    try:
        res = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=2)
        for line in res.stdout.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                total_ram_mb = int(parts[1])
                free_ram_mb = int(parts[6])  # available RAM
    except Exception:
        pass

    llm_backend_label = "n/a"
    try:
        from llm_backend import get_backend
        llm_backend_label = get_backend().label
    except Exception as e:
        print(f"[Health] llm backend probe failed: {e}")

    return {
        "status": "healthy" if (redis_ok and ollama_ok) else "degraded",
        "redis_connected": redis_ok,
        "ollama_running": ollama_ok,
        "llm_backend": llm_backend_label,
        "available_models_count": len(active_models),
        "cpu_temperature_c": temp_c,
        "ram_available_mb": free_ram_mb,
        "ram_total_mb": total_ram_mb,
        "queue_length": _queue_len() if redis_ok else 0,
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/api/models", tags=["Models"])
def list_models() -> Dict[str, Any]:
    """List available LLM models in local Ollama instance."""
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to query Ollama models: {str(e)}"
        )


@app.post("/api/tasks", response_model=TaskEnqueueResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Tasks"])
def create_task(req: CodeTaskRequest) -> TaskEnqueueResponse:
    """Submit a new code review task to the Redis Queue."""
    task_payload = req.model_dump()
    job = task_queue.enqueue(
        analyze_code_task,
        task_payload,
        job_timeout=600,
        result_ttl=86400  # Keep result in Redis for 24 hours
    )

    return TaskEnqueueResponse(
        task_id=job.id,
        status=job.get_status() or "queued",
        queue_name=QUEUE_NAME,
        position_in_queue=len(task_queue),
        model=req.model,
        filename=req.filename,
        submitted_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
def get_task_status(task_id: str) -> TaskStatusResponse:
    """Retrieve execution status and generated results of a submitted task."""
    try:
        job = Job.fetch(task_id, connection=redis_conn)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task ID '{task_id}' not found in Redis."
        )

    job_status = job.get_status()
    created_at = job.created_at.strftime("%Y-%m-%d %H:%M:%S") if job.created_at else None
    started_at = job.started_at.strftime("%Y-%m-%d %H:%M:%S") if job.started_at else None
    ended_at = job.ended_at.strftime("%Y-%m-%d %H:%M:%S") if job.ended_at else None

    return TaskStatusResponse(
        task_id=task_id,
        status=job_status,
        created_at=created_at,
        started_at=started_at,
        ended_at=ended_at,
        result=job.result if job.is_finished else None,
        error=str(job.exc_info) if job.is_failed else None
    )


@app.get("/api/queue", tags=["Monitoring"])
def get_queue_status() -> Dict[str, Any]:
    """Inspect queue depth and active jobs."""
    try:
        return {
            "queue_name": QUEUE_NAME,
            "queued_jobs_count": len(task_queue),
            "started_jobs_count": task_queue.started_job_registry.count,
            "finished_jobs_count": task_queue.finished_job_registry.count,
            "failed_jobs_count": task_queue.failed_job_registry.count
        }
    except Exception as e:   # redis 套件在、但 redis-server 沒跑 (LocalNote 不需要 redis)
        raise HTTPException(status_code=503, detail=f"任務佇列無法使用 (redis): {e}")


class NetworkSwitchRequest(BaseModel):
    target_mode: str = Field(..., description="'ap' for standalone hotspot (10.20.0.1) or 'sta' for home Wi-Fi")


@app.get("/api/network/status", tags=["Network"])
def get_network_status() -> Dict[str, Any]:
    """Get active Wi-Fi profile, IP address, and operational mode."""
    mode = "unknown"
    active_conn = "none"
    current_ip = "0.0.0.0"

    try:
        # Query active connection for wlan0
        res = subprocess.run(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device"], capture_output=True, text=True, timeout=3)
        for line in res.stdout.splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 4 and parts[0] == "wlan0":
                active_conn = parts[3]
                if "Hotspot" in active_conn or "ap" in active_conn.lower():
                    mode = "ap"
                elif active_conn != "--" and len(active_conn) > 0:
                    mode = "sta"
                break
    except Exception as e:
        pass

    try:
        # Query IP on wlan0
        ip_res = subprocess.run(["ip", "-4", "addr", "show", "wlan0"], capture_output=True, text=True, timeout=3)
        for line in ip_res.stdout.splitlines():
            line = line.strip()
            if line.startswith("inet "):
                current_ip = line.split()[1].split("/")[0]
                break
    except Exception:
        pass

    return {
        "mode": mode,
        "active_connection": active_conn,
        "current_ip": current_ip,
        "hostname": SYSTEM_HOSTNAME,
        "hotspot_gateway_ip": "10.20.0.1",
        "hotspot_ssid": DEFAULT_HOTSPOT_SSID,
        "sta_ssid": "My_5G_Guest1"
    }


@app.post("/api/network/switch", tags=["Network"])
def switch_network_mode(req: NetworkSwitchRequest) -> Dict[str, Any]:
    """Switch Raspberry Pi 5 Wi-Fi mode between Client (STA) and Hotspot (AP) with Watchdog Fail-Safe."""
    import hotspot_controller

    target = req.target_mode.lower()
    if target not in ["ap", "sta"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="target_mode must be 'ap' (hotspot) or 'sta' (client WiFi)"
        )

    profile_name = f"Pi5-Hotspot ({DEFAULT_HOTSPOT_SSID})" if target == "ap" else "My_5G_Guest1"
    new_ip = "10.20.0.1" if target == "ap" else "192.168.50.228"

    if target == "ap":
        hotspot_controller.switch_to_ap(timeout_sec=180)
    else:
        hotspot_controller.switch_to_sta()

    return {
        "status": "success",
        "switched_to": target,
        "profile": profile_name,
        "target_ip": new_ip,
        "watchdog_timeout_sec": 180 if target == "ap" else 0,
        "message": f"切換指令已安全啟動！Pi 5 即將切換至 {profile_name} (IP: {new_ip})。\n🛡️ 防失聯看門狗已啟動：若 3 分鐘內未有連線將自動復歸家用 Wi-Fi！"
    }


@app.post("/api/network/keepalive", tags=["Network"])
def keepalive_network() -> Dict[str, Any]:
    """Signal watchdog that Web UI client is actively browsing, extending AP session."""
    import hotspot_controller
    hotspot_controller.touch_keepalive()
    return {"status": "ok", "message": "Watchdog keepalive touched"}


@app.get("/manifest.json", tags=["PWA"])
def get_pwa_manifest() -> Dict[str, Any]:
    """PWA Web Manifest for iOS/Android Add to Home Screen."""
    return {
        "name": "Pi 5 Edge AI Arena",
        "short_name": "Pi5 Arena",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#090d16",
        "theme_color": "#4f46e5",
        "description": "Raspberry Pi 5 Local LLM Arena & Code Gateway",
        "icons": [
            {
                "src": "https://img.icons8.com/color/192/raspberry-pi.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "https://img.icons8.com/color/512/raspberry-pi.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }


# =============================================================================
# Bluetooth & Voice Application API Endpoints
# =============================================================================

# Add telegram_agent to sys.path for direct access to bt & voice modules
_possible_agent_dirs = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "telegram_agent"),
    "/home/pi/pi5-llm-arena/telegram_agent",
    "/home/pi/telegram_agent"
]
for d in _possible_agent_dirs:
    if os.path.exists(d) and d not in sys.path:
        sys.path.append(d)

from voice_memory import get_in_memory_cache_dir, evict_audio_ring_buffer, get_cache_stats

AUDIO_CACHE_DIR = get_in_memory_cache_dir()
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
print(f"[Main] In-Memory Audio Cache initialized at: {AUDIO_CACHE_DIR}")


class BTPairRequest(BaseModel):
    target: str = Field(..., description="Device MAC address, list index, or name")


class BTConnectRequest(BaseModel):
    target: Optional[str] = Field(default=None, description="Device MAC address, or None for default")


class BTAutoConnectRequest(BaseModel):
    mac: str = Field(..., description="Device MAC address")
    auto_connect: bool = Field(..., description="Whether to auto-connect to this device")
    set_default: Optional[bool] = Field(default=False, description="Whether to set as primary default device")


class BTRemoveRequest(BaseModel):
    mac: str = Field(..., description="Device MAC address to untrust and remove")


class VoiceRecordRequest(BaseModel):
    duration_sec: int = Field(default=5, ge=2, le=120, description="Recording duration in seconds")


class VoiceTranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    direction: str = Field(default="zh2en", description="'zh2en' or 'en2zh'")


class VoiceTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(default="zh-TW-HsiaoChenNeural", description="Neural TTS Voice")
    play_on_speaker: Optional[bool] = Field(default=True, description="Whether to play on Bluetooth speaker directly")


_last_bt_status_cache: Dict[str, Any] = {}
_last_bt_status_time: float = 0.0


@app.get("/api/bt/status", tags=["Bluetooth"])
def get_bt_status() -> Dict[str, Any]:
    """Retrieve Bluetooth controller and paired audio devices status."""
    global _last_bt_status_cache, _last_bt_status_time
    import time
    import re
    from voice_streaming import is_any_recording_active

    now = time.time()
    # 錄音進行中或快取未過期 (3秒內)：直接回傳快取，嚴禁執行任何藍牙終端子進程干擾 SCO 語音
    if is_any_recording_active() and _last_bt_status_cache:
        return _last_bt_status_cache

    if (now - _last_bt_status_time < 3.0) and _last_bt_status_cache:
        return _last_bt_status_cache

    try:
        from bluetooth_manager import get_full_bt_status, load_bt_config
        status_text = get_full_bt_status()
        config = load_bt_config()

        # 1. 檢測真實連線中的藍牙設備 (優先查詢 devices Connected，避免誤判離線設備)
        connected_mac = None
        connected_name = None
        battery_level = None
        rssi = None
        rssi_kind = None  # "dbm" (BlueZ RSSI property) | "golden_delta" (HCI Read_RSSI: dB relative to Golden Receive Power Range)
        signal_level_pct = 0
        link_quality = 0

        res_conn = subprocess.run(["bluetoothctl", "devices", "Connected"], capture_output=True, text=True, timeout=3)
        for line in res_conn.stdout.strip().split("\n"):
            m_c = re.match(r"^Device\s+([0-9A-Fa-f:]{17})\s*(.*)$", line.strip())
            if m_c:
                connected_mac = m_c.group(1).upper()
                connected_name = m_c.group(2).strip() or None
                break

        # 備用檢查：若 devices Connected 未列出，查詢 bluetoothctl info 確認是否包含 Connected: yes
        if not connected_mac:
            res_info_any = subprocess.run(["bluetoothctl", "info"], capture_output=True, text=True, timeout=3)
            if "Connected: yes" in res_info_any.stdout:
                m_dev = re.search(r"Device\s+([0-9A-Fa-f:]{17})", res_info_any.stdout)
                if m_dev:
                    connected_mac = m_dev.group(1).upper()
                    m_name = re.search(r"Name:\s*(.+)", res_info_any.stdout)
                    if m_name:
                        connected_name = m_name.group(1).strip()

        # 2. 若有設備已連線，針對該設備提取電池電量與訊號強度
        if connected_mac:
            res_dev_info = subprocess.run(["bluetoothctl", "info", connected_mac], capture_output=True, text=True, timeout=3)
            if not connected_name:
                m_n = re.search(r"Name:\s*(.+)", res_dev_info.stdout)
                if m_n:
                    connected_name = m_n.group(1).strip()

            # 電量解析 (支援 0x.. (85) 與純數字 (85))
            m_bat = re.search(r"Battery Percentage:\s*.*\((\d+)\)", res_dev_info.stdout) or re.search(r"Battery Percentage:\s*(\d+)", res_dev_info.stdout)
            if m_bat:
                battery_level = int(m_bat.group(1))

            # RSSI 訊號強度解析
            m_rssi = re.search(r"RSSI:\s*(-?\d+)", res_dev_info.stdout)
            if m_rssi:
                rssi = int(m_rssi.group(1))
                rssi_kind = "dbm"

            # 錄音未進行時，若 BlueZ 未回報 RSSI 且具備 hcitool，嘗試安全獲取
            if rssi is None and not is_any_recording_active():
                try:
                    res_hcitool = subprocess.run(["hcitool", "rssi", connected_mac], capture_output=True, text=True, timeout=2)
                    m_h_rssi = re.search(r"RSSI return value:\s*(-?\d+)", res_hcitool.stdout)
                    if m_h_rssi:
                        rssi = int(m_h_rssi.group(1))
                        # NOTE: for BR/EDR links HCI Read_RSSI is NOT dBm. It is the offset from the
                        # Golden Receive Power Range: 0 = inside the range (ideal), negative = dB below it.
                        rssi_kind = "golden_delta"
                except Exception:
                    pass

            # 計算訊號百分比：dBm 依 -100 ~ -40 轉換；golden_delta 以 0 為滿分、每低 1 dB 扣 3%
            if rssi is not None and rssi_kind == "dbm":
                signal_level_pct = max(10, min(100, int((rssi + 100) * 1.8)))
            elif rssi is not None and rssi_kind == "golden_delta":
                signal_level_pct = max(5, min(100, 100 + rssi * 3)) if rssi <= 0 else 100
            else:
                signal_level_pct = 90
            link_quality = int(signal_level_pct * 2.55)

        state_hash = f"{connected_mac}_{connected_name}_{battery_level}_{rssi}"

        # 3. 取得已信任設備清單
        trusted_devices = []
        try:
            res_trusted = subprocess.run(["bluetoothctl", "devices", "Trusted"], capture_output=True, text=True, timeout=3)
            auto_map = config.get("auto_connect_devices", {})
            for line in res_trusted.stdout.strip().split("\n"):
                m_dev = re.match(r"^Device\s+([0-9A-Fa-f:]{17})\s+(.*)$", line.strip())
                if m_dev:
                    d_mac = m_dev.group(1).upper()
                    d_name = m_dev.group(2).strip()
                    is_def = (d_mac == (config.get("default_mac") or "").upper())
                    is_conn = (d_mac == (connected_mac or "").upper())
                    auto_conn = auto_map.get(d_mac, is_def)
                    trusted_devices.append({
                        "mac": d_mac,
                        "name": d_name,
                        "connected": is_conn,
                        "is_default": is_def,
                        "auto_connect": auto_conn,
                        "battery_level": battery_level if is_conn else None,
                        "rssi": rssi if is_conn else None,
                        "rssi_kind": rssi_kind if is_conn else None
                    })
        except Exception as e_t:
            logger.warning(f"Error fetching trusted devices: {e_t}")

        res_data = {
            "status_text": status_text,
            "connected": connected_mac is not None,
            "connected_mac": connected_mac,
            "connected_name": connected_name,
            "battery_level": battery_level,
            "link_quality": link_quality,
            "signal_level_pct": signal_level_pct,
            "rssi": rssi,
            "rssi_kind": rssi_kind,
            "state_hash": state_hash,
            "default_mac": config.get("default_mac"),
            "default_name": config.get("default_name"),
            "trusted_devices": trusted_devices
        }
        _last_bt_status_cache = res_data
        _last_bt_status_time = now
        return res_data
    except Exception as e:
        if _last_bt_status_cache:
            return _last_bt_status_cache
        return {
            "status_text": f"藍牙狀態獲取失敗: {str(e)}",
            "connected": False,
            "state_hash": f"err_{str(e)}",
            "trusted_devices": [],
            "error": str(e)
        }


@app.post("/api/bt/scan", tags=["Bluetooth"])
def scan_bt_devices(timeout_sec: int = 10) -> Dict[str, Any]:
    """Scan for nearby Bluetooth audio devices."""
    try:
        from bluetooth_manager import scan_devices
        devices = scan_devices(timeout_sec=timeout_sec)
        return {
            "status": "success",
            "ok": True,
            "msg": f"找到 {len(devices)} 個可用藍牙音訊設備",
            "count": len(devices),
            "devices": devices
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"藍牙掃描失敗: {str(e)}")


@app.post("/api/bt/pair", tags=["Bluetooth"])
def pair_bt_device(req: BTPairRequest) -> Dict[str, Any]:
    """Pair and trust a target Bluetooth headset."""
    try:
        from bluetooth_manager import pair_and_trust
        ok, msg = pair_and_trust(req.target)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {"status": "success", "message": msg}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"配對失敗: {str(e)}")


@app.post("/api/bt/trusted/auto_connect", tags=["Bluetooth"])
def set_bt_device_auto_connect(req: BTAutoConnectRequest) -> Dict[str, Any]:
    """Toggle auto-connect setting or set default for a trusted Bluetooth device."""
    try:
        from bluetooth_manager import load_bt_config, save_bt_config
        config = load_bt_config()
        if "auto_connect_devices" not in config or not isinstance(config["auto_connect_devices"], dict):
            config["auto_connect_devices"] = {}

        target_mac = req.mac.upper()
        config["auto_connect_devices"][target_mac] = req.auto_connect

        if req.set_default:
            config["default_mac"] = target_mac
            # Resolve name
            res_info = subprocess.run(["bluetoothctl", "info", target_mac], capture_output=True, text=True, timeout=3)
            m = re.search(r"Name:\s*(.+)", res_info.stdout)
            if m:
                config["default_name"] = m.group(1).strip()

        save_bt_config(config)

        # Invalidate status cache
        global _last_bt_status_time, _last_bt_status_cache
        _last_bt_status_time = 0.0
        _last_bt_status_cache = {}

        return {
            "status": "success",
            "message": f"已更新 {target_mac} 自動連線設定為 {'啟用' if req.auto_connect else '停用'}",
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設定失敗: {str(e)}")


@app.post("/api/bt/remove", tags=["Bluetooth"])
@app.post("/api/bt/untrust", tags=["Bluetooth"], include_in_schema=False)
def remove_bt_device(req: BTRemoveRequest) -> Dict[str, Any]:
    """Untrust and remove a paired Bluetooth device."""
    try:
        from bluetooth_manager import load_bt_config, save_bt_config
        target_mac = req.mac.upper()

        # 1. If currently connected, disconnect first
        subprocess.run(["bluetoothctl", "disconnect", target_mac], capture_output=True, text=True, timeout=5)

        # 2. Untrust & Remove from BlueZ
        subprocess.run(["bluetoothctl", "untrust", target_mac], capture_output=True, text=True, timeout=5)
        res_rm = subprocess.run(["bluetoothctl", "remove", target_mac], capture_output=True, text=True, timeout=5)

        # 3. Clean up config (auto_connect_devices & default_mac)
        config = load_bt_config()
        if "auto_connect_devices" in config and isinstance(config["auto_connect_devices"], dict):
            if target_mac in config["auto_connect_devices"]:
                del config["auto_connect_devices"][target_mac]
        if (config.get("default_mac") or "").upper() == target_mac:
            config["default_mac"] = None
            config["default_name"] = None
        save_bt_config(config)

        # Invalidate status cache
        global _last_bt_status_time, _last_bt_status_cache
        _last_bt_status_time = 0.0
        _last_bt_status_cache = {}

        return {
            "status": "success",
            "ok": True,
            "message": f"已成功移除耳麥設備 {target_mac}",
            "mac": target_mac
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移除設備失敗: {str(e)}")


@app.post("/api/bt/connect", tags=["Bluetooth"])
def connect_bt_device(req: Optional[BTConnectRequest] = None) -> Dict[str, Any]:
    """Connect to default or specified Bluetooth headset, automatically disconnecting any existing device first."""
    try:
        import time
        from bluetooth_manager import load_bt_config, _switch_pipewire_default_sink
        
        target_mac = req.target.strip() if (req and req.target) else None
        if not target_mac:
            cfg = load_bt_config()
            target_mac = cfg.get("default_mac")
            if not target_mac:
                raise HTTPException(status_code=400, detail="尚未設定預設耳麥，請指定欲連線設備的 MAC 位址。")

        target_mac_upper = target_mac.upper()

        # 0. 查詢目標設備目前狀態，若已經是連線中，直接確保 PipeWire sink 並回傳成功
        res_target_check = subprocess.run(["bluetoothctl", "info", target_mac_upper], capture_output=True, text=True, timeout=3)
        if "Connected: yes" in res_target_check.stdout:
            m_target_name = re.search(r"Name:\s*(.+)", res_target_check.stdout)
            dname = m_target_name.group(1).strip() if m_target_name else "耳麥"
            _switch_pipewire_default_sink(dname)
            return {
                "status": "success",
                "ok": True,
                "msg": f"🎧 耳麥 {dname} ({target_mac_upper}) 已處於連線狀態",
                "message": f"🎧 耳麥 {dname} ({target_mac_upper}) 已處於連線狀態"
            }

        # 1. 偵測目前所有連線中的藍牙設備，若存在非目標設備則先自動中斷 (Auto-Disconnect)
        res_conn = subprocess.run(["bluetoothctl", "devices", "Connected"], capture_output=True, text=True, timeout=4)
        connected_macs = []
        for line in res_conn.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].lower() == "device":
                connected_macs.append(parts[1].upper())

        for curr_mac in connected_macs:
            if curr_mac != target_mac_upper:
                print(f"[BT Handover] 自動中斷已連線設備 {curr_mac}，以準備切換至 {target_mac}")
                subprocess.run(["bluetoothctl", "disconnect", curr_mac], capture_output=True, text=True, timeout=5)
                # 等待確認斷開，避免 BlueZ 狀態競爭
                for _ in range(5):
                    time.sleep(0.3)
                    chk = subprocess.run(["bluetoothctl", "info", curr_mac], capture_output=True, text=True, timeout=2)
                    if "Connected: yes" not in chk.stdout:
                        break

        # 2. 連線到目標 MAC (提供 12 秒交握逾時)
        res = subprocess.run(["bluetoothctl", "connect", target_mac_upper], capture_output=True, text=True, timeout=12)
        res_info = subprocess.run(["bluetoothctl", "info", target_mac_upper], capture_output=True, text=True, timeout=3)
        dev_name = "耳麥"
        m = re.search(r"Name:\s*(.+)", res_info.stdout)
        if m:
            dev_name = m.group(1).strip()

        is_conn_now = ("Connected: yes" in res_info.stdout) or ("Connection successful" in res.stdout)
        
        # 3. 若初次連線因交握競態失敗，嘗試第二次快速重試
        if not is_conn_now:
            time.sleep(0.8)
            res2 = subprocess.run(["bluetoothctl", "connect", target_mac_upper], capture_output=True, text=True, timeout=8)
            res_info2 = subprocess.run(["bluetoothctl", "info", target_mac_upper], capture_output=True, text=True, timeout=3)
            is_conn_now = ("Connected: yes" in res_info2.stdout) or ("Connection successful" in res2.stdout)

        # Invalidate cache
        global _last_bt_status_time
        _last_bt_status_time = 0.0

        if is_conn_now:
            _switch_pipewire_default_sink(dev_name)
            return {
                "status": "success",
                "ok": True,
                "msg": f"🎧 已成功連線至耳麥：{dev_name} ({target_mac_upper})",
                "message": f"🎧 已成功連線至耳麥：{dev_name} ({target_mac_upper})"
            }
        else:
            # 分析具體錯誤原因回報
            fail_detail = res.stdout + " " + res.stderr
            if "page-timeout" in fail_detail.lower():
                reason = "連線超時 (Page Timeout)。請確認該耳麥已開機、未在充電倉中，且未被其他手機/電腦佔線。"
            elif "not available" in fail_detail.lower():
                reason = "藍牙控制器回報設備不可用。請確認耳麥處於靠近樹莓派的有效距離內。"
            else:
                reason = "請確認耳麥處於開機或待連線狀態後重試。"
            raise HTTPException(status_code=400, detail=f"連線至 {dev_name} ({target_mac_upper}) 失敗：{reason}")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"連線失敗: {str(e)}")


@app.post("/api/bt/disconnect", tags=["Bluetooth"])
def disconnect_bt_device() -> Dict[str, Any]:
    """Disconnect active Bluetooth device."""
    try:
        from bluetooth_manager import disconnect_device
        ok, msg = disconnect_device()
        # Invalidate cache
        global _last_bt_status_time
        _last_bt_status_time = 0.0
        return {
            "status": "success" if ok else "failed",
            "ok": ok,
            "msg": msg,
            "message": msg
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"中斷連線失敗: {str(e)}")


@app.get("/api/system/memory_cache", tags=["System"])
def get_memory_cache_info() -> Dict[str, Any]:
    """Retrieve in-memory audio ring buffer capacity, RAM usage, and 0-disk-wear status."""
    return get_cache_stats()


class VoiceProcessRequest(BaseModel):
    audio_id: str = Field(..., description="Unique ID returned from /api/voice/record")


# -------------------------------------------------------------
# Streaming Chunking STT Endpoints (Task 01)
# -------------------------------------------------------------
@app.post("/api/voice/stream/start", tags=["Voice Streaming"])
async def start_voice_stream(req: VoiceRecordRequest) -> Dict[str, Any]:
    """
    Start real-time pipelined audio recording and streaming chunking STT.
    Records via ffmpeg while simultaneously cutting 6s chunks and transcribing in background.
    """
    from voice_streaming import create_streaming_session
    session = create_streaming_session(duration_sec=req.duration_sec, base_dir=AUDIO_CACHE_DIR)
    started = await session.start()
    if not started:
        raise HTTPException(status_code=500, detail="無法啟動串流錄音，請檢查藍牙耳麥是否連線。")
    return {
        "status": "started",
        "session_id": session.session_id,
        "duration_sec": req.duration_sec
    }


@app.get("/api/voice/stream/{session_id}/events", tags=["Voice Streaming"])
async def get_voice_stream_events(session_id: str):
    """
    SSE endpoint providing real-time progressive STT events (chunk, stt_completed, translating, session_completed).
    """
    from voice_streaming import get_streaming_session
    session = get_streaming_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="找不到指定的串流錄音工作階段。")

    return StreamingResponse(
        session.event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/voice/stream/{session_id}/stop", tags=["Voice Streaming"])
async def stop_voice_stream(session_id: str) -> Dict[str, Any]:
    """Manually stop streaming recording early."""
    from voice_streaming import get_streaming_session
    session = get_streaming_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="找不到指定的串流錄音工作階段。")
    await session.stop()
    return {"status": "stopping", "session_id": session_id}


@app.post("/api/voice/record", tags=["Voice"])
async def record_only_endpoint(req: VoiceRecordRequest) -> Dict[str, Any]:
    """
    Step 1 of progressive voice pipeline:
    Record audio from Bluetooth headset and convert to MP3 within 0.5s.
    Returns immediately so user can preview/playback audio without waiting for AI models.
    """
    import uuid
    import asyncio
    from voice_pipeline import record_audio_from_mic

    record_id = uuid.uuid4().hex[:8]
    raw_ogg = os.path.join(AUDIO_CACHE_DIR, f"rec_{record_id}.ogg")
    mp3_file = f"rec_{record_id}.mp3"
    raw_mp3 = os.path.join(AUDIO_CACHE_DIR, mp3_file)

    # 1. Record from mic (asyncio execution)
    success = await record_audio_from_mic(req.duration_sec, raw_ogg)
    if not success or not os.path.exists(raw_ogg) or os.path.getsize(raw_ogg) == 0:
        raise HTTPException(status_code=500, detail="麥克風錄音失敗，請確認藍牙耳麥連線正常且麥克風開啟。")

    # 2. Non-blocking MP3 conversion via asyncio subprocess
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", raw_ogg, "-c:a", "libmp3lame", "-b:a", "128k", raw_mp3,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.communicate()
        audio_url = f"/audio/{mp3_file}" if os.path.exists(raw_mp3) else f"/audio/rec_{record_id}.ogg"
    except Exception:
        audio_url = f"/audio/rec_{record_id}.ogg"

    return {
        "status": "success",
        "audio_id": record_id,
        "audio_url": audio_url,
        "duration_sec": req.duration_sec
    }


@app.post("/api/voice/transcribe", tags=["Voice"])
async def transcribe_only_endpoint(req: VoiceProcessRequest) -> Dict[str, Any]:
    """
    Step 2 of progressive voice pipeline:
    Runs faster-whisper STT in a background thread and returns Traditional Chinese text immediately.
    """
    import asyncio
    from voice_pipeline import speech_to_text

    raw_ogg = os.path.join(AUDIO_CACHE_DIR, f"rec_{req.audio_id}.ogg")
    if not os.path.exists(raw_ogg):
        raise HTTPException(status_code=404, detail="音訊檔案不存在，請重新錄音。")

    transcribed_text = await asyncio.to_thread(speech_to_text, raw_ogg)
    if not transcribed_text:
        return {
            "status": "empty",
            "transcribed_text": "",
            "message": "未能辨識出清晰人聲語音。您可點擊上方播放器試聽耳麥是否有錄到聲音。"
        }

    return {
        "status": "success",
        "transcribed_text": transcribed_text
    }


@app.post("/api/voice/transcribe_and_translate", tags=["Voice"])
async def transcribe_and_translate_endpoint(req: VoiceProcessRequest) -> Dict[str, Any]:
    """
    Step 2 of progressive voice pipeline:
    Runs faster-whisper STT and local Ollama translation in background worker threads,
    preventing any event loop blocking.
    """
    import asyncio
    from voice_pipeline import speech_to_text

    raw_ogg = os.path.join(AUDIO_CACHE_DIR, f"rec_{req.audio_id}.ogg")
    if not os.path.exists(raw_ogg):
        raise HTTPException(status_code=404, detail="音訊檔案不存在，請重新錄音。")

    # 1. Run Whisper STT in worker thread (non-blocking)
    transcribed_text = await asyncio.to_thread(speech_to_text, raw_ogg)
    if not transcribed_text:
        return {
            "status": "empty",
            "transcribed_text": "",
            "english_translation": "",
            "message": "未能辨識出清晰人聲語音。您可點擊上方播放器試聽耳麥是否有收到聲音。"
        }

    # 2. Run LLM translation in worker thread (non-blocking; Hailo-10H preferred, CPU fallback)
    def do_translate(text: str) -> str:
        from llm_backend import chat_complete, LLMError
        system = (
            "You are a professional bilingual interpreter. "
            "Translate the user's Traditional Chinese speech into natural, fluent, idiomatic English. "
            "Output ONLY the translated English text without explanations, notes, or Chinese characters."
        )
        try:
            translated, _ = chat_complete(text, system=system, temperature=0.2, num_predict=256, timeout=60)
            return translated
        except LLMError as e:
            return f"[翻譯失敗: {str(e)}]"

    english_translation = await asyncio.to_thread(do_translate, transcribed_text)

    return {
        "status": "success",
        "transcribed_text": transcribed_text,
        "english_translation": english_translation
    }


@app.post("/api/voice/record_and_transcribe", tags=["Voice"])
async def record_and_transcribe(req: VoiceRecordRequest) -> Dict[str, Any]:
    """
    Combined endpoint (backward-compatible, non-blocking):
    Records audio and executes AI processing in background threads.
    """
    import uuid
    import asyncio
    from voice_pipeline import record_audio_from_mic, speech_to_text

    record_id = uuid.uuid4().hex[:8]
    raw_ogg = os.path.join(AUDIO_CACHE_DIR, f"rec_{record_id}.ogg")
    mp3_file = f"rec_{record_id}.mp3"
    raw_mp3 = os.path.join(AUDIO_CACHE_DIR, mp3_file)

    # 1. Record from mic
    success = await record_audio_from_mic(req.duration_sec, raw_ogg)
    if not success or not os.path.exists(raw_ogg) or os.path.getsize(raw_ogg) == 0:
        raise HTTPException(status_code=500, detail="麥克風錄音失敗，請檢查藍牙耳麥是否連線並開啟麥克風。")

    # 2. Fast non-blocking MP3 conversion
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", raw_ogg, "-c:a", "libmp3lame", "-b:a", "128k", raw_mp3,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await proc.communicate()
        audio_url = f"/audio/{mp3_file}" if os.path.exists(raw_mp3) else f"/audio/rec_{record_id}.ogg"
    except Exception:
        audio_url = f"/audio/rec_{record_id}.ogg"

    # 3. Whisper STT in worker thread
    transcribed_text = await asyncio.to_thread(speech_to_text, raw_ogg)
    if not transcribed_text:
        return {
            "status": "empty",
            "transcribed_text": "",
            "english_translation": "",
            "audio_url": audio_url,
            "message": "未能辨識出清晰人聲語音，但錄音已完成。您可點擊下方播放器試聽錄音內容。"
        }

    # 4. LLM translation in worker thread (Hailo-10H preferred, CPU fallback)
    def do_translate(text: str) -> str:
        from llm_backend import chat_complete, LLMError
        system = (
            "You are a professional bilingual interpreter. "
            "Translate the user's Traditional Chinese speech into natural, fluent, idiomatic English. "
            "Output ONLY the translated English text without explanations, notes, or Chinese characters."
        )
        try:
            translated, _ = chat_complete(text, system=system, temperature=0.2, num_predict=256, timeout=60)
            return translated
        except LLMError as e:
            print(f"[Voice] translation failed: {e}")
            return ""

    english_translation = await asyncio.to_thread(do_translate, transcribed_text)

    return {
        "status": "success",
        "transcribed_text": transcribed_text,
        "english_translation": english_translation,
        "audio_url": audio_url
    }


# -------------------------------------------------------------
# Captive Portal & Hotspot Probes (Prevent OS from dropping Wi-Fi)
# -------------------------------------------------------------
@app.get("/hotspot-detect.html", response_class=HTMLResponse, include_in_schema=False)
def captive_apple():
    """Satisfy Apple iOS / macOS captive portal probe to prevent disconnecting hotspot."""
    return HTMLResponse(content="<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>")

@app.get("/generate_204", include_in_schema=False)
@app.get("/gen_204", include_in_schema=False)
def captive_google():
    """Satisfy Android / Chrome OS captive portal probe."""
    from fastapi import Response
    return Response(status_code=204)

@app.get("/ncsi.txt", include_in_schema=False)
def captive_windows():
    """Satisfy Windows NCSI probe."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("Microsoft NCSI")

@app.get("/connecttest.txt", include_in_schema=False)
def captive_android_txt():
    """Satisfy Android network probe."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("Microsoft Connect Test")


@app.post("/api/voice/translate", tags=["Voice"])
async def translate_text(req: VoiceTranslateRequest) -> Dict[str, Any]:
    """Translate text between Traditional Chinese and English via local LLM."""
    import asyncio

    clean_text = req.text.strip() if req.text else ""
    if not clean_text:
        return {"status": "success", "translated_text": "（尚無內容可翻譯，請先錄音生成會議紀要）"}

    # 若傳入的文本包含逐字稿段落，自動剝離，僅翻譯核心紀要（摘要、待辦、決議），節省邊緣運算開銷
    for marker in ["## 逐字稿", "## 完整逐字稿", "### 逐字稿"]:
        if marker in clean_text:
            clean_text = clean_text.split(marker)[0].strip()

    # 針對長篇紀要進行邊緣推論長度保護（600 字以內）
    if len(clean_text) > 600:
        clean_text = clean_text[:600] + "\n..."

    if req.direction == "zh2en":
        prompt = (
            "You are a professional bilingual interpreter. "
            "Translate the following meeting minutes summary and action items into concise, natural, fluent English. "
            "Keep clean bullet points format. Output ONLY the translated English text without explanations, notes, or Chinese characters:\n\n"
            f"{clean_text}"
        )
    else:
        prompt = (
            "請將以下英文會議紀要翻譯成道地流暢的繁體中文 (台灣)，保持列點格式，僅輸出翻譯結果，不要包含任何多餘說明：\n\n"
            f"{clean_text}"
        )

    def do_call():
        from llm_backend import chat_complete, LLMError
        try:
            out, _ = chat_complete(prompt, temperature=0.2, num_predict=250, timeout=90)
            return out if out else "（模型未輸出翻譯內容）"
        except LLMError as e:
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                return "[翻譯逾時：模型推論忙碌中，請稍後重試]"
            return f"[翻譯失敗: {str(e)}]"

    translated = await asyncio.to_thread(do_call)
    return {"status": "success", "translated_text": translated}


@app.post("/api/voice/tts", tags=["Voice"])
async def text_to_speech_endpoint(req: VoiceTTSRequest) -> Dict[str, Any]:
    """Synthesize text to speech using edge-tts and optionally play on speaker/headset."""
    import uuid
    from voice_pipeline import text_to_speech_async

    tts_id = uuid.uuid4().hex[:8]
    out_ogg = os.path.join(AUDIO_CACHE_DIR, f"tts_{tts_id}.ogg")

    ok = await text_to_speech_async(req.text, out_ogg, voice=req.voice or "zh-TW-HsiaoChenNeural")
    if not ok or not os.path.exists(out_ogg):
        raise HTTPException(status_code=500, detail="語音合成失敗")

    if req.play_on_speaker:
        # Play asynchronously through PipeWire (to bluetooth headset or onboard speaker)
        subprocess.Popen(["pw-play", out_ogg])

    return {
        "status": "success",
        "audio_url": f"/audio/tts_{tts_id}.ogg",
        "played_on_speaker": req.play_on_speaker
    }


from fastapi.staticfiles import StaticFiles
app.mount("/audio", StaticFiles(directory=AUDIO_CACHE_DIR), name="audio")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -------------------------------------------------------------
# Task 31: LocalNote Demo UI Adapter Endpoints
# -------------------------------------------------------------

class DemoRecordRequest(BaseModel):
    seconds: int = Field(default=90, ge=10, le=14400, description="錄音秒數 (最長 4 小時)")
    bypass_summary: Optional[bool] = Field(default=None, description="是否跳過 AI 總結，僅輸出完整逐字稿")


class DemoTranslateRequest(BaseModel):
    text: str = Field(..., description="待翻譯文字")
    to: str = Field(default="en", description="目標語言 'en' 或 'zh'")


class DemoTTSRequest(BaseModel):
    text: str = Field(..., description="待朗讀文字")


class DemoNetSwitchRequest(BaseModel):
    mode: str = Field(..., description="'ap' 切換為獨立熱點或 'sta' 切回家用 Wi-Fi")


class BypassSummaryConfigRequest(BaseModel):
    enabled: bool = Field(..., description="是否啟用調校模式跳過總結")


@app.post("/api/config/bypass_summary", tags=["Demo UI"])
async def set_bypass_summary_config(req: BypassSummaryConfigRequest) -> Dict[str, Any]:
    """Toggle bypassing AI summary for rapid STT tuning."""
    from meeting_minutes import minutes_manager
    minutes_manager.bypass_summary = req.enabled
    return {"bypass_summary": minutes_manager.bypass_summary}


@app.get("/api/config/bypass_summary", tags=["Demo UI"])
async def get_bypass_summary_config() -> Dict[str, Any]:
    """Get current bypass summary setting."""
    from meeting_minutes import minutes_manager
    return {"bypass_summary": getattr(minutes_manager, "bypass_summary", False)}


# NPU 遙測 (2026-10-27 修正)：
# - 舊版讀不到 hailortcli 輸出時不更新、也沒有時間戳，網頁會一直顯示舊值 (實測 12:27–13:48 凍結在 96.8% / 59.2 °C / 3734 MB)；
#   推論中還會塞入假的 95%。改為只存真實取樣與取樣時間，超過 NPU_STALE_SEC 未更新即標示 stale，前端顯示「—」。
# - hailortcli monitor 連線裝置時若撞上 hailo-ollama / Whisper 建立 VDevice，會造成 HAILO_DEVICE_TEMPORARILY_UNAVAILABLE；
#   因此錄音、語音轉文字與紀要整理期間都暫停取樣 (paused 告知前端原因)。
NPU_STALE_SEC = 10.0
_hailo_cache: Dict[str, Any] = {"time": 0.0, "data": None, "paused": None}
_hailo_monitor_started: bool = False


def _npu_pause_reason() -> Optional[str]:
    """NPU 正被使用 (或即將被使用) 時回傳原因，期間不跑 hailortcli。"""
    try:
        import npu_share
        if npu_share.is_resetting():
            return "npu_reset"
    except Exception:
        pass
    try:
        from voice_streaming import is_any_recording_active
        if is_any_recording_active():
            return "recording"
    except Exception:
        pass
    try:
        from meeting_minutes import minutes_manager
        st = minutes_manager.status
        if st.state == "recording":
            return "recording"
        if st.state == "processing":
            return "summary" if st.phase == "summarize" else "stt"
    except Exception:
        pass
    try:
        from stt_worker import stt_worker
        if stt_worker.is_busy():
            return "stt"
    except Exception:
        pass
    return None


def _hailo_telemetry_loop():
    """背景執行緒：只在 NPU 閒置時以 hailortcli monitor 取樣，不阻塞 HTTP。

    2026-10-28 實測 (raspi678)：hailortcli monitor 被 SIGTERM 砍掉 (timeout 的預設訊號) 時不會清理，
    Hailo-10H 裝置端每次漏約 3.7 MB；app 閒置時每 4.5 s 一次 → 每小時約 2.3 GB，是 NPU RAM 一直增加的主因。
    改送 SIGINT 讓它正常結束 (20 次實測不漏)；-k 2 只在它 2 秒內沒結束時才補 SIGKILL。
    """
    cmd = ["script", "-q", "-c", "timeout -s INT -k 2 1.5 hailortcli monitor", "/dev/null"]
    while True:
        reason = _npu_pause_reason()
        _hailo_cache["paused"] = reason
        if reason:
            time.sleep(1.0)
            continue
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5.0)
            data_line = next((l.strip() for l in res.stdout.splitlines() if "pci/" in l), "")
            parts = re.split(r"\s{2,}", data_line) if data_line else []
            if len(parts) >= 8:
                _hailo_cache["data"] = {
                    "detected": True,
                    "device": "Hailo-10H",
                    "tops": 40,
                    "device_id": parts[0],
                    "arch": parts[1],
                    "nnc_util_pct": float(parts[2]),
                    "cpu_util_pct": float(parts[3]),
                    "ram_util_pct": float(parts[4]),
                    "ram_usage_mb": parts[5],
                    "temp_c": float(parts[6]),
                    "voltage_mv": int(parts[7]),
                }
                _hailo_cache["time"] = time.time()
                _maybe_reset_npu_group(parts[5])
        except Exception:
            pass
        time.sleep(3.0)


def _maybe_reset_npu_group(ram_usage: str) -> None:
    """共用模式 (npu_share.py)：閒置時 NPU RAM 異常偏高 (LLM 卸載/重啟或 STT 子行程被 kill 的殘留) → 背景群組重置。"""
    try:
        import npu_share
        from llm_backend import hailo_idle_seconds, prewarm_hailo
        npu_share.maybe_reset(npu_share.parse_ram_used_mb(ram_usage), npu_busy=_npu_pause_reason() is not None,
                              llm_idle_sec=hailo_idle_seconds(), prewarm=prewarm_hailo)
    except Exception as e:
        print(f"[NPU] group reset check failed: {e}")


def get_hailo10_stats() -> Dict[str, Any]:
    """回傳最後一次真實取樣 + 取樣年齡 (age_s)、是否過期 (stale) 與暫停原因 (paused)。"""
    global _hailo_monitor_started
    if not _hailo_monitor_started:
        _hailo_monitor_started = True
        th = threading.Thread(target=_hailo_telemetry_loop, daemon=True, name="hailo-monitor-daemon")
        th.start()
    t = _hailo_cache.get("time") or 0.0
    age = round(time.time() - t, 1) if t else None
    out = dict(_hailo_cache.get("data") or {"detected": True, "device": "Hailo-10H", "tops": 40})
    out.update({"age_s": age, "stale": age is None or age > NPU_STALE_SEC, "paused": _hailo_cache.get("paused")})
    try:
        import npu_share
        out["shared_group"] = npu_share.shared_group()
        out["last_group_reset"] = npu_share.reset_status()
    except Exception:
        pass
    return out


@app.get("/api/system/hailo", tags=["System"])
def get_hailo_system_status() -> Dict[str, Any]:
    """Live Hailo-10H NPU metrics (temperature, NNC load, RAM utilization)."""
    return get_hailo10_stats()


def get_system_telemetry_payload() -> Dict[str, Any]:
    """Compile comprehensive system and Hailo-10H telemetry for HTTP and WebSocket."""
    from meeting_minutes import minutes_manager
    qr_path = os.path.join(STATIC_DIR, "wifi.png")

    # 探測 Hailo-10H 或 Hailo-8 NPU 狀態
    npu_info = {"detected": False, "device": "CPU ARM Cortex-A76", "tops": 0}
    hailo10 = get_hailo10_stats()
    if hailo10.get("detected"):
        npu_info = {
            "detected": True,
            "device": "Hailo-10H",
            "tops": 40,
            "detail": "Hailo-10H (40 TOPS)",
            "temp_c": hailo10.get("temp_c"),
            "nnc_util_pct": hailo10.get("nnc_util_pct"),
            "ram_usage_mb": hailo10.get("ram_usage_mb"),
            "ram_util_pct": hailo10.get("ram_util_pct"),
            "voltage_mv": hailo10.get("voltage_mv"),
            "age_s": hailo10.get("age_s"),
            "stale": hailo10.get("stale"),
            "paused": hailo10.get("paused"),
        }
    else:
        try:
            hailo_adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AI_HAT_plus"))
            if os.path.exists(hailo_adapter_path) and hailo_adapter_path not in sys.path:
                sys.path.append(hailo_adapter_path)
            from hailo_engine.device_manager import is_hailo_available, HailoDeviceManager
            if is_hailo_available():
                info = HailoDeviceManager().get_info()
                npu_info = {
                    "detected": True,
                    "device": "AI NPU",
                    "tops": 26,
                    "detail": "AI NPU"
                }
        except Exception:
            pass

    return {
        "offline": minutes_manager.offline_detector.is_offline,
        "cpu": minutes_manager.get_cpu_load(),
        "temp": minutes_manager.get_temperature(),
        "mode": minutes_manager.get_network_mode(),
        "hostname": SYSTEM_HOSTNAME,
        "ssid": DEFAULT_HOTSPOT_SSID,
        "host": f"{SYSTEM_HOSTNAME}.local",
        "wifi_qr": "/static/wifi.png" if os.path.exists(qr_path) else None,
        "npu": npu_info,
        "hailo": hailo10,
        "bypass_summary": getattr(minutes_manager, "bypass_summary", False)
    }


@app.get("/api/system", tags=["Demo UI"])
def get_demo_system() -> Dict[str, Any]:
    """Retrieve simplified system status for LocalNote client dashboard."""
    return get_system_telemetry_payload()


@app.get("/api/bt", tags=["Demo UI"])
def get_demo_bt() -> Dict[str, Any]:
    """Simplified Bluetooth status without polling overhead."""
    st = get_bt_status()
    return {
        "connected": bool(st.get("connected", False)),
        "name": st.get("connected_name") or ""
    }


@app.post("/api/record", tags=["Demo UI"])
async def post_demo_record(req: DemoRecordRequest) -> Dict[str, Any]:
    """Start fixed-duration recording and pipeline job."""
    from meeting_minutes import minutes_manager
    try:
        job_id = await minutes_manager.start_record(req.seconds, bypass_summary=req.bypass_summary)
        return {"job": job_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/record/cancel", tags=["Demo UI"])
@app.post("/api/job/cancel", tags=["Demo UI"])
async def post_demo_cancel() -> Dict[str, Any]:
    """Instantly cancel active recording or inference job and release hardware resources."""
    from meeting_minutes import minutes_manager
    return await minutes_manager.cancel_current_job()


@app.post("/api/record/finish", tags=["Demo UI"])
async def post_demo_finish_record() -> Dict[str, Any]:
    """Instantly finish recording early, preserve all captured audio and proceed to minutes synthesis."""
    from meeting_minutes import minutes_manager
    return await minutes_manager.finish_recording_early()


class RegionTranscribeRequest(BaseModel):
    job_id: Optional[str] = Field(default=None, description="目標工作 ID")
    start_sec: float = Field(default=0.0, ge=0.0, description="開始秒數")
    end_sec: float = Field(..., gt=0.0, description="結束秒數")


@app.post("/api/transcribe/region", tags=["Demo UI"])
async def post_transcribe_region(req: RegionTranscribeRequest) -> Dict[str, Any]:
    """Transcribe selected audio region via faster-whisper without full-length re-processing."""
    from meeting_minutes import minutes_manager
    try:
        return await minutes_manager.transcribe_region(
            job_id=req.job_id,
            start_sec=req.start_sec,
            end_sec=req.end_sec
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/audio/latest", tags=["Demo UI"])
def get_latest_audio() -> Dict[str, Any]:
    """Retrieve the URL of the most recent meeting audio file for Wavesurfer.js."""
    from meeting_minutes import minutes_manager, AUDIO_CACHE_DIR
    target_ogg = None
    if minutes_manager.status and minutes_manager.status.job:
        cand = os.path.join(AUDIO_CACHE_DIR, f"{minutes_manager.status.job}.ogg")
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            target_ogg = f"{minutes_manager.status.job}.ogg"

    if not target_ogg and minutes_manager.result and minutes_manager.result.job:
        cand = os.path.join(AUDIO_CACHE_DIR, f"{minutes_manager.result.job}.ogg")
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            target_ogg = f"{minutes_manager.result.job}.ogg"

    if not target_ogg:
        try:
            ogg_files = [
                f for f in os.listdir(AUDIO_CACHE_DIR)
                if f.endswith(".ogg") and not f.startswith("tts_")
            ]
            if ogg_files:
                ogg_files.sort(key=lambda x: os.path.getmtime(os.path.join(AUDIO_CACHE_DIR, x)), reverse=True)
                target_ogg = ogg_files[0]
        except Exception:
            pass

    if target_ogg:
        global _peaks_cache
        if "_peaks_cache" not in globals():
            _peaks_cache = {}

        peaks = _peaks_cache.get(target_ogg)
        if peaks is None:
            try:
                from two_layer_audio import extract_waveform_peaks
                full_path = os.path.join(AUDIO_CACHE_DIR, target_ogg)
                peaks = extract_waveform_peaks(full_path, num_peaks=200)
                _peaks_cache[target_ogg] = peaks
            except Exception:
                peaks = []

        return {
            "status": "success",
            "audio_url": f"/audio/{target_ogg}",
            "filename": target_ogg,
            "duration": minutes_manager.result.duration if minutes_manager.result else 0,
            "peaks": peaks
        }
    return {"status": "none", "audio_url": None, "message": "尚未有錄音檔"}


@app.get("/api/status", tags=["Demo UI"])
def get_demo_status() -> Dict[str, Any]:
    """Poll pipeline progress, ETA, and current stage."""
    from meeting_minutes import minutes_manager
    return minutes_manager.status.model_dump()


@app.get("/api/result", tags=["Demo UI"])
def get_demo_result() -> Dict[str, Any]:
    """Fetch meeting minutes result with structured summary, todos, decisions and markdown."""
    from meeting_minutes import minutes_manager
    if not minutes_manager.result:
        return {
            "job": "",
            "title": "會議紀要",
            "started_at": datetime.now().isoformat(),
            "duration": 0,
            "summary": [],
            "todos": [],
            "decisions": [],
            "transcript": "",
            "markdown": ""
        }
    return minutes_manager.result.model_dump()


@app.post("/api/regenerate", tags=["Demo UI"])
@app.post("/api/minutes/regenerate", tags=["Demo UI"], include_in_schema=False)
async def post_demo_regenerate() -> Dict[str, Any]:
    """Regenerate meeting summary from transcript without recording again."""
    from meeting_minutes import minutes_manager
    try:
        job_id = await minutes_manager.regenerate()
        return {"job": job_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/translate", tags=["Demo UI"])
async def post_demo_translate(req: DemoTranslateRequest) -> Dict[str, Any]:
    """Translate text between Traditional Chinese and English for Demo UI."""
    direction = "zh2en" if req.to.lower() == "en" else "en2zh"
    v_req = VoiceTranslateRequest(text=req.text, direction=direction)
    res = await translate_text(v_req)
    return {"text": res.get("translated_text", "")}


@app.post("/api/tts", tags=["Demo UI"])
async def post_demo_tts(req: DemoTTSRequest) -> Dict[str, Any]:
    """Synthesize text and play directly on Bluetooth headset."""
    voice = "en-US-AriaNeural" if any(ord(c) < 128 and c.isalpha() for c in req.text[:20]) else "zh-TW-HsiaoChenNeural"
    v_req = VoiceTTSRequest(text=req.text, voice=voice, play_on_speaker=True)
    res = await text_to_speech_endpoint(v_req)
    return {"ok": res.get("status") == "success"}


@app.post("/api/net/switch", tags=["Demo UI"])
def post_demo_net_switch(req: DemoNetSwitchRequest) -> Dict[str, Any]:
    """Switch between standalone AP mode and home Wi-Fi."""
    try:
        from hotspot_controller import switch_to_ap, switch_to_sta
        if req.mode.lower() == "ap":
            switch_to_ap(timeout_sec=180)
        else:
            switch_to_sta()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"網路切換失敗: {str(e)}")


# -------------------------------------------------------------
# Task 08: WebSocket Real-Time Status & Telemetry Push
# -------------------------------------------------------------
async def ws_telemetry_loop():
    """Periodic telemetry broadcaster replacing high-frequency HTTP polling."""
    from ws_manager import ws_manager
    from meeting_minutes import minutes_manager
    qr_path = os.path.join(STATIC_DIR, "wifi.png")

    while True:
        try:
            await asyncio.sleep(1.5)
            if not ws_manager.active_connections:
                continue

            # 1. 廣播系統與 Hailo-10H 硬體指標 (System & Hailo Telemetry)
            sys_data = get_system_telemetry_payload()
            await ws_manager.broadcast("system", sys_data)

            # 2. 廣播藍牙基本狀態 (Bluetooth Status)
            st = get_bt_status()
            await ws_manager.broadcast("bt", {
                "connected": bool(st.get("connected", False)),
                "name": st.get("connected_name") or ""
            })

            # 3. 廣播藍牙詳細 Hub 狀態 (若有客戶端開啟了藍牙中心)
            await ws_manager.broadcast("bt_hub", st, topic="bt_hub")

        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(2.0)


async def bt_auto_reconnect_loop():
    """Background loop to auto-reconnect trusted Bluetooth headsets when disconnected."""
    logger.info("[Startup] Bluetooth auto-reconnect daemon started.")
    while True:
        try:
            await asyncio.sleep(12)
            # Strict AGENTS.md rule: Zero HCI or contention during recording
            from voice_streaming import is_any_recording_active
            if is_any_recording_active():
                continue

            from bluetooth_manager import auto_reconnect_tick
            await asyncio.to_thread(auto_reconnect_tick)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"[BT AutoReconnect Loop] Notice: {e}")
            await asyncio.sleep(5)


@app.websocket("/ws/status")
async def websocket_status_endpoint(websocket: WebSocket):
    """Bidirectional WebSocket endpoint for zero-latency status push."""
    from ws_manager import ws_manager
    from meeting_minutes import minutes_manager

    await ws_manager.connect(websocket)
    try:
        # 連線建立瞬間，立即回傳一筆當前全量狀態快照 (Immediate initial snapshot)
        sys_data = get_system_telemetry_payload()
        await ws_manager.send_personal(websocket, "system", sys_data)
        await ws_manager.send_personal(websocket, "status", minutes_manager.status.model_dump())
        bt_data = get_bt_status()
        await ws_manager.send_personal(websocket, "bt", {
            "connected": bool(bt_data.get("connected", False)),
            "name": bt_data.get("connected_name") or ""
        })

        while True:
            text = await websocket.receive_text()
            try:
                cmd = json.loads(text)
                act = cmd.get("action")
                if act == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "ts": int(time.time())}))
                elif act == "refresh":
                    sys_data = get_system_telemetry_payload()
                    await ws_manager.send_personal(websocket, "system", sys_data)
                    await ws_manager.send_personal(websocket, "status", minutes_manager.status.model_dump())
                    bt_data = get_bt_status()
                    await ws_manager.send_personal(websocket, "bt", {
                        "connected": bool(bt_data.get("connected", False)),
                        "name": bt_data.get("connected_name") or ""
                    })
                elif act == "subscribe_bt_hub":
                    ws_manager.subscribe(websocket, "bt_hub", enable=bool(cmd.get("enable", True)))
            except Exception:
                pass

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"[WS] Exception in client session: {e}")
        await ws_manager.disconnect(websocket)



