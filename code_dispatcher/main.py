# f:\12_prj_raspi5\code_dispatcher\main.py
"""
Raspberry Pi 5 Code Dispatcher & Task Gateway (FastAPI)
Exposes REST endpoints for pushing code analysis tasks to Redis Queue.
"""

import os
import sys
import subprocess
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
from redis import Redis
from rq import Queue
from rq.job import Job

# Ensure PipeWire / PulseAudio session is accessible for systemd service
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
os.environ.setdefault("PULSE_SERVER", "unix:/run/user/1000/pulse/native")

# Ensure task functions and telegram_agent modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
agent_candidate_dirs = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "telegram_agent")),
    "/home/pi/pi5-llm-arena/telegram_agent",
    os.path.expanduser("~/pi5-llm-arena/telegram_agent"),
    r"f:\12_prj_raspi5\telegram_agent"
]
for d in agent_candidate_dirs:
    if os.path.exists(d) and d not in sys.path:
        sys.path.insert(0, d)

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

@app.on_event("startup")
async def startup_event():
    import asyncio
    try:
        from voice_pipeline import get_whisper_model
        print("[Startup] Pre-warming faster-whisper model in background...")
        asyncio.create_task(asyncio.to_thread(get_whisper_model))
    except Exception as e:
        print(f"[Startup] Whisper preload notice: {e}")

redis_conn = Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
task_queue = Queue(QUEUE_NAME, connection=redis_conn)

HTML_UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
HTML_DBG01_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui_dbg01.html")


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
    """Serve the Bluetooth Device Management & Voice Application Studio (Default on Port 80)."""
    if os.path.exists(HTML_UI_PATH):
        with open(HTML_UI_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Pi 5 Edge Gateway</h1><p>ui.html not found.</p>"


@app.get("/dbg01", response_class=HTMLResponse, tags=["Web UI"])
@app.get("/dgb01", response_class=HTMLResponse, include_in_schema=False)
def get_dbg01_ui():
    """Serve the original Code Review & LLM Arena interface at /dbg01."""
    if os.path.exists(HTML_DBG01_PATH):
        with open(HTML_DBG01_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Pi 5 LLM Arena (DBG01)</h1><p>ui_dbg01.html not found.</p>"



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

    return {
        "status": "healthy" if (redis_ok and ollama_ok) else "degraded",
        "redis_connected": redis_ok,
        "ollama_running": ollama_ok,
        "available_models_count": len(active_models),
        "cpu_temperature_c": temp_c,
        "ram_available_mb": free_ram_mb,
        "ram_total_mb": total_ram_mb,
        "queue_length": len(task_queue),
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
    return {
        "queue_name": QUEUE_NAME,
        "queued_jobs_count": len(task_queue),
        "started_jobs_count": task_queue.started_job_registry.count,
        "finished_jobs_count": task_queue.finished_job_registry.count,
        "failed_jobs_count": task_queue.failed_job_registry.count
    }


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
        "hotspot_gateway_ip": "10.20.0.1",
        "hotspot_ssid": "raspi543_AI",
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

    profile_name = "Pi5-Hotspot (raspi543_AI)" if target == "ap" else "My_5G_Guest1"
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
        sys.path.insert(0, d)

AUDIO_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)


class BTPairRequest(BaseModel):
    target: str = Field(..., description="Device MAC address, list index, or name")


class VoiceRecordRequest(BaseModel):
    duration_sec: int = Field(default=5, ge=2, le=120, description="Recording duration in seconds")


class VoiceTranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    direction: str = Field(default="zh2en", description="'zh2en' or 'en2zh'")


class VoiceTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(default="zh-TW-HsiaoChenNeural", description="Neural TTS Voice")
    play_on_speaker: Optional[bool] = Field(default=True, description="Whether to play on Bluetooth speaker directly")


@app.get("/api/bt/status", tags=["Bluetooth"])
def get_bt_status() -> Dict[str, Any]:
    """Retrieve Bluetooth controller and paired audio devices status."""
    try:
        from bluetooth_manager import get_full_bt_status, load_bt_config
        status_text = get_full_bt_status()
        config = load_bt_config()

        # Check connected device
        res = subprocess.run(["bluetoothctl", "info"], capture_output=True, text=True, timeout=3)
        connected_mac = None
        connected_name = None
        battery_level = None
        link_quality = None
        signal_level_pct = None
        rssi = None

        if "Device " in res.stdout:
            parts = res.stdout.split()
            idx = parts.index("Device") if "Device" in parts else -1
            if idx != -1 and idx + 1 < len(parts):
                connected_mac = parts[idx + 1]
            import re
            m = re.search(r"Name:\s*(.+)", res.stdout)
            if m:
                connected_name = m.group(1).strip()

            # Battery parsing e.g. "Battery Percentage: 0x28 (40)"
            m_bat = re.search(r"Battery Percentage:\s*.*\((\d+)\)", res.stdout)
            if m_bat:
                battery_level = int(m_bat.group(1))

            # Query real-time link quality & RSSI if device is connected
            if connected_mac:
                try:
                    lq_res = subprocess.run(["hcitool", "lq", connected_mac], capture_output=True, text=True, timeout=2)
                    m_lq = re.search(r"Link quality:\s*(\d+)", lq_res.stdout)
                    if m_lq:
                        link_quality = int(m_lq.group(1))
                        signal_level_pct = round((link_quality / 255.0) * 100)
                except Exception:
                    pass

                try:
                    rssi_res = subprocess.run(["hcitool", "rssi", connected_mac], capture_output=True, text=True, timeout=2)
                    m_rssi = re.search(r"RSSI return value:\s*(-?\d+)", rssi_res.stdout)
                    if m_rssi:
                        rssi = int(m_rssi.group(1))
                except Exception:
                    pass

        state_hash = f"{connected_mac}_{connected_name}_{battery_level}_{signal_level_pct}"

        return {
            "status_text": status_text,
            "connected": connected_mac is not None,
            "connected_mac": connected_mac,
            "connected_name": connected_name,
            "battery_level": battery_level,
            "link_quality": link_quality,
            "signal_level_pct": signal_level_pct,
            "rssi": rssi,
            "state_hash": state_hash,
            "default_mac": config.get("default_mac"),
            "default_name": config.get("default_name")
        }
    except Exception as e:
        return {
            "status_text": f"藍牙狀態獲取失敗: {str(e)}",
            "connected": False,
            "state_hash": f"err_{str(e)}",
            "error": str(e)
        }


@app.post("/api/bt/scan", tags=["Bluetooth"])
def scan_bt_devices(timeout_sec: int = 8) -> Dict[str, Any]:
    """Scan for nearby Bluetooth audio devices."""
    try:
        from bluetooth_manager import scan_devices
        devices = scan_devices(timeout_sec=timeout_sec)
        return {
            "status": "success",
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


@app.post("/api/bt/connect", tags=["Bluetooth"])
def connect_bt_device() -> Dict[str, Any]:
    """Connect to default configured Bluetooth headset."""
    try:
        from bluetooth_manager import connect_default_device
        ok, msg = connect_default_device()
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {"status": "success", "message": msg}
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
        return {"status": "success" if ok else "failed", "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"中斷連線失敗: {str(e)}")


class VoiceProcessRequest(BaseModel):
    audio_id: str = Field(..., description="Unique ID returned from /api/voice/record")


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

    # 2. Run Ollama translation in worker thread (non-blocking)
    def do_translate(text: str) -> str:
        prompt = (
            "You are a professional bilingual interpreter. "
            "Translate the following Traditional Chinese speech into natural, fluent, idiomatic English. "
            "Output ONLY the translated English sentence without explanations, notes, or Chinese characters:\n\n"
            f"{text}"
        )
        try:
            resp = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": "qwen2.5:3b",
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "24h"
                },
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return f"[翻譯狀態碼 {resp.status_code}]"
        except Exception as e:
            return f"[翻譯超時/失敗: {str(e)}]"

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

    # 4. Ollama translation in worker thread
    def do_translate(text: str) -> str:
        prompt = (
            "You are a professional bilingual interpreter. "
            "Translate the following Traditional Chinese speech into natural, fluent, idiomatic English. "
            "Output ONLY the translated English sentence without explanations, notes, or Chinese characters:\n\n"
            f"{text}"
        )
        try:
            resp = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": "qwen2.5:3b",
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "24h"
                },
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return ""
        except Exception:
            return ""

    english_translation = await asyncio.to_thread(do_translate, transcribed_text)

    return {
        "status": "success",
        "transcribed_text": transcribed_text,
        "english_translation": english_translation,
        "audio_url": audio_url
    }


@app.post("/api/voice/translate", tags=["Voice"])
async def translate_text(req: VoiceTranslateRequest) -> Dict[str, Any]:
    """Translate text between Traditional Chinese and English via local LLM."""
    import asyncio

    if req.direction == "zh2en":
        prompt = (
            "You are a professional bilingual interpreter. "
            "Translate the following Traditional Chinese speech into natural, fluent, idiomatic English. "
            "Output ONLY the translated English sentence without explanations, notes, or Chinese characters:\n\n"
            f"{req.text}"
        )
    else:
        prompt = (
            "請將以下英文翻譯成道地流暢的繁體中文 (台灣)，僅輸出翻譯結果，不要包含任何多餘說明：\n\n"
            f"{req.text}"
        )

    def do_call():
        try:
            resp = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": "qwen2.5:3b",
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "24h"
                },
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return f"[LLM 狀態碼 {resp.status_code}]"
        except Exception as e:
            return f"[翻譯逾時: {str(e)}]"

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


