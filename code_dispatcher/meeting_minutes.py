# code_dispatcher/meeting_minutes.py
"""
LocalNote Meeting Minutes Engine & State Machine
Handles recording lifecycle, non-blocking offline probing, faster-whisper STT,
and Ollama LLM meeting minutes generation with markdown parsing.
"""

import os
import re
import time
import asyncio
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional
import threading
import requests
from pydantic import BaseModel, Field
import sys

# Ensure current module directory has highest priority in sys.path
_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir in sys.path:
    sys.path.remove(_cur_dir)
sys.path.insert(0, _cur_dir)

# Append telegram_agent to end of sys.path so it never shadows code_dispatcher modules
_agent_dir = os.path.abspath(os.path.join(_cur_dir, "..", "telegram_agent"))
if os.path.exists(_agent_dir) and _agent_dir not in sys.path:
    sys.path.append(_agent_dir)

from voice_memory import get_in_memory_cache_dir
from voice_pipeline import record_audio_from_mic, speech_to_text

AUDIO_CACHE_DIR = get_in_memory_cache_dir()

# System configuration defaults
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b-opt"
FALLBACK_MODEL = "qwen2.5:3b"


class MeetingResult(BaseModel):
    job: str
    title: str = "會議紀要"
    started_at: str
    duration: int = 90
    summary: List[str] = Field(default_factory=list)
    todos: List[Dict[str, str]] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    transcript: str = ""
    markdown: str = ""


class JobStatus(BaseModel):
    state: str = "idle"  # idle / recording / processing / done / error
    phase: str = "record"  # record / transcribe / summarize
    progress: int = 0  # 0 - 100
    elapsed: float = 0.0
    total: float = 90.0
    eta: Optional[float] = None
    job: Optional[str] = None
    msg: Optional[str] = None
    transcript: Optional[str] = None
    audio_url: Optional[str] = None
    summary_stream: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class OfflineDetector:
    """Non-blocking background watchdog to probe internet connectivity."""

    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self._is_offline: bool = True
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    @property
    def is_offline(self) -> bool:
        return self._is_offline

    async def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _check_once(self) -> bool:
        """Ping a public DNS resolver (1.1.1.1) with 0.8s timeout."""
        try:
            # On Linux: ping -c 1 -W 1 1.1.1.1; on Windows: ping -n 1 -w 800 1.1.1.1
            cmd = ["ping", "-c", "1", "-W", "1", "1.1.1.1"] if os.name != "nt" else ["ping", "-n", "1", "-w", "800", "1.1.1.1"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            ret = await proc.wait()
            return ret != 0  # returncode != 0 means offline
        except Exception:
            return True

    async def _loop(self):
        while self._running:
            try:
                self._is_offline = await self._check_once()
            except Exception:
                self._is_offline = True
            await asyncio.sleep(self.check_interval)


def parse_markdown_minutes(raw_text: str) -> Dict[str, Any]:
    """
    Robust regex parser for Ollama meeting minutes markdown.
    Extracts summary, todos table, and decisions.
    """
    summaries: List[str] = []
    todos: List[Dict[str, str]] = []
    decisions: List[str] = []

    # 1. Extract Summary
    summary_match = re.search(r"##\s*摘要\s*\n([\s\S]*?)(?=\n##|\Z)", raw_text, re.IGNORECASE)
    if summary_match:
        lines = [l.strip() for l in summary_match.group(1).splitlines() if l.strip()]
        bullet_items = []
        raw_items = []
        for line_str in lines:
            m = re.match(r"^(?:[\*\-•]|\d+\.)\s*(.*)", line_str)
            if m:
                item = m.group(1).strip()
                if item:
                    bullet_items.append(item)
            elif not line_str.startswith("#"):
                raw_items.append(line_str)
        summaries = bullet_items if bullet_items else raw_items

    # 2. Extract Todos Table
    todo_match = re.search(r"##\s*待辦\s*\n([\s\S]*?)(?=\n##|\Z)", raw_text, re.IGNORECASE)
    if todo_match:
        lines = todo_match.group(1).strip().splitlines()
        for line in lines:
            line_str = line.strip()
            if not line_str.startswith("|") or "---" in line_str:
                continue
            cols = [c.strip() for c in line_str.split("|")[1:-1]]
            if len(cols) >= 3:
                # Skip header row
                if "事項" in cols[0] or "待辦" in cols[0]:
                    continue
                todos.append({
                    "item": cols[0],
                    "owner": cols[1] if cols[1] else "待定",
                    "due": cols[2] if cols[2] else "待定"
                })

    # 3. Extract Decisions
    decision_match = re.search(r"##\s*決議\s*\n([\s\S]*?)(?=\n##|\Z)", raw_text, re.IGNORECASE)
    if decision_match:
        lines = [l.strip() for l in decision_match.group(1).splitlines() if l.strip()]
        bullet_items = []
        raw_items = []
        for line_str in lines:
            m = re.match(r"^(?:[\*\-•]|\d+\.)\s*(.*)", line_str)
            if m:
                item = m.group(1).strip()
                if item:
                    bullet_items.append(item)
            elif not line_str.startswith("#"):
                raw_items.append(line_str)
        decisions = bullet_items if bullet_items else raw_items

    # Fallback heuristics if markdown was freeform
    if not summaries and not todos and not decisions:
        for line in raw_text.splitlines():
            line_str = line.strip()
            if line_str.startswith(("-", "*", "•")):
                summaries.append(line_str.lstrip("-*• ").strip())

    return {
        "summary": summaries,
        "todos": todos,
        "decisions": decisions
    }


class MeetingJobManager:
    """Singleton manager controlling the state machine and pipeline execution."""

    def __init__(self):
        self.status = JobStatus()
        self.result: Optional[MeetingResult] = None
        self._current_task: Optional[asyncio.Task] = None
        self._record_task: Optional[asyncio.Task] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self.offline_detector = OfflineDetector(check_interval=10.0)
        self._last_cpu_time: float = 0.0
        self._last_cpu_idle: int = 0
        self._last_cpu_total: int = 0
        self._cpu_cache: float = 0.0
        self._cancellation_token: asyncio.Event = asyncio.Event()
        self._thread_cancellation_token: threading.Event = threading.Event()
        self._current_ffmpeg_proc = None

    def notify_status_update(self):
        """Notify WebSocket subscribers of status update safely across async and worker threads."""
        try:
            from ws_manager import ws_manager
            coro = ws_manager.broadcast("status", self.status.model_dump())
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                # In background worker thread (e.g. STT segment callback)
                if self._main_loop and self._main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(coro, self._main_loop)
                else:
                    coro.close()
        except Exception:
            pass

    def get_cpu_load(self) -> float:
        """Calculate CPU percentage from /proc/stat non-blockingly."""
        now = time.time()
        if now - self._last_cpu_time < 1.0:
            return self._cpu_cache

        try:
            with open("/proc/stat", "r") as f:
                first_line = f.readline()
            fields = [int(x) for x in first_line.split()[1:]]
            idle = fields[3] + fields[4]
            total = sum(fields)
            diff_idle = idle - self._last_cpu_idle
            diff_total = total - self._last_cpu_total
            if diff_total > 0:
                self._cpu_cache = max(0.0, min(100.0, (1.0 - diff_idle / diff_total) * 100.0))
            self._last_cpu_idle = idle
            self._last_cpu_total = total
            self._last_cpu_time = now
        except Exception:
            # Fallback for dev / non-linux environments
            self._cpu_cache = 12.0
        return self._cpu_cache

    def get_temperature(self) -> float:
        """Fetch temperature from vcgencmd or Linux thermal sysfs."""
        try:
            res = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=1)
            if res.returncode == 0:
                return float(res.stdout.strip().replace("temp=", "").replace("'C", ""))
        except Exception:
            pass

        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            pass
        return 45.0

    def get_network_mode(self) -> str:
        """Determine if wlan0 is in AP mode or STA mode."""
        try:
            res = subprocess.run(["ip", "-4", "addr", "show", "wlan0"], capture_output=True, text=True, timeout=1)
            if "10.20.0.1" in res.stdout:
                return "ap"
        except Exception:
            pass
        return "sta"

    async def cancel_current_job(self) -> Dict[str, Any]:
        """
        Instantly abort active recording or inference job and release hardware resources.
        Task 11: 異步工作佇列與取消權杖 (Job Queue & Cancellation Token)
        """
        print("[MeetingMinutes] cancel_current_job requested.")
        self._cancellation_token.set()
        self._thread_cancellation_token.set()

        # 1. Kill active ffmpeg recording process if still alive
        if self._current_ffmpeg_proc:
            try:
                self._current_ffmpeg_proc.terminate()
                try:
                    self._current_ffmpeg_proc.kill()
                except Exception:
                    pass
                print("[MeetingMinutes] Active ffmpeg recording process terminated and killed.")
            except Exception as pe:
                print(f"[MeetingMinutes] Error terminating ffmpeg process: {pe}")
            self._current_ffmpeg_proc = None

        # 2. Cancel record sub-task if active
        if self._record_task and not self._record_task.done():
            self._record_task.cancel()
            self._record_task = None
            print("[MeetingMinutes] Active recording sub-task cancelled.")

        # 3. Cancel asyncio pipeline task
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            print("[MeetingMinutes] Background pipeline task cancelled.")

        # 4. Broadcast VU reset
        try:
            from ws_manager import ws_manager
            await ws_manager.broadcast("vu", {
                "rms": 0.0,
                "peak": 0.0,
                "db": -60.0,
                "level_pct": 0,
                "status": "idle",
                "bands": [0] * 8,
                "active": False
            })
        except Exception:
            pass

        # 5. Reset status to idle
        self.status = JobStatus(
            state="idle",
            phase="record",
            progress=0,
            elapsed=0.0,
            total=0.0,
            eta=0.0,
            msg="任務已由使用者手動中斷取消。"
        )
        self.notify_status_update()
        return {"status": "cancelled", "message": "任務已成功取消並釋放資源"}

    async def start_record(self, duration_sec: int) -> str:
        """Validate and spawn recording background task."""
        if self.status.state in ("recording", "processing"):
            raise ValueError(f"盒子目前正處於 {self.status.state} 狀態，請等待完成或點擊取消。")

        # Capture running loop for thread-safe cross-thread calls
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        # Always instantiate brand new cancellation tokens to avoid race condition with zombie tasks
        self._cancellation_token = asyncio.Event()
        self._thread_cancellation_token = threading.Event()
        self._current_ffmpeg_proc = None
        self._record_task = None

        job_id = f"m_{int(time.time())}"
        self.status = JobStatus(
            state="recording",
            phase="record",
            progress=0,
            elapsed=0.0,
            total=float(duration_sec),
            eta=float(duration_sec),
            job=job_id,
            msg="正在進行耳麥錄音..."
        )
        self._current_task = asyncio.create_task(self._pipeline_worker(job_id, duration_sec))
        self.notify_status_update()
        return job_id

    async def _pipeline_worker(self, job_id: str, duration_sec: int):
        """Sequential pipeline: Record (CPU exclusive) -> Transcribe (Whisper 4t) -> Summarize (Ollama)."""
        raw_ogg = os.path.join(AUDIO_CACHE_DIR, f"{job_id}.ogg")
        start_dt = datetime.now()

        try:
            # --- PHASE 1: Recording ---
            t0 = time.time()
            self.status.metrics = {
                "job_id": job_id,
                "record_status": "running",
                "record_elapsed": 0.0,
                "record_total": float(duration_sec),
                "audio_file": f"{job_id}.ogg",
                "stt_status": "waiting",
                "stt_elapsed": 0.0,
                "stt_duration": None,
                "stt_rtf": None,
                "stt_chars": 0,
                "stt_paragraphs": 0,
                "stt_engine": "Hailo-8 NPU (26 TOPS)",
                "llm_status": "waiting",
                "llm_elapsed": 0.0,
                "llm_duration": None,
                "llm_tokens": 0,
                "llm_tps": 0.0,
                "total_latency": None
            }

            async def _on_vu(vu_data: Dict[str, Any]):
                try:
                    from ws_manager import ws_manager
                    await ws_manager.broadcast("vu", vu_data)
                except Exception:
                    pass

            def _on_proc(proc):
                self._current_ffmpeg_proc = proc

            self._record_task = asyncio.create_task(
                record_audio_from_mic(
                    duration_sec,
                    raw_ogg,
                    vu_callback=_on_vu,
                    cancellation_event=self._cancellation_token,
                    on_proc_started=_on_proc
                )
            )

            # Progress ticker during recording
            while not self._record_task.done():
                if self._cancellation_token.is_set():
                    break
                elapsed = time.time() - t0
                self.status.elapsed = round(elapsed, 1)
                self.status.progress = min(100, int((elapsed / duration_sec) * 100))
                self.status.eta = max(0.0, float(duration_sec - elapsed))
                self.status.metrics["record_elapsed"] = round(elapsed, 1)
                self.notify_status_update()
                await asyncio.sleep(0.5)

            if self._cancellation_token.is_set():
                print("[MeetingMinutes] Recording aborted by cancellation token.")
                if self._record_task and not self._record_task.done():
                    self._record_task.cancel()
                return

            record_ok = await self._record_task
            rec_finish_time = time.time()
            self.status.metrics["record_status"] = "done"
            self.status.metrics["record_elapsed"] = float(duration_sec)

            # Broadcast reset frame upon recording completion
            try:
                from ws_manager import ws_manager
                await ws_manager.broadcast("vu", {
                    "rms": 0.0,
                    "peak": 0.0,
                    "db": -60.0,
                    "level_pct": 0,
                    "status": "idle",
                    "bands": [0] * 8,
                    "active": False
                })
            except Exception:
                pass

            if not record_ok or not os.path.exists(raw_ogg) or os.path.getsize(raw_ogg) == 0:
                if self._cancellation_token.is_set():
                    return
                raise RuntimeError("麥克風收音失敗，請確認藍牙耳麥已開機並保持連線。")

            # 錄音完成立即向前端廣播音檔就緒事件，波形圖 0 延遲秒開試聽！
            try:
                from ws_manager import ws_manager
                from two_layer_audio import extract_waveform_peaks
                peaks = extract_waveform_peaks(raw_ogg, num_peaks=200)
                await ws_manager.broadcast("audio_ready", {
                    "job_id": job_id,
                    "audio_url": f"/audio/{job_id}.ogg",
                    "duration": duration_sec,
                    "peaks": peaks
                })
            except Exception:
                pass

            # --- PHASE 2: Transcribe (Whisper STT) ---
            if self._cancellation_token.is_set():
                return

            self.status.state = "processing"
            self.status.phase = "transcribe"
            self.status.progress = 30
            self.status.metrics["stt_status"] = "running"
            est_stt_sec = max(10.0, float(duration_sec) * 0.26)
            self.status.eta = est_stt_sec
            self.status.msg = f"🎧 錄音已就緒！AI 正在進行語音聽寫（預計需 {int(est_stt_sec)} 秒）..."
            self.notify_status_update()

            stt_done_event = asyncio.Event()
            stt_start = time.time()

            async def _stt_ticker():
                while not stt_done_event.is_set() and not self._cancellation_token.is_set():
                    await asyncio.sleep(0.8)
                    elapsed_stt = time.time() - stt_start
                    rem_eta = max(1.0, est_stt_sec - elapsed_stt)
                    pct = min(72, int(30 + (elapsed_stt / est_stt_sec) * 42))
                    self.status.progress = pct
                    self.status.eta = round(rem_eta, 1)
                    self.status.metrics["stt_elapsed"] = round(elapsed_stt, 1)
                    self.status.msg = f"🎧 錄音完成！AI 正在聽寫語音（已耗時 {int(elapsed_stt)} 秒，剩餘約 {int(rem_eta)} 秒）..."
                    self.notify_status_update()

            loop = asyncio.get_running_loop()

            def _on_stt_segment(current_preview: str, new_seg: str, is_final: bool):
                self.status.transcript = current_preview
                paras = [p for p in current_preview.split("\n\n") if p.strip()]
                self.status.metrics["stt_chars"] = len(current_preview.replace("\n", "").strip())
                self.status.metrics["stt_paragraphs"] = len(paras)
                self.notify_status_update()
                try:
                    from ws_manager import ws_manager
                    asyncio.run_coroutine_threadsafe(
                        ws_manager.broadcast("transcript_stream", {
                            "job_id": job_id,
                            "transcript": current_preview,
                            "segment": new_seg,
                            "is_final": is_final
                        }),
                        loop
                    )
                except Exception:
                    pass

            ticker_task = asyncio.create_task(_stt_ticker())
            try:
                transcript = await asyncio.to_thread(
                    speech_to_text,
                    raw_ogg,
                    self._thread_cancellation_token,
                    None,
                    None,
                    _on_stt_segment
                )
            finally:
                stt_done_event.set()
                ticker_task.cancel()

            stt_dur = round(time.time() - stt_start, 2)
            stt_rtf = round(stt_dur / max(1.0, float(duration_sec)), 2)

            if self._cancellation_token.is_set():
                print("[MeetingMinutes] Transcribe aborted by cancellation token.")
                return

            if not transcript or not transcript.strip():
                transcript = "（錄音結束，未能從耳麥中辨識出清晰人聲語音）"

            paras_count = len([p for p in transcript.split("\n\n") if p.strip()])
            chars_count = len(transcript.replace("\n", "").strip())
            self.status.metrics["stt_status"] = "done"
            self.status.metrics["stt_duration"] = stt_dur
            self.status.metrics["stt_rtf"] = stt_rtf
            self.status.metrics["stt_chars"] = chars_count
            self.status.metrics["stt_paragraphs"] = paras_count

            # Intermediate result update: frontend can immediately inspect transcript
            self.status.transcript = transcript
            self.status.audio_url = f"/audio/{job_id}.ogg"
            self.status.phase = "summarize"
            self.status.progress = 65
            self.status.eta = 20.0
            self.status.msg = "🎧 逐字稿已出爐！AI 正在即時提煉會議紀要..."
            self.notify_status_update()

            self.result = MeetingResult(
                job=job_id,
                title="會議紀要",
                started_at=start_dt.isoformat(),
                duration=duration_sec,
                transcript=transcript,
                markdown=f"# 會議紀要\n\n## 逐字稿\n{transcript}"
            )

            # ⚡ [漸進式展示] 聽寫完成立即向前端推播逐字稿與音訊波形，0秒等待！
            try:
                from ws_manager import ws_manager
                await ws_manager.broadcast("transcript_ready", {
                    "job_id": job_id,
                    "transcript": transcript,
                    "duration": duration_sec,
                    "audio_url": f"/audio/{job_id}.ogg"
                })
            except Exception as b_err:
                print(f"[MeetingMinutes] Notice broadcasting transcript_ready: {b_err}")

            # --- PHASE 3: Summarize (Ollama Meeting Minutes LLM with Token Streaming) ---
            if self._cancellation_token.is_set():
                return

            self.status.metrics["llm_status"] = "running"
            llm_done_event = asyncio.Event()
            llm_start = time.time()

            async def _llm_ticker():
                while not llm_done_event.is_set() and not self._cancellation_token.is_set():
                    await asyncio.sleep(0.8)
                    elapsed_llm = time.time() - llm_start
                    pct = min(96, int(65 + (elapsed_llm / 25.0) * 30))
                    rem_eta = max(1.0, 25.0 - elapsed_llm)
                    self.status.progress = pct
                    self.status.eta = round(rem_eta, 1)
                    self.status.metrics["llm_elapsed"] = round(elapsed_llm, 1)
                    self.status.msg = f"🎧 逐字稿已就緒！AI 正在提煉紀要（串流打字中 {int(elapsed_llm)}s）..."
                    self.notify_status_update()

            llm_ticker_task = asyncio.create_task(_llm_ticker())
            try:
                raw_markdown = await self._generate_summary_llm(transcript, job_id=job_id)
            finally:
                llm_done_event.set()
                llm_ticker_task.cancel()

            llm_dur = round(time.time() - llm_start, 2)
            total_lat = round(time.time() - rec_finish_time, 2)
            self.status.metrics["llm_status"] = "done"
            self.status.metrics["llm_duration"] = llm_dur
            self.status.metrics["total_latency"] = max(0.0, total_lat)

            if self._cancellation_token.is_set():
                return

            parsed = parse_markdown_minutes(raw_markdown)

            formatted_dt = start_dt.strftime("%m-%d %H:%M")
            final_markdown = (
                f"# 會議紀要 · {formatted_dt} · {duration_sec} 秒\n\n"
                f"{raw_markdown.strip()}\n\n"
                f"## 逐字稿\n{transcript}"
            )

            self.result = MeetingResult(
                job=job_id,
                title="會議紀要",
                started_at=start_dt.isoformat(),
                duration=duration_sec,
                summary=parsed["summary"],
                todos=parsed["todos"],
                decisions=parsed["decisions"],
                transcript=transcript,
                markdown=final_markdown
            )

            # --- PHASE 4: Done ---
            self.status.state = "done"
            self.status.progress = 100
            self.status.eta = 0.0
            self.status.msg = "會議紀要已生成完畢。"
            self.notify_status_update()

        except asyncio.CancelledError:
            self.status.state = "idle"
            self.status.msg = "任務已手動中斷。"
            self.notify_status_update()
        except Exception as e:
            if self._cancellation_token.is_set():
                return
            self.status.state = "error"
            self.status.msg = f"處理失敗: {str(e)}"
            self.notify_status_update()

    async def regenerate(self) -> str:
        """Regenerate meeting summary from existing transcript without recording again."""
        if not self.result or not self.result.transcript:
            raise ValueError("目前沒有可重新整理的逐字稿紀錄，請先錄音。")

        if self.status.state in ("recording", "processing"):
            raise ValueError("盒子目前正忙碌中，請稍候再試。")

        job_id = self.result.job or f"m_{int(time.time())}"
        self.status = JobStatus(
            state="processing",
            phase="summarize",
            progress=50,
            eta=25.0,
            job=job_id,
            msg="正在重新整理會議紀要..."
        )

        async def _regen_worker():
            try:
                transcript = self.result.transcript
                raw_markdown = await self._generate_summary_llm(transcript, job_id=job_id)
                parsed = parse_markdown_minutes(raw_markdown)

                start_dt = datetime.fromisoformat(self.result.started_at) if self.result.started_at else datetime.now()
                formatted_dt = start_dt.strftime("%m-%d %H:%M")
                final_markdown = (
                    f"# 會議紀要 · {formatted_dt} · {self.result.duration} 秒\n\n"
                    f"{raw_markdown.strip()}\n\n"
                    f"## 逐字稿\n{transcript}"
                )

                self.result.summary = parsed["summary"]
                self.result.todos = parsed["todos"]
                self.result.decisions = parsed["decisions"]
                self.result.markdown = final_markdown

                self.status.state = "done"
                self.status.progress = 100
                self.status.eta = 0.0
                self.status.msg = "會議紀要已重新整理完畢。"
            except Exception as e:
                self.status.state = "error"
                self.status.msg = f"重新整理失敗: {str(e)}"

        asyncio.create_task(_regen_worker())
        return job_id

    async def _generate_summary_llm(self, transcript: str, job_id: Optional[str] = None) -> str:
        """Call Ollama LLM to synthesize meeting minutes with real-time SSE token streaming."""
        import httpx
        import json
        from ws_manager import ws_manager

        prompt = (
            "你是專業會議記錄員。根據下面的逐字稿，用繁體中文輸出符合以下格式的紀要：\n"
            "## 摘要\n"
            "3 條，每條不超過 25 字\n"
            "## 待辦\n"
            "Markdown 表格三欄：| 事項 | 負責人 | 時間 |。若未提及負責人或時間請填寫「待定」\n"
            "## 決議\n"
            "1–3 條，只寫已經定下來的事項\n"
            "嚴禁加入任何開場白、問候語或結尾廢話，直接輸出 Markdown 內容。\n\n"
            f"逐字稿：\n{transcript}"
        )

        accumulated = []
        target_models = [DEFAULT_MODEL, FALLBACK_MODEL]

        for model_name in target_models:
            accumulated.clear()
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    async with client.stream(
                        "POST",
                        DEFAULT_OLLAMA_URL,
                        json={
                            "model": model_name,
                            "prompt": prompt,
                            "stream": True,
                            "keep_alive": "24h",
                            "options": {"temperature": 0.2}
                        }
                    ) as resp:
                        if resp.status_code == 200:
                            async for line in resp.aiter_lines():
                                if self._cancellation_token and self._cancellation_token.is_set():
                                    print("[MeetingMinutes] LLM generation aborted by cancellation token.")
                                    return ""
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    chunk = json.loads(line)
                                    token = chunk.get("response", "")
                                    if token:
                                        accumulated.append(token)
                                        # 逐 Token 打字機串流廣播
                                        curr_text = "".join(accumulated)
                                        self.status.summary_stream = curr_text
                                        self.status.metrics["llm_tokens"] = len(accumulated)
                                        if job_id:
                                            await ws_manager.broadcast("summary_stream", {
                                                "job_id": job_id,
                                                "token": token,
                                                "accumulated": curr_text
                                            })
                                except Exception:
                                    pass

                            full_text = "".join(accumulated).strip()
                            if full_text:
                                self.status.summary_stream = full_text
                                self.status.metrics["llm_tokens"] = len(accumulated)
                                if job_id:
                                    await ws_manager.broadcast("summary_stream", {
                                        "job_id": job_id,
                                        "token": "",
                                        "accumulated": full_text
                                    })
                                return full_text
            except Exception as e:
                print(f"[MeetingMinutes] Streaming Ollama error with model {model_name}: {e}")

        if accumulated:
            return "".join(accumulated).strip()
        return "## 摘要\n- 會議內容已完整保留於逐字稿\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 確認逐字稿細節 | 全體 | 待定 |\n\n## 決議\n- 錄音與逐字稿存檔完成"

    async def transcribe_region(self, job_id: Optional[str], start_sec: float, end_sec: float) -> Dict[str, Any]:
        """
        Transcribe a specific time window from cached meeting audio.
        Task 12: 互動式音訊波形視覺化與剪輯 (Waveform Visualizer & Trimming - Wavesurfer.js)
        """
        target_ogg = None
        if job_id:
            cand = os.path.join(AUDIO_CACHE_DIR, f"{job_id}.ogg")
            if os.path.exists(cand):
                target_ogg = cand

        if not target_ogg and self.result and self.result.job:
            cand = os.path.join(AUDIO_CACHE_DIR, f"{self.result.job}.ogg")
            if os.path.exists(cand):
                target_ogg = cand

        if not target_ogg:
            # Find most recently modified .ogg file in audio cache
            try:
                ogg_files = [
                    os.path.join(AUDIO_CACHE_DIR, f) for f in os.listdir(AUDIO_CACHE_DIR)
                    if f.endswith(".ogg") and not f.startswith("tts_")
                ]
                if ogg_files:
                    ogg_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    target_ogg = ogg_files[0]
            except Exception:
                pass

        if not target_ogg or not os.path.exists(target_ogg):
            raise ValueError("找不到可供剪輯轉錄的音訊母帶，請先進行錄音。")

        print(f"[MeetingMinutes] Transcribing audio region {start_sec:.2f}s - {end_sec:.2f}s from {target_ogg}")
        transcript = await asyncio.to_thread(
            speech_to_text,
            target_ogg,
            self._thread_cancellation_token,
            start_sec,
            end_sec
        )

        clean_text = transcript.strip() if transcript else "（所選範圍內未檢測到清晰人聲語音）"
        return {
            "status": "success",
            "job_id": job_id or (self.result.job if self.result else "latest"),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "transcript": clean_text
        }


# Global singleton instance
minutes_manager = MeetingJobManager()
