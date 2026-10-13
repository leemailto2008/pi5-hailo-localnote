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
import requests
from pydantic import BaseModel, Field
import sys

# Ensure current module and telegram_agent directories are in sys.path
_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)

_agent_dir = os.path.abspath(os.path.join(_cur_dir, "..", "telegram_agent"))
if os.path.exists(_agent_dir) and _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

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
        self.offline_detector = OfflineDetector(check_interval=10.0)
        self._last_cpu_time: float = 0.0
        self._last_cpu_idle: int = 0
        self._last_cpu_total: int = 0
        self._cpu_cache: float = 0.0

    def notify_status_update(self):
        """Notify WebSocket subscribers of status update."""
        try:
            from ws_manager import ws_manager
            asyncio.create_task(ws_manager.broadcast("status", self.status.model_dump()))
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

    async def start_record(self, duration_sec: int) -> str:
        """Validate and spawn recording background task."""
        if self.status.state in ("recording", "processing"):
            raise ValueError(f"盒子目前正處於 {self.status.state} 狀態，請等待完成。")

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
            record_task = asyncio.create_task(record_audio_from_mic(duration_sec, raw_ogg))

            # Progress ticker during recording
            while not record_task.done():
                elapsed = time.time() - t0
                self.status.elapsed = round(elapsed, 1)
                self.status.progress = min(100, int((elapsed / duration_sec) * 100))
                self.status.eta = max(0.0, float(duration_sec - elapsed))
                self.notify_status_update()
                await asyncio.sleep(0.5)

            record_ok = await record_task
            if not record_ok or not os.path.exists(raw_ogg) or os.path.getsize(raw_ogg) == 0:
                raise RuntimeError("麥克風收音失敗，請確認藍牙耳麥已開機並保持連線。")

            # --- PHASE 2: Transcribe (Whisper STT) ---
            self.status.state = "processing"
            self.status.phase = "transcribe"
            self.status.progress = 35
            self.status.eta = 45.0
            self.status.msg = "錄音完成，正在進行語音聽寫..."
            self.notify_status_update()

            # Execute faster-whisper in worker thread (full 4 threads, zero-contention)
            transcript = await asyncio.to_thread(speech_to_text, raw_ogg)
            if not transcript or not transcript.strip():
                transcript = "（錄音結束，未能從耳麥中辨識出清晰人聲語音）"

            # Intermediate result update: frontend can immediately inspect transcript
            self.result = MeetingResult(
                job=job_id,
                title="會議紀要",
                started_at=start_dt.isoformat(),
                duration=duration_sec,
                transcript=transcript,
                markdown=f"# 會議紀要\n\n## 逐字稿\n{transcript}"
            )

            # --- PHASE 3: Summarize (Ollama Meeting Minutes LLM) ---
            self.status.phase = "summarize"
            self.status.progress = 75
            self.status.eta = 20.0
            self.status.msg = "聽寫完成，正在提煉會議紀要..."
            self.notify_status_update()

            raw_markdown = await self._generate_summary_llm(transcript)
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
            self.status.state = "error"
            self.status.msg = "錄音與處理任務已被取消。"
            self.notify_status_update()
        except Exception as e:
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
                raw_markdown = await self._generate_summary_llm(transcript)
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

    async def _generate_summary_llm(self, transcript: str) -> str:
        """Call Ollama LLM to synthesize meeting minutes in Traditional Chinese."""
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

        def _call_ollama():
            target_model = DEFAULT_MODEL
            try:
                resp = requests.post(
                    DEFAULT_OLLAMA_URL,
                    json={
                        "model": target_model,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "24h",
                        "options": {"temperature": 0.2}
                    },
                    timeout=90
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
            except Exception:
                pass

            # Fallback to general qwen2.5:3b
            try:
                resp = requests.post(
                    DEFAULT_OLLAMA_URL,
                    json={
                        "model": FALLBACK_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "24h",
                        "options": {"temperature": 0.2}
                    },
                    timeout=90
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
            except Exception as e:
                return f"## 摘要\n- LLM 處理發生逾時或未就緒 ({str(e)})\n\n## 待辦\n| 事項 | 負責人 | 時間 |\n|---|---|---|\n| 檢查模型服務 | 工程師 | 即刻 |\n\n## 決議\n- 維持目前本機逐字稿紀錄"

            return "（未能成功取得模型摘要）"

        return await asyncio.to_thread(_call_ollama)


# Global singleton instance
minutes_manager = MeetingJobManager()
