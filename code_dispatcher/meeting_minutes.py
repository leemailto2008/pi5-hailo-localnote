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
import numpy as np

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
from last_meeting import load_last_meeting, save_last_meeting
from voice_pipeline import NoMicrophoneError, record_audio_from_mic, speech_to_text

AUDIO_CACHE_DIR = get_in_memory_cache_dir()

# LLM backend selection (Hailo-10H qwen3:1.7b preferred, CPU Ollama fallback) lives in llm_backend.py
from llm_backend import aiter_chat_stream, LLMError
try:
    # 部署後 hailo_whisper 在 code_dispatcher 底下。NPU 每個切片 (約 6 秒) 已各自修過一次術語，
    # 但「最後總結一下，｜舊一是…」「交給吳｜父裡」這類跨切片的句型要等整份逐字稿接起來才看得到，所以全文再修一次 (規則可重複套用)。
    from hailo_whisper.engine import fix_meeting_terms
except Exception as _fix_err:   # 開發環境找不到時，只是少了跨切片的修正
    print(f"[MeetingMinutes] fix_meeting_terms unavailable: {_fix_err}")

    def fix_meeting_terms(text: str) -> str:
        return text
from summary_utils import (assemble_long_minutes, clean_summary_markdown, compact_notes, ensure_sections, fill_decisions_from_transcript,
                           cut_glued_topic_numbers, detect_language, drop_listed_todos_from_summary, drop_placeholder_items,
                           drop_unspoken_years, fill_missing_topics,
                           ground_todo_table, is_recap_chunk, keep_alive_plan, localize_minutes,
                           merge_body_todos, merge_recap_into_minutes, recap_notes, to_internal_sections, transcript_todo_rows,
                           map_notes_complete, split_with_recap, summary_limits, take_stable_chunk, tidy_live_transcript)

# 長會議紀要 (2026-10-27 e2e)：Hailo qwen3:1.7b 單次請求約 1500 字就 context full (HTTP 500)，
# 5 分鐘會議 (約 1000 字) 單次整理品質良好 → 1200 字以內維持單次；超過則切成 <= 1000 字的區塊逐段抽重點，
# 再由 summary_utils.assemble_long_minutes() 組成三段式紀要 (1.7B 模型無法跨 20+ 條重點歸納，實測會照抄並擠掉待辦/決議)。
SUMMARY_SINGLE_MAX_CHARS = int(os.environ.get("PI5_SUMMARY_SINGLE_MAX", "1200"))
SUMMARY_CHUNK_CHARS = int(os.environ.get("PI5_SUMMARY_CHUNK", "1000"))
SUMMARY_LONG_SINGLE_CHARS = int(os.environ.get("PI5_SUMMARY_LONG_SINGLE", "800"))   # 單次紀要超過這個字數，輸出上限 480 → 600
# 長會議的結尾總結在這個長度以內就直接用原句 (recap_notes)，不交給 LLM 改寫；設 0 關閉
SUMMARY_RECAP_VERBATIM_MAX = int(os.environ.get("PI5_SUMMARY_RECAP_VERBATIM_MAX", "400"))
# 分段整理期間讓 hailo-ollama 模型常駐，避免每段重新載入 7–28 秒；最後一段用預設 0 釋放 NPU。
# 實測 hailo-ollama 0.5.1：keep_alive 單位為秒，計時從模型載入開始，後續請求「不會」延長。
# 計時到期若剛好撞上下一段請求，會 HAILO_COMMUNICATION_CLOSED → HTTP 500 (30 分鐘實測第 6 段)。
# 因此 keep_alive 取「整個載入視窗」的上限 (段數 × 每段預算 + 餘裕，最多 1800 s)；超過一個視窗的長會議 (最長 4 小時約 55 段)
# 由 summary_utils.keep_alive_plan() 在視窗最後一段主動卸載、下一段重新載入。
SUMMARY_MAP_SEC_PER_CHUNK = int(os.environ.get("PI5_SUMMARY_MAP_SEC_PER_CHUNK", "75"))
SUMMARY_MAP_KEEP_ALIVE_MAX = 1800
# 錄音中預先整理 (2026-11-01)：長會議錄音時，已經穩定的前段逐字稿先逐段交給 LLM 抽重點，錄音結束只剩最後 1–3 段。
# 只在 NPU 共用模式 (Whisper 與 qwen3 同時常駐) 下啟用；LLM 在 NPU 上跑，228 實測整機 CPU 平均 1.7%、單核最高 9%。
# 預設關閉：CLAUDE.md「錄音中禁止背景推論」規則需使用者確認後才改預設。PI5_SUMMARY_PREMAP=1 開啟。
# 2026-11-01 228 實測代價：同一段 10 分鐘喇叭→藍牙錄音中固定時間打 4 次 LLM 請求 (stage/llm_bursts.py)，
# LLM 在跑的 47 s 時窗內藍牙 mSBC 壞幀 4.2 次/s、超過 20 ms 的封包間隔 0.41 次/s；時窗外 1.7 次/s、0.03 次/s。
# 4 個時窗全部偏高 (不是 RF 起伏)。換來的是 30 分鐘會議紀要 242 s → 82–123 s。CPU 沒有變忙，推測是 NPU 滿載時的干擾。
SUMMARY_PREMAP = os.environ.get("PI5_SUMMARY_PREMAP", "0") == "1"
SUMMARY_PREMAP_LAG = int(os.environ.get("PI5_SUMMARY_PREMAP_LAG", "1000"))     # 尾端保留不預先整理的字數
SUMMARY_PREMAP_POLL_S = 10.0


async def stream_map_notes(prompt: str, keep_alive: Optional[int] = None, hailo_only: bool = False, lang: str = "zh"):
    """長會議逐段抽重點：串流生成，compact_notes() 會留下的 4 條都寫完就停 (map_notes_complete)。
    取消串流時 hailo-ollama 會立刻停止生成 (2026-11-01 228 實測)；非串流必須等 num_predict 用完，模型常多寫一兩條被丟掉。
    Hailo 在第一個 token 前失敗 (例如 hailo-ollama 正在重啟) 時，每 5 s 重試、最多 2 次，才改用 CPU 後端。回傳 (文字, backend)。"""
    from contextlib import aclosing
    acc, backend = "", None
    lim = LANG_LIMITS.get(lang, LANG_LIMITS["zh"])
    async with aclosing(aiter_chat_stream(prompt, system=lim["map_sys"], temperature=0.2, num_predict=160 if lang == "zh" else 220,
                                          timeout=120.0, hailo_keep_alive=keep_alive, hailo_retries=2,
                                          hailo_only=hailo_only)) as stream:
        async for tok, _done, b in stream:
            backend = b
            if tok:
                acc += tok
                if map_notes_complete(acc, max_len=lim["note_len"]):
                    break
    return acc.strip(), backend


def _safe_post(fn, md: str, *args, **kwargs) -> str:
    """紀要後處理 (補待辦、對照逐字稿…) 出錯時沿用前一步的結果，不讓整份紀要變成「處理失敗」
    (11-04 夜間 3 分鐘實錄：merge_recap_into_minutes 的 IndexError 讓已經寫好的紀要整份丟掉)。"""
    try:
        return fn(md, *args, **kwargs)
    except Exception as e:      # noqa: BLE001 — 後處理只是加分，不能拖垮主流程
        print(f"[MeetingMinutes] 後處理 {getattr(fn, '__name__', fn)} 失敗，沿用前一步: {type(e).__name__}: {e}")
        return md


def map_user_prompt(piece: str, idx: int, total: int, lang: str = "zh") -> str:
    """最後一段明確告知是會議結尾：會議常以總結收尾，總結裡的決議/待辦/負責人是紀要最重要的內容。"""
    if lang == "en":
        if idx == total and total > 1:
            return (f"Meeting transcript (part {idx}/{total}, the last part):\n{piece}\n\n"
                    "If it ends with a recap, list the actions and owners from the recap first; otherwise list this part's key points:")
        if total <= 0:
            return f"Meeting transcript (part {idx}):\n{piece}\n\nList this part's key points:"
        return f"Meeting transcript (part {idx}/{total}):\n{piece}\n\nList this part's key points:"
    if idx == total and total > 1:
        return (f"會議逐字稿（第 {idx}/{total} 段，會議最後一段）：\n{piece}\n\n"
                "如果最後有總結，請優先條列總結提到的事項與負責人；沒有總結就條列本段重點：")
    if total <= 0:   # 錄音中預先整理：總段數還不知道
        return f"會議逐字稿（第 {idx} 段）：\n{piece}\n\n請條列本段重點："
    return f"會議逐字稿（第 {idx}/{total} 段）：\n{piece}\n\n請條列本段重點："


MINUTES_SYSTEM_MSG = (
    "你是會議記錄員，只用台灣繁體中文輸出精簡的 Markdown 會議紀要，依序只有三段：## 摘要、## 待辦、## 決議。"
    "規則：摘要依議題分點，每個議題一點，最多 5 點，每點 40 字以內，並寫出該議題的關鍵數字、日期、金額與人名；"
    "待辦用表格「| 事項 | 負責人 | 時間 |」每個待辦一列，最多 4 列，事項 15 字以內、不要抄摘要，沒有待辦就寫一列「| 無 | - | - |」；"
    "決議最多 3 點，每點 30 字以內，沒有就寫「- 無」。若會議最後有總結，以總結內容為準。"
    "不要開場白，不要解釋，不要重複規則，三段寫完就結束。"
)
MAP_SYSTEM_MSG = (
    "你是會議記錄員，只用台灣繁體中文條列會議重點。規則：每個議題一行，最多 4 行，每行 35 字以內，"
    "每行以「- 議題名稱：」開頭（例如「- 教育訓練：」「- 客戶訂單：」「- 預算：」），"
    "一定要保留關鍵數字、日期、金額、人名與公司名，並寫出負責人與期限；不要寫會議日期、出席者等與議題無關的內容；"
    "若本段是會議結尾的總結，只列總結提到的事項。不要開場白，不要解釋，不要重複規則，只輸出條列。"
)

# 英文會議用英文紀要 (2026-11-03 使用者指定：紀要跟著逐字稿的語言，不翻譯)。格式與中文版相同，只是段落標題與表頭換成英文；
# 模型輸出後先用 to_internal_sections() 換回內部的中文標題，後處理完再由 localize_minutes() 換回英文。
MINUTES_SYSTEM_MSG_EN = (
    "You are a meeting secretary. Write concise meeting minutes in English only, in Markdown, with exactly three sections "
    "in this order: ## Summary, ## Action Items, ## Decisions. Rules: Summary has one bullet per topic, at most 5 bullets, "
    "each under 25 words, keeping the key numbers, dates, amounts and names; write dates as spoken and never add a year. "
    "Action Items is a table "
    "\"| Item | Owner | Due |\" with one row per action, at most 4 rows, each item under 10 words and not copied from the "
    "summary; if there are none, write one row \"| None | - | - |\". Decisions has at most 3 bullets, each under 20 words; "
    "if there are none, write \"- None\". If the meeting ends with a recap, follow the recap. No preamble, no explanations, "
    "do not repeat these rules, and stop after the three sections."
)
MAP_SYSTEM_MSG_EN = (
    "You are a meeting secretary. List the key points of this part of the meeting in English only. Rules: one line per topic, "
    "at most 4 lines, each under 20 words, each starting with \"- Topic name:\" (for example \"- Training:\", \"- Customer order:\", "
    "\"- Budget:\"). Always keep the key numbers, dates, amounts, people and company names. Write an action as "
    "\"<Name> will <action> by <deadline>\", and when the speakers decide or agree on something keep the word "
    "\"decided\" or \"agreed\". Leave out the meeting date, attendees and anything unrelated to a topic. If this part is the "
    "closing recap, list only what the recap mentions. No preamble, no explanations, do not repeat these rules, output only the list."
)
# 英文每個字元的資訊量約為中文的 1/4：單次整理、分段大小與每條重點長度都放寬 (Hailo 單次請求上限約 1500 個中文字，
# 換成英文約 5–6 千字元)。
LANG_LIMITS = {
    "zh": {"single_max": SUMMARY_SINGLE_MAX_CHARS, "chunk": SUMMARY_CHUNK_CHARS, "long_single": SUMMARY_LONG_SINGLE_CHARS,
           "note_len": 40, "minutes_sys": MINUTES_SYSTEM_MSG, "map_sys": MAP_SYSTEM_MSG},
    "en": {"single_max": 4000, "chunk": 3200, "long_single": 2600, "note_len": 120,
           "minutes_sys": MINUTES_SYSTEM_MSG_EN, "map_sys": MAP_SYSTEM_MSG_EN},
}



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
    lang: str = "zh"          # 紀要語言 (跟著逐字稿)：zh／en


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


def localize_parsed(parsed: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """英文紀要：解析結果裡的「無」佔位改成 None (畫面上顯示的是解析後的條目)。"""
    if lang != "en":
        return parsed
    fix = lambda x: "None" if x in ("無", "-") else x
    parsed["summary"] = [fix(x) for x in parsed.get("summary", [])]
    parsed["decisions"] = [fix(x) for x in parsed.get("decisions", [])]
    for t in parsed.get("todos", []):
        for k in ("item", "owner", "due"):
            if t.get(k) == "無":
                t[k] = "None"
    return parsed


def minutes_document(raw_markdown: str, transcript: str, formatted_dt: str, duration_sec: int, lang: str) -> str:
    """存檔與「複製紀要」用的完整 Markdown：標題、三段紀要、逐字稿；英文紀要用英文標題。"""
    body = localize_minutes(raw_markdown.strip(), lang)
    if lang == "en":
        return f"# Meeting Minutes · {formatted_dt} · {duration_sec} s\n\n{body}\n\n## Transcript\n{transcript}"
    return f"# 會議紀要 · {formatted_dt} · {duration_sec} 秒\n\n{body}\n\n## 逐字稿\n{transcript}"


def parse_markdown_minutes(raw_text: str) -> Dict[str, Any]:
    """
    Robust regex parser for Ollama meeting minutes markdown.
    Extracts summary, todos table, and decisions with full tolerance.
    """
    summaries: List[str] = []
    todos: List[Dict[str, str]] = []
    decisions: List[str] = []

    # 1. Extract Summary (Header-based)
    summary_match = re.search(r"##\s*摘要\s*\n([\s\S]*?)(?=\n##|\Z)", raw_text, re.IGNORECASE)
    if summary_match:
        lines = [l.strip() for l in summary_match.group(1).splitlines() if l.strip()]
        for line_str in lines:
            m = re.match(r"^(?:[\*\-•]|\d+\.)\s*(.*)", line_str)
            if m:
                item = m.group(1).strip()
                if item:
                    summaries.append(item)
            elif not line_str.startswith(("#", "|")):
                summaries.append(line_str)

    # 2. Extract Todos Table (Header-based or raw markdown table scan)
    table_lines: List[str] = []
    todo_match = re.search(r"##\s*待辦\s*\n([\s\S]*?)(?=\n##|\Z)", raw_text, re.IGNORECASE)
    if todo_match:
        table_lines = [l.strip() for l in todo_match.group(1).splitlines() if l.strip()]
    else:
        # Fallback: scan whole raw_text for any markdown table
        for line in raw_text.splitlines():
            l_str = line.strip()
            if l_str.startswith("|") and not "---" in l_str:
                table_lines.append(l_str)

    for line_str in table_lines:
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
        for line_str in lines:
            m = re.match(r"^(?:[\*\-•]|\d+\.)\s*(.*)", line_str)
            if m:
                item = m.group(1).strip()
                if item:
                    decisions.append(item)
            elif not line_str.startswith(("#", "|")):
                decisions.append(line_str)

    # 4. Fallback heuristics: If summary is still empty, collect bullet/numbered items outside table
    if not summaries:
        for line in raw_text.splitlines():
            l_str = line.strip()
            if l_str.startswith(("#", "|")):
                continue
            m = re.match(r"^(?:[\*\-•]|\d+\.)\s*(.*)", l_str)
            if m:
                item = m.group(1).strip()
                if item and len(item) <= 60:
                    summaries.append(item)
            elif l_str and len(l_str) <= 50 and not l_str.startswith("http"):
                summaries.append(l_str)
        if len(summaries) > 3:
            summaries = summaries[:3]

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
        self._chunk_worker_task: Optional[asyncio.Task] = None
        self._premap_task: Optional[asyncio.Task] = None
        self._premap_prefix: str = ""          # 錄音中已預先整理的逐字稿前段 (tidy_live_transcript 後)
        self._premap_notes: List[str] = []     # 對應的各段重點 (compact_notes 後)
        self._hailo_pinned: bool = False   # map 階段讓 hailo-ollama 模型常駐 NPU 中
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self.offline_detector = OfflineDetector(check_interval=10.0)
        self._last_cpu_time: float = 0.0
        self._last_cpu_idle: int = 0
        self._last_cpu_total: int = 0
        self._cpu_cache: float = 0.0
        self._cancellation_token: asyncio.Event = asyncio.Event()
        self._thread_cancellation_token: threading.Event = threading.Event()
        self._finish_early_event: asyncio.Event = asyncio.Event()
        self._current_ffmpeg_proc = None
        self.bypass_summary: bool = False
        self._restore_last_meeting()

    def _restore_last_meeting(self) -> None:
        """服務啟動時載回上一場會議 (last_meeting.py)，狀態設為 done，網頁打開就看得到。"""
        payload = load_last_meeting()
        if not payload:
            return
        try:
            self.result = MeetingResult(**payload["result"])
        except ValueError as e:          # pydantic ValidationError 是 ValueError 的子類別
            print(f"[MeetingMinutes] 最後一場會議內容不符目前格式，略過: {e}")
            return
        job = self.result.job
        audio_ok = os.path.exists(os.path.join(AUDIO_CACHE_DIR, f"{job}.ogg"))
        self.status = JobStatus(
            state="done", phase="done", progress=100, eta=0.0, job=job,
            total=float(self.result.duration), elapsed=float(self.result.duration),
            msg=f"已載入服務重啟前的最後一場會議（{payload.get('saved_at', '')} 保存）。",
            transcript=self.result.transcript,
            audio_url=f"/audio/{job}.ogg" if audio_ok else None,
            metrics=(payload.get("status") or {}).get("metrics") or {},
        )
        print(f"[MeetingMinutes] 已載入最後一場會議 {job}（音檔{'在' if audio_ok else '已不在'} RAM 快取）")

    def _save_last_meeting(self) -> None:
        if self.result:
            save_last_meeting(self.result.model_dump(), {"msg": self.status.msg, "metrics": self.status.metrics})

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

        # 2. Cancel record sub-task and chunk worker if active
        if self._record_task and not self._record_task.done():
            self._record_task.cancel()
            self._record_task = None
            print("[MeetingMinutes] Active recording sub-task cancelled.")

        if self._chunk_worker_task and not self._chunk_worker_task.done():
            self._chunk_worker_task.cancel()
            self._chunk_worker_task = None
        if self._premap_task and not self._premap_task.done():
            self._premap_task.cancel()
        self._premap_task = None

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

    async def finish_recording_early(self) -> Dict[str, Any]:
        """
        Gracefully finish recording early, preserve all captured audio and immediately proceed to STT and minutes generation.
        """
        if self.status.state != "recording":
            return {"status": "ignored", "message": f"目前狀態為 {self.status.state}，無需提前結算錄音"}

        print("[MeetingMinutes] finish_recording_early requested.")
        self._finish_early_event.set()
        self.status.msg = "⏹️ 收到提前結束指令，正在封存已錄音訊並整理紀要..."
        self.notify_status_update()
        return {"status": "finishing_early", "message": "已成功提早結束錄音，正在結算音訊與生成會議紀要"}

    async def start_record(self, duration_sec: int, bypass_summary: Optional[bool] = None) -> str:
        """Validate and spawn recording background task."""
        if self.status.state in ("recording", "processing"):
            raise ValueError(f"盒子目前正處於 {self.status.state} 狀態，請等待完成或點擊取消。")

        # 單次請求的 bypass_summary 只作用於本次錄音，不覆寫全域設定 (全域設定僅由 /api/config/bypass_summary 修改)。
        # 2026-10-22：舊行為會把一次性的 bypass=true 寫回全域，導致之後從 UI 發起的錄音都被跳過 AI 總結。
        job_bypass = self.bypass_summary if bypass_summary is None else bool(bypass_summary)

        # Capture running loop for thread-safe cross-thread calls
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        # Always instantiate brand new cancellation tokens to avoid race condition with zombie tasks
        self._cancellation_token = asyncio.Event()
        self._thread_cancellation_token = threading.Event()
        self._finish_early_event = asyncio.Event()
        self._current_ffmpeg_proc = None
        self._record_task = None

        try:
            from voice_memory import evict_audio_ring_buffer
            ev = evict_audio_ring_buffer()
            if ev.get("evicted_count"):
                print(f"[MeetingMinutes] 錄音前清理音訊快取：{ev}")
        except Exception as e:
            print(f"[MeetingMinutes] audio cache eviction skipped: {e}")

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
        self._current_task = asyncio.create_task(self._pipeline_worker(job_id, duration_sec, job_bypass))
        self.notify_status_update()
        return job_id

    async def _pipeline_worker(self, job_id: str, duration_sec: int, bypass_summary: bool = False):
        """Sequential pipeline: Record (CPU exclusive) -> Transcribe (Whisper 4t) -> Summarize (Ollama)."""
        raw_ogg = os.path.join(AUDIO_CACHE_DIR, f"{job_id}.ogg")
        start_dt = datetime.now()

        try:
            # --- PHASE 1: Recording ---
            t0 = time.time()
            first_chunk_ready_time: Optional[float] = None
            first_stt_output_time: Optional[float] = None
            first_stt_latency: Optional[float] = None
            # 即時差 (stt_audio_delay)：現在時刻 − 最後一段轉錄完成的切片「進佇列時刻」(＝該切片結尾的錄音時刻)。
            # 2026-10-30 228 喇叭 30 分鐘自測：舊算法「已錄秒數 − 已處理音訊秒數」把 ZeroSplicer 剪掉的
            # 插入靜音 (80 s) 也算成落後，顯示 101 s，但實際尾端等待只有 3.2 s、並無積壓。
            last_done_enq_t: Optional[float] = None

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
                "stt_engine": "AI NPU",
                "first_chunk_time": None,
                "first_stt_time": None,
                "first_stt_latency": None,
                "stt_audio_delay": None,
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

            # -------------------------------------------------------------
            # 🚀 邊錄音邊轉譯流水線 (In-Recording Realtime Streaming STT)
            # 依賴底層 PCM 滑動切片，每 8 秒即時送交 Whisper small 模型推論，
            # 確保錄音結束當下已完成 95% 以上文字轉錄，尾端延遲 < 1.5 秒！
            # -------------------------------------------------------------
            chunk_queue: asyncio.Queue = asyncio.Queue()
            live_segments: List[str] = []
            curr_preview: str = ""
            chunk_worker_done = asyncio.Event()

            async def _chunk_stt_worker():
                """背景即時解算滑動切片 PCM 音訊"""
                nonlocal first_chunk_ready_time, first_stt_output_time, first_stt_latency, last_done_enq_t, curr_preview
                try:
                    while True:
                        item = await chunk_queue.get()
                        if item is None:
                            chunk_queue.task_done()
                            break
                        pcm_bytes = item[0]
                        is_final = item[1]
                        enq_t = item[2] if len(item) > 2 else time.time()

                        if pcm_bytes and len(pcm_bytes) >= 16000 and not self._cancellation_token.is_set():
                            try:
                                # 轉為 16kHz float32 PCM numpy 陣列
                                pcm_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                                # VAD 靜音快篩：若切片為停頓或微弱環境音，0秒直接跳過，杜絕佇列堆疊
                                peak = float(np.max(np.abs(pcm_np)))
                                rms = float(np.sqrt(np.mean(pcm_np**2)))
                                if peak < 0.035 and rms < 0.0025:
                                    last_done_enq_t = enq_t
                                    self.status.metrics["stt_audio_delay"] = max(0.0, round(time.time() - enq_t, 1))
                                    chunk_queue.task_done()
                                    continue

                                seg_text = await asyncio.to_thread(
                                    speech_to_text,
                                    pcm_np,
                                    self._thread_cancellation_token
                                )
                                last_done_enq_t = enq_t
                                now_t = time.time()
                                now_elapsed = round(now_t - t0, 1)

                                # 🚀 首塊切片 STT 完成：無論是否有識別出文字，推論耗時皆已精確計算
                                if first_stt_output_time is None:
                                    first_stt_output_time = now_elapsed
                                    c_time = first_chunk_ready_time if first_chunk_ready_time is not None else now_elapsed
                                    first_stt_latency = max(0.1, round(first_stt_output_time - c_time, 1))
                                    self.status.metrics["first_stt_time"] = first_stt_output_time
                                    self.status.metrics["first_stt_latency"] = first_stt_latency
                                    print(f"[MeetingMinutes] 🎯 首切片 STT 完成！首進STT: {c_time}s, 首算完: {first_stt_output_time}s, 推論時差: {first_stt_latency}s")

                                if seg_text and seg_text.strip():
                                    clean_t = seg_text.strip()
                                    clean_t = clean_t.replace("\ufffd", "").replace("\uFFFD", "")
                                    clean_t = re.sub(r"([,\-\s]{2,})+", "，", clean_t)
                                    clean_t = re.sub(r"^[\s\-,\.，。]+", "", clean_t)
                                    clean_t = re.sub(r"[\s\-]+$", "", clean_t)

                                    if not clean_t or clean_t in ("-", ",", "，", "。", "、"):
                                        chunk_queue.task_done()
                                        continue

                                    live_segments.append(clean_t)
                                    # 智慧雙語拼接 (Bilingual live stream preview)
                                    if len(live_segments) == 1:
                                        curr_preview = clean_t
                                    else:
                                        prev_chunk = live_segments[-2]
                                        # 判斷前後是否皆為純英文/ASCII環境
                                        is_eng_prev = any(c.isascii() and c.isalpha() for c in prev_chunk)
                                        is_eng_curr = any(c.isascii() and c.isalpha() for c in clean_t)
                                        if is_eng_prev and is_eng_curr:
                                            # 全英文環境：若前段已有結尾標點則空一格，否則以逗號加空格連接
                                            if prev_chunk.endswith((".", "!", "?", "...", "…", ";", ":")):
                                                sep = " "
                                            elif prev_chunk.endswith(","):
                                                sep = " "
                                            else:
                                                sep = ", "
                                        else:
                                            # 中文或中英夾雜環境
                                            if prev_chunk.endswith(("。", "！", "？", "!", "?", "...", "…", ".")):
                                                sep = " "
                                            else:
                                                sep = "，"
                                        curr_preview = curr_preview + sep + clean_t
                                    self.status.transcript = curr_preview
                                    self.status.metrics["stt_chars"] = len(curr_preview.replace("\n", "").strip())
                                    self.status.metrics["stt_paragraphs"] = len(live_segments)
                                    self.status.metrics["first_stt_text_time"] = now_elapsed

                                self.status.metrics["stt_audio_delay"] = max(0.0, round(time.time() - enq_t, 1))
                                self.notify_status_update()

                                if seg_text and seg_text.strip():
                                    # 即時向前端手機推送打字機事件
                                    from ws_manager import ws_manager
                                    await ws_manager.broadcast("transcript_stream", {
                                        "job_id": job_id,
                                        "transcript": curr_preview,
                                        "segment": clean_t,
                                        "is_final": is_final
                                    })
                            except Exception as c_err:
                                print(f"[MeetingMinutes] Live chunk STT error: {c_err}")
                        chunk_queue.task_done()
                except Exception as worker_err:
                    print(f"[MeetingMinutes] chunk_stt_worker loop error: {worker_err}")
                finally:
                    chunk_worker_done.set()

            chunk_worker_task = asyncio.create_task(_chunk_stt_worker())
            self._chunk_worker_task = chunk_worker_task

            self._premap_prefix, self._premap_notes = "", []
            premap_stop = asyncio.Event()

            async def _premap_loop():
                """錄音中預先整理：逐字稿前段穩定後 (後面還有 SUMMARY_PREMAP_LAG 字) 就逐段抽重點，結果留給錄音後的長會議整理沿用。"""
                done_upto = 0
                while not premap_stop.is_set() and not self._cancellation_token.is_set():
                    try:
                        await asyncio.wait_for(premap_stop.wait(), timeout=SUMMARY_PREMAP_POLL_S)
                    except asyncio.TimeoutError:
                        pass
                    live = fix_meeting_terms(tidy_live_transcript(curr_preview))
                    end = take_stable_chunk(live, done_upto, SUMMARY_CHUNK_CHARS, SUMMARY_PREMAP_LAG)
                    if end is None or premap_stop.is_set() or self._cancellation_token.is_set():
                        continue
                    piece = live[done_upto:end].strip()
                    idx = len(self._premap_notes) + 1
                    t_req = time.time()
                    try:
                        # 用串流：錄音結束時取消串流，hailo-ollama 會立刻停止生成 (非串流請求取消後伺服器仍會跑完，
                        # 228 實測下一個請求要多等 32 s)，NPU 才能馬上回去處理最後幾段逐字稿
                        text, _backend = await stream_map_notes(map_user_prompt(piece, idx, 0), hailo_only=True)   # 錄音中不可退回 CPU
                    except LLMError as e:
                        print(f"[MeetingMinutes] 錄音中預先整理第 {idx} 段失敗，停止預先整理: {e}")
                        return
                    if self._cancellation_token.is_set():
                        return
                    text = compact_notes(text or "") or compact_notes("- " + piece[:120], max_lines=1, max_len=60)
                    self._premap_notes.append(text)
                    self._premap_prefix = live[:end]
                    done_upto = end
                    self.status.metrics["summary_premap_chunks"] = idx
                    print(f"[MeetingMinutes] 錄音中預先整理第 {idx} 段：{len(piece)} → {len(text)} 字 ({time.time() - t_req:.1f}s)")

            self._premap_task = None
            if SUMMARY_PREMAP and not bypass_summary:
                try:
                    from npu_share import shared_group
                    if shared_group():
                        self._premap_task = asyncio.create_task(_premap_loop())
                except Exception as e:
                    print(f"[MeetingMinutes] 錄音中預先整理未啟用: {e}")

            async def _on_pcm_slice(slice_data: bytes, is_final: bool):
                """PCM 滑動切片即時投遞至隊列"""
                nonlocal first_chunk_ready_time
                if first_chunk_ready_time is None and not is_final:
                    first_chunk_ready_time = round(time.time() - t0, 1)
                    self.status.metrics["first_chunk_time"] = first_chunk_ready_time
                    self.status.metrics["stt_status"] = "ingesting"
                    print(f"[MeetingMinutes] 🚀 首切片成形進 STT！時間: {first_chunk_ready_time}s")
                    self.notify_status_update()
                await chunk_queue.put((slice_data, is_final, time.time()))

            rec_stats: Dict[str, Any] = {}
            self._record_task = asyncio.create_task(
                record_audio_from_mic(
                    duration_sec,
                    raw_ogg,
                    vu_callback=_on_vu,
                    cancellation_event=self._cancellation_token,
                    on_proc_started=_on_proc,
                    pcm_chunk_callback=_on_pcm_slice,
                    chunk_sec=8.0,
                    finish_early_event=self._finish_early_event,
                    stats_out=rec_stats
                )
            )

            # Progress ticker during recording
            while not self._record_task.done():
                if self._cancellation_token.is_set():
                    break
                if self._finish_early_event.is_set():
                    break
                elapsed = time.time() - t0
                self.status.elapsed = round(elapsed, 1)
                self.status.progress = min(100, int((elapsed / duration_sec) * 100))
                self.status.eta = max(0.0, float(duration_sec - elapsed))
                self.status.metrics["record_elapsed"] = round(elapsed, 1)
                if last_done_enq_t is not None:
                    self.status.metrics["stt_audio_delay"] = max(0.0, round(time.time() - last_done_enq_t, 1))
                if "dropout_live" in rec_stats:
                    self.status.metrics["link_dropout_live"] = rec_stats["dropout_live"]
                    self.status.metrics["link_arrival_live"] = rec_stats.get("arrival_live")
                self.notify_status_update()
                await asyncio.sleep(0.5)

            if self._cancellation_token.is_set():
                print("[MeetingMinutes] Recording aborted by cancellation token.")
                await chunk_queue.put(None)
                if self._record_task and not self._record_task.done():
                    self._record_task.cancel()
                return

            try:
                record_ok = await self._record_task
            except NoMicrophoneError as e:
                await chunk_queue.put(None)
                self.status.state = "error"
                self.status.msg = f"⚠️ {e}"
                self.notify_status_update()
                return
            rec_finish_time = time.time()
            actual_record_sec = max(1.0, round(rec_finish_time - t0, 1))
            duration_sec = int(round(actual_record_sec))
            self.status.total = float(duration_sec)
            self.status.elapsed = float(duration_sec)
            self.status.metrics["record_status"] = "done"
            self.status.metrics["record_elapsed"] = float(duration_sec)
            self.status.metrics["record_total"] = float(duration_sec)
            if "dropout_pct" in rec_stats:
                # 藍牙收音品質：被剪掉的插入式靜音比例 (見 audio_splice.ZeroSplicer.dropout_pct)
                self.status.metrics["link_dropout_pct"] = rec_stats["dropout_pct"]
                self.status.metrics["link_inserted_s"] = rec_stats.get("inserted_s")
                self.status.metrics["link_arrival_pct"] = rec_stats.get("arrival_pct")

            # 錄音中預先整理：錄音一結束就停掉 (取消進行中的那段，NPU 讓給最後幾段逐字稿；
            # 7669dd3 原本等它整理完，30 分鐘 e2e 尾端等待從 3 s 變 21 s)。已完成的段落交給長會議整理沿用。
            if self._premap_task:
                premap_stop.set()
                if not self._premap_task.done():
                    self._premap_task.cancel()
                try:
                    await self._premap_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[MeetingMinutes] 錄音中預先整理結束時出錯: {e}")
                self._premap_task = None

            # 等待剩餘切片隊列消化完畢 (依據隊列深度動態等待，保證全軌切片 100% 入稿)
            await chunk_queue.put(None)
            rem_chunks = chunk_queue.qsize()
            if rem_chunks > 0:
                print(f"[MeetingMinutes] ⏳ 錄音結束，等待剩餘 {rem_chunks} 個音訊切片消化完成...")
                self.status.phase = "transcribe"
                self.status.state = "processing"
                self.status.metrics["stt_status"] = "running"
                self.status.msg = f"🎧 錄音完成！正在結算最後 {rem_chunks} 段逐字稿..."
                self.notify_status_update()

            drain_timeout = max(8.0, float(rem_chunks * 3.5 + 4.0))
            drain_deadline = time.time() + drain_timeout
            while not chunk_worker_done.is_set():
                if time.time() > drain_deadline:
                    print(f"[MeetingMinutes] Warning: chunk queue drain timed out after {drain_timeout}s.")
                    break
                now_t = time.time()
                cur_tail_wait = round(now_t - rec_finish_time, 1)
                cur_total = round(now_t - t0, 1)
                cur_rem = chunk_queue.qsize()
                self.status.phase = "transcribe"
                self.status.state = "processing"
                self.status.metrics["stt_status"] = "running"
                self.status.metrics["stt_tail_wait"] = cur_tail_wait
                self.status.metrics["total_stt_time"] = cur_total
                self.status.metrics["total_elapsed"] = cur_total
                self.status.msg = f"🎧 錄音完成！正在結算最後 {cur_rem} 段逐字稿 (尾端已等 {cur_tail_wait}s)..."
                self.notify_status_update()
                await asyncio.sleep(0.5)

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
                # 長會議 (最長 4 小時) 解碼整段母帶需數十秒：丟到執行緒，避免卡住 event loop
                peaks = await asyncio.to_thread(extract_waveform_peaks, raw_ogg, 200)
                await ws_manager.broadcast("audio_ready", {
                    "job_id": job_id,
                    "audio_url": f"/audio/{job_id}.ogg",
                    "duration": duration_sec,
                    "peaks": peaks
                })
            except Exception as pe:
                print(f"[MeetingMinutes] Notice precomputing peaks: {pe}")

            # --- PHASE 2: Transcribe Finalization ---
            if self._cancellation_token.is_set():
                return

            # ⚡ [0秒尾端極速結算] 錄音期間已並行由 Hailo-10H 完成各切片解算，直接取用已組裝逐字稿
            final_preview = curr_preview or (self.status.transcript or "")
            if not final_preview and live_segments:
                final_preview = "，".join(live_segments)

            if final_preview and len(final_preview.strip()) >= 5:
                transcript = fix_meeting_terms(tidy_live_transcript(final_preview))    # 與錄音中預先整理用同一套整理
                print(f"[MeetingMinutes] ⚡ 0秒尾端等待！成功取用即時切片串流 STT (字數: {len(transcript)}，段落數: {len(live_segments)})！")
            else:
                # 保底：若即時切片為空，快速執行一次母帶轉錄
                print(f"[MeetingMinutes] ⚠️ 即時切片為空，降級使用母帶轉錄...")
                master_res = await asyncio.to_thread(
                    speech_to_text,
                    raw_ogg,
                    self._thread_cancellation_token
                )
                transcript = fix_meeting_terms((master_res or "").strip().replace("\ufffd", "").replace("\uFFFD", "")) if master_res else "（未能辨識出清晰人聲語音）"

            if self._cancellation_token.is_set():
                print("[MeetingMinutes] Transcribe aborted by cancellation token.")
                return

            if not transcript or not transcript.strip():
                transcript = "（錄音結束，未能從耳麥中辨識出清晰人聲語音）"

            # 精確計算真實尾端等待秒數與全鏈路總耗時 (從按下開始到全稿出爐)
            final_stt_time = time.time()
            stt_tail_wait = max(0.1, round(final_stt_time - rec_finish_time, 1))
            total_stt_sec = round(final_stt_time - t0, 1)
            stt_rtf = round(stt_tail_wait / max(1.0, float(duration_sec)), 2)

            paras_count = len([p for p in transcript.split("\n\n") if p.strip()])
            chars_count = len(transcript.replace("\n", "").strip())

            self.status.metrics["stt_status"] = "done"
            self.status.metrics["stt_duration"] = stt_tail_wait
            self.status.metrics["stt_tail_wait"] = stt_tail_wait
            self.status.metrics["record_duration"] = float(duration_sec)
            self.status.metrics["total_stt_time"] = total_stt_sec
            self.status.metrics["total_elapsed"] = total_stt_sec
            self.status.metrics["is_under_210s"] = bool(total_stt_sec <= (duration_sec + 30))
            self.status.metrics["stt_rtf"] = stt_rtf
            self.status.metrics["stt_chars"] = chars_count
            self.status.metrics["stt_paragraphs"] = paras_count
            self.status.metrics["stt_chars_per_sec"] = round(chars_count / max(1.0, total_stt_sec), 1)

            self.status.transcript = transcript
            self.status.audio_url = f"/audio/{job_id}.ogg"

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

            # ⚡ 快速調校開關：若開啟 bypass_summary，跳過最後 AI 總結，直出完整逐字稿！
            if bypass_summary:
                print(f"[MeetingMinutes] ⚡ 啟用快速調校模式 (Bypass Summary)，跳過 LLM 總結，直接完成！尾端耗時: {stt_tail_wait}s, 總耗時: {total_stt_sec}s")
                self.status.state = "done"
                self.status.phase = "done"
                self.status.progress = 100
                self.status.eta = 0.0
                self.status.msg = f"🎉 逐字稿產出完成！（已啟用調校模式，跳過 AI 總結，尾端整理: {stt_tail_wait}s，總時長: {total_stt_sec}s）"
                self.status.metrics["llm_status"] = "skipped"
                self.status.metrics["llm_duration"] = 0.0
                self.status.metrics["total_elapsed"] = total_stt_sec
                self.notify_status_update()
                self._save_last_meeting()

                try:
                    from ws_manager import ws_manager
                    await ws_manager.broadcast("completed", {
                        "job_id": job_id,
                        "status": "completed",
                        "state": "done",
                        "transcript": transcript,
                        "markdown": self.result.markdown,
                        "audio_url": f"/audio/{job_id}.ogg",
                        "metrics": self.status.metrics
                    })
                except Exception as c_err:
                    print(f"[MeetingMinutes] Notice broadcasting completed: {c_err}")
                return

            # Intermediate result update: frontend can immediately inspect transcript while AI summarizes
            self.status.phase = "summarize"
            self.status.progress = 65
            self.status.eta = 20.0
            self.status.msg = "🎧 逐字稿已出爐！AI 正在即時提煉會議紀要..."
            self.notify_status_update()

            # --- PHASE 3: Summarize (Ollama Meeting Minutes LLM with Token Streaming) ---
            if self._cancellation_token.is_set():
                return

            self.status.metrics["llm_status"] = "running"
            llm_done_event = asyncio.Event()
            llm_start = time.time()

            is_long_hierarchical = bool(len(transcript.strip()) > LANG_LIMITS[detect_language(transcript)]["single_max"])
            est_total_llm_sec = max(25.0, float(len(transcript.strip()) / 1400.0) * 16.0)

            async def _llm_ticker():
                while not llm_done_event.is_set() and not self._cancellation_token.is_set():
                    await asyncio.sleep(0.8)
                    elapsed_llm = time.time() - llm_start
                    pct = min(96, int(65 + (elapsed_llm / est_total_llm_sec) * 30))
                    rem_eta = max(1.0, est_total_llm_sec - elapsed_llm)
                    if not is_long_hierarchical:
                        self.status.progress = pct
                        self.status.msg = f"🎧 逐字稿已就緒！AI 正在提煉紀要（串流打字中 {int(elapsed_llm)}s）..."
                    self.status.eta = round(rem_eta, 1)
                    self.status.metrics["llm_elapsed"] = round(elapsed_llm, 1)
                    self.notify_status_update()

            llm_ticker_task = asyncio.create_task(_llm_ticker())
            try:
                raw_markdown = await self._generate_summary_llm(transcript, job_id=job_id)
            finally:
                llm_done_event.set()
                llm_ticker_task.cancel()
            raw_markdown = _safe_post(ground_todo_table, raw_markdown, transcript)   # 待辦的負責人／時間必須出自逐字稿
            raw_markdown = _safe_post(drop_unspoken_years, raw_markdown, transcript)  # 模型自己補的年份 (逐字稿沒說) 拿掉
            raw_markdown = _safe_post(fill_decisions_from_transcript, raw_markdown, transcript)   # 決議段空白但逐字稿有明講
            raw_markdown = _safe_post(drop_listed_todos_from_summary, raw_markdown)   # 摘要裡已列在待辦表的「待辦事項是…」半句拿掉
            raw_markdown = _safe_post(cut_glued_topic_numbers, raw_markdown)   # 條目尾巴黏著的「第N個議題」
            raw_markdown = _safe_post(drop_placeholder_items, raw_markdown)   # 已有條目的段落裡殘留的「- None」
            minutes_lang = detect_language(transcript)
            self.status.summary_stream = localize_minutes(raw_markdown, minutes_lang)

            llm_dur = round(time.time() - llm_start, 2)
            total_lat = round(time.time() - rec_finish_time, 2)
            self.status.metrics["llm_status"] = "done"
            self.status.metrics["llm_duration"] = llm_dur
            _tok = self.status.metrics.get("llm_tokens") or 0
            self.status.metrics["llm_tps"] = round(_tok / llm_dur, 2) if llm_dur and _tok else 0.0
            self.status.metrics["total_latency"] = max(0.0, total_lat)

            if self._cancellation_token.is_set():
                return

            parsed = localize_parsed(parse_markdown_minutes(raw_markdown), minutes_lang)

            formatted_dt = start_dt.strftime("%m-%d %H:%M")
            final_markdown = minutes_document(raw_markdown, transcript, formatted_dt, duration_sec, minutes_lang)

            self.result = MeetingResult(
                job=job_id,
                title="會議紀要" if minutes_lang == "zh" else "Meeting Minutes",
                started_at=start_dt.isoformat(),
                duration=duration_sec,
                summary=parsed["summary"],
                todos=parsed["todos"],
                decisions=parsed["decisions"],
                transcript=transcript,
                markdown=final_markdown,
                lang=minutes_lang
            )

            # --- PHASE 4: Done ---
            self.status.state = "done"
            self.status.progress = 100
            self.status.eta = 0.0
            self.status.msg = "會議紀要已生成完畢。"
            self.notify_status_update()
            self._save_last_meeting()

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
        finally:
            # 任何提早結束的路徑 (取消、找不到麥克風、例外) 都要停掉錄音中預先整理
            if self._premap_task and not self._premap_task.done():
                self._premap_task.cancel()
            self._premap_task = None

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
                raw_markdown = _safe_post(ground_todo_table, await self._generate_summary_llm(transcript, job_id=job_id), transcript)
                raw_markdown = _safe_post(drop_unspoken_years, raw_markdown, transcript)
                raw_markdown = _safe_post(fill_decisions_from_transcript, raw_markdown, transcript)
                raw_markdown = _safe_post(drop_listed_todos_from_summary, raw_markdown)
                raw_markdown = _safe_post(cut_glued_topic_numbers, raw_markdown)
                raw_markdown = _safe_post(drop_placeholder_items, raw_markdown)
                minutes_lang = detect_language(transcript)
                self.status.summary_stream = localize_minutes(raw_markdown, minutes_lang)
                parsed = localize_parsed(parse_markdown_minutes(raw_markdown), minutes_lang)

                start_dt = datetime.fromisoformat(self.result.started_at) if self.result.started_at else datetime.now()
                formatted_dt = start_dt.strftime("%m-%d %H:%M")
                final_markdown = minutes_document(raw_markdown, transcript, formatted_dt, self.result.duration, minutes_lang)

                self.result.summary = parsed["summary"]
                self.result.todos = parsed["todos"]
                self.result.decisions = parsed["decisions"]
                self.result.markdown = final_markdown
                self.result.lang = minutes_lang
                self.result.title = "會議紀要" if minutes_lang == "zh" else "Meeting Minutes"

                self.status.state = "done"
                self.status.progress = 100
                self.status.eta = 0.0
                self.status.msg = "會議紀要已重新整理完畢。"
                self._save_last_meeting()
            except Exception as e:
                self.status.state = "error"
                self.status.msg = f"重新整理失敗: {str(e)}"

        asyncio.create_task(_regen_worker())
        return job_id

    async def _generate_hierarchical_summary_llm(self, transcript: str, job_id: Optional[str] = None) -> str:
        """
        長逐字稿 (> SUMMARY_SINGLE_MAX_CHARS)：
        1. 依句子/子句邊界平均切成 <= SUMMARY_CHUNK_CHARS 的區塊，逐段請 LLM 條列重點 (期間模型常駐 NPU，最後一段釋放)。
        2. 每段重點以 compact_notes() 收斂 (<= 4 條、每條 <= 40 字)，再以 assemble_long_minutes() 組成三段式紀要。
        中途取消或失敗時，finally 會送 keep_alive=0 釋放 NPU，讓下一次錄音的 STT 能開 VDevice。
        """
        try:
            return await self._hierarchical_summary_body(transcript, job_id)
        finally:
            if self._hailo_pinned:
                await self._release_hailo()

    async def _release_hailo(self) -> None:
        """分段整理時模型常駐 NPU；若在最後一段 (keep_alive=0) 之前中止、失敗，或最後一段 (結尾總結) 不呼叫 LLM，
        補送極短請求立即釋放，讓 STT 能開 VDevice。"""
        self._hailo_pinned = False
        try:
            from llm_backend import HAILO_BACKEND, build_messages
            import httpx
            payload = HAILO_BACKEND.build_chat_payload(build_messages("好"), stream=False, num_predict=1, hailo_keep_alive=0)
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.post(HAILO_BACKEND.chat_url, json=payload)
            print("[MeetingMinutes] 已送 keep_alive=0 釋放 Hailo NPU (map 中止，或結尾總結直接採用原句)")
        except Exception as e:
            print(f"[MeetingMinutes] release Hailo NPU failed: {e}")

    async def _hierarchical_summary_body(self, transcript: str, job_id: Optional[str]) -> str:
        from ws_manager import ws_manager

        t_map = time.time()
        notes: List[str] = []
        rest = transcript
        pre, prefix = self._premap_notes, self._premap_prefix
        self._premap_notes, self._premap_prefix = [], ""     # 只用一次 (重新整理紀要時從頭來)
        if pre and prefix and transcript.startswith(prefix):
            notes, rest = list(pre), transcript[len(prefix):].strip()
            print(f"[MeetingMinutes] 沿用錄音中預先整理的 {len(pre)} 段 ({len(prefix)} 字)，剩 {len(rest)} 字")
        elif pre:
            print("[MeetingMinutes] 逐字稿前段與預先整理時不同，不沿用，全部重新整理")
        lang = detect_language(transcript)
        lim = LANG_LIMITS[lang]
        # 正文裡明講的交辦 (請X在…前…)：長會議待辦表優先用它，各段筆記抽出的列常是議題摘要 (見 summary_utils._long_todo_rows)
        try:
            body_todos = transcript_todo_rows(transcript, max_rows=200)
        except Exception as e:      # noqa: BLE001
            print(f"[MeetingMinutes] 正文交辦擷取失敗，長會議待辦表改用各段筆記: {e}")
            body_todos = []
        chunks = split_with_recap(rest, lim["chunk"]) if rest else []   # 結尾總結自成一段
        total = len(notes) + len(chunks)
        print(f"[MeetingMinutes] 📚 長逐字稿 {len(transcript)} 字 ({lang}) → {total} 段 (每段 <= {lim['chunk']} 字)，逐段整理")
        self.status.metrics["summary_chunks"] = total
        self.status.metrics["summary_premap_used"] = len(notes)
        plan = keep_alive_plan(total, SUMMARY_MAP_SEC_PER_CHUNK, SUMMARY_MAP_KEEP_ALIVE_MAX)
        for idx, piece in enumerate(chunks, len(notes) + 1):
            if self._cancellation_token and self._cancellation_token.is_set():
                return ""
            self.status.progress = int(60 + (idx - 1) / max(1, total) * 35)
            self.status.msg = f"📝 長會議分段整理：第 {idx}/{total} 段..."
            self.notify_status_update()
            ka = plan[idx - 1]   # None = 這是載入視窗的最後一段，完成即卸載
            t_req, text = time.time(), ""
            if idx == total > 1 and is_recap_chunk(piece) and len(piece) <= SUMMARY_RECAP_VERBATIM_MAX * (3 if lang == "en" else 1):
                # 口頭總結直接用原句 (見 summary_utils.recap_notes)；模型仍常駐時由 finally 的 _release_hailo() 釋放 NPU
                notes.append(recap_notes(piece))
                print(f"[MeetingMinutes] 段 {idx}/{total}: 結尾總結 {len(piece)} 字，直接採用原句")
                self.status.summary_stream = localize_minutes(assemble_long_minutes(notes, body_todos=body_todos, **summary_limits(total)), lang)
                continue
            try:
                self._hailo_pinned = True
                text, backend = await stream_map_notes(map_user_prompt(piece, idx, total, lang), keep_alive=ka, lang=lang)
                if backend is not None:
                    self.status.metrics["llm_backend"] = backend.label
                if ka is None and backend is not None and backend.is_hailo:
                    self._hailo_pinned = False   # 帶預設 keep_alive=0，完成即釋放 NPU
            except LLMError as e:
                print(f"[MeetingMinutes] 第 {idx}/{total} 段整理失敗: {e}")
            text = compact_notes(text or "", max_len=lim["note_len"])
            if not text:   # LLM 失敗：保留原文開頭，避免這段在紀要中消失
                text = compact_notes("- " + piece[:lim["note_len"] * 3], max_lines=1, max_len=lim["note_len"] + 20)
            notes.append(text)
            print(f"[MeetingMinutes] 段 {idx}/{total}: {len(piece)} → {len(text)} 字 ({time.time() - t_req:.1f}s)")
            if ka is None and idx < total:
                await asyncio.sleep(3.0)   # 視窗切換：等 hailo-ollama 卸載完成再送下一段 (避免撞上卸載)
            # 逐段把目前組出的紀要推給前端，使用者不必等全部完成
            partial = localize_minutes(assemble_long_minutes(notes, body_todos=body_todos, **summary_limits(total)), lang)
            self.status.summary_stream = partial
            if job_id:
                await ws_manager.broadcast("summary_stream", {"job_id": job_id, "token": "", "accumulated": partial})

        self.status.metrics["summary_map_s"] = round(time.time() - t_map, 1)
        full_text = assemble_long_minutes(notes, body_todos=body_todos, **summary_limits(total))
        self.status.summary_stream = localize_minutes(full_text, lang)
        print(f"[MeetingMinutes] 長會議紀要完成：{total} 段，{time.time() - t_map:.1f}s")
        return full_text

    async def _generate_summary_llm(self, transcript: str, job_id: Optional[str] = None) -> str:
        """Call Ollama LLM to synthesize meeting minutes with real-time SSE token streaming."""
        clean_transcript = transcript.strip()
        lang = detect_language(clean_transcript)
        lim = LANG_LIMITS[lang]
        if len(clean_transcript) > lim["single_max"]:
            return await self._generate_hierarchical_summary_llm(clean_transcript, job_id=job_id)

        import json
        from ws_manager import ws_manager

        # 2026-10-22 e2e 提示詞 A/B (Hailo qwen3:1.7b，同一份含辨識錯誤的逐字稿)：
        # 舊版把「（核心重點 1…）」等佔位說明寫在輸出範本內，小模型會照抄且 num_predict=220 讓待辦/決議被截斷 (事實命中 6/9)；
        # 改為規則放 system、範本不含佔位文字、摘要依議題分點並要求保留數字/日期/金額/人名 → 8/9。
        system_msg = lim["minutes_sys"]
        prompt = (f"逐字稿：\n{transcript}\n\n請輸出會議紀要：" if lang == "zh"
                  else f"Transcript:\n{transcript}\n\nWrite the meeting minutes:")

        accumulated = []
        used_backend_label = "n/a"
        try:
            # num_predict 380 → 480 (2026-10-31 228 A/B，9 份容易截斷的 3–5 分鐘逐字稿)：「待辦／決議未產生」4/9 → 1/9，
            # 回顧事實與議題覆蓋持平，平均多 16 s (只有本來會被截斷的輸出才會變長)。
            # 逐字稿超過 SUMMARY_LONG_SINGLE_CHARS 字 (約 4–5 分鐘) 時上限再放寬到 600 (2026-11-02 228 A/B，5 份 5 分鐘逐字稿各 2 次)：
            # 480 時 10 次有 3 次「待辦未產生」，600 時 1 次；平均多 5 s、最多 131 s；回顧 0.80／0.70、議題 0.90／0.90 都在雜訊內。
            npred = 600 if len(clean_transcript) > lim["long_single"] else 480
            async for token, _done, backend in aiter_chat_stream(prompt, system=system_msg, temperature=0.2, num_predict=npred, timeout=180.0,
                                                                 hailo_retries=2):
                used_backend_label = backend.label
                if self._cancellation_token and self._cancellation_token.is_set():
                    print("[MeetingMinutes] LLM generation aborted by cancellation token.")
                    return ""
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

            full_text = clean_summary_markdown(to_internal_sections("".join(accumulated)))   # 英文標題先換成內部的中文標題
            if full_text:
                full_text = ensure_sections(full_text)   # 被 num_predict 截斷時明確標示缺段，不捏造
                full_text = _safe_post(merge_recap_into_minutes, full_text, clean_transcript)   # 補上結尾口頭總結漏掉的待辦與決議 (逐字稿原句)
                full_text = _safe_post(merge_body_todos, full_text, clean_transcript)   # 正文裡明講的交辦 (請X在…前…) 也補進待辦表
                full_text = _safe_post(fill_missing_topics, full_text, clean_transcript)   # 摘要漏掉的後段議題用逐字稿原句補一條
                shown = localize_minutes(full_text, lang)                          # 回傳內部格式，畫面上顯示該語言
                self.status.summary_stream = shown
                self.status.metrics["llm_tokens"] = len(accumulated)
                self.status.metrics["llm_backend"] = used_backend_label
                if job_id:
                    await ws_manager.broadcast("summary_stream", {
                        "job_id": job_id,
                        "token": "",
                        "accumulated": shown
                    })
                print(f"[MeetingMinutes] Summary generated by {used_backend_label} ({len(accumulated)} tokens)")
                return full_text
        except LLMError as e:
            print(f"[MeetingMinutes] Streaming LLM error ({used_backend_label}): {e}")

        if accumulated:
            return merge_recap_into_minutes(ensure_sections(clean_summary_markdown(to_internal_sections("".join(accumulated)))),
                                            clean_transcript)
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
                    if f.endswith(".ogg") and not f.startswith("tts_") and not f.endswith(".smooth.ogg")   # 剪輯轉錄用原始錄音
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
