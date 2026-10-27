# -*- coding: utf-8 -*-
"""
llm_backend.py — Unified LLM backend selector for Pi 5 Local Note.

Backends (in preference order, configurable with env PI5_LLM_BACKEND=hailo|cpu):
  1. hailo : hailo-ollama on 127.0.0.1:11435 running qwen3:1.7b on the Hailo-10H NPU
             (CPU stays idle -> zero contention with Bluetooth mSBC / WirePlumber).
  2. cpu   : stock Ollama on 127.0.0.1:11434 running qwen2.5:3b-opt (qwen2.5:3b fallback).

hailo-ollama API quirks verified on 2026-10-22 (hailo-ollama 0.5.1, oatpp server):
  - Only /api/chat honours a system message; /api/generate accepts "system" but ignores it.
  - "keep_alive" MUST be an integer or omitted; the Ollama-style string "24h" -> HTTP 500.
  - Responses carry no eval_duration / prompt_eval_duration -> TPS must be measured client-side.
  - "think": false is accepted (Qwen3 thinking disabled); streaming is NDJSON like Ollama.
  - Message content is embedded into an internal JSON string WITHOUT escaping control characters
    or backslashes (only double quotes are escaped). A raw LF/TAB or a backslash in the prompt ->
    HTTP 500 "Failed to render prompt". _sanitize_for_hailo() pre-escapes them; the model then
    receives the intended newlines because hailo-ollama re-parses that JSON.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

import httpx
import requests

logger = logging.getLogger("Pi5LLM")

HAILO_URL = "http://127.0.0.1:11435"
HAILO_MODEL = "qwen3:1.7b"
CPU_URL = "http://127.0.0.1:11434"
CPU_MODEL = "qwen2.5:3b-opt"
CPU_FALLBACK_MODEL = "qwen2.5:3b"

PREFERRED_BACKEND = os.environ.get("PI5_LLM_BACKEND", "hailo").strip().lower()
# hailo-ollama keep_alive (seconds, integer only). 0 = release the NPU right after each request so that
# HailoWhisper (STT) can open the VDevice during recording. Override with env PI5_HAILO_KEEP_ALIVE.
HAILO_KEEP_ALIVE = int(os.environ.get("PI5_HAILO_KEEP_ALIVE", "0"))
_HEALTH_TTL_SEC = 20.0


_HAILO_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_for_hailo(text: str) -> str:
    """Pre-escape text so hailo-ollama's internal JSON prompt rendering does not choke on it."""
    out = text.replace("\\", "\\\\")      # backslash -> literal \\  (re-parsed to a single backslash)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("\n", "\\n")           # newline  -> literal \n
    out = out.replace("\t", "\\t")           # tab      -> literal \t
    return _HAILO_CONTROL_RE.sub("", out)


def _free_npu_for_llm() -> None:
    """Hailo-10H 的 VDevice 一次只能給一個行程：送 Hailo LLM 請求前，先請 STT 子行程放開 NPU (沒開就什麼都不做)。"""
    try:
        from stt_worker import stt_worker
        stt_worker.release()
    except Exception as e:
        logger.debug(f"[LLM] stt worker release skipped: {e}")


class LLMError(RuntimeError):
    """Raised when every configured backend failed."""


@dataclass(frozen=True)
class LLMBackend:
    name: str          # "hailo" | "cpu"
    base_url: str
    model: str
    is_hailo: bool

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    @property
    def label(self) -> str:
        return f"{'Hailo-10H NPU' if self.is_hailo else 'CPU Ollama'} {self.model}"

    def build_chat_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        temperature: float = 0.2,
        num_predict: int = 200,
        top_p: Optional[float] = None,
        hailo_keep_alive: Optional[int] = None,
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
        if top_p is not None:
            options["top_p"] = top_p
        if self.is_hailo:
            messages = [{"role": m["role"], "content": _sanitize_for_hailo(m["content"])} for m in messages]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if self.is_hailo:
            payload["think"] = False
            # NPU 仲裁 (2026-10-22 e2e 實測)：Hailo-10H 的 VDevice 為單一行程獨占。hailo-ollama 未帶 keep_alive 時
            # 模型常駐 NPU 5 分鐘，期間 HailoWhisper 一律 HAILO_OUT_OF_PHYSICAL_DEVICES → STT 全部退回 CPU (錄音中 CPU ~100%)。
            # keep_alive=0：回應完成後約 3 秒即卸載並釋放 NPU，代價為每次冷載入約 +7~8 秒。
            # 長會議 map 階段會連續送多個請求：呼叫端可傳 hailo_keep_alive (秒) 讓模型暫時常駐，最後一個請求再用預設 0 釋放。
            payload["keep_alive"] = HAILO_KEEP_ALIVE if hailo_keep_alive is None else int(hailo_keep_alive)
        else:
            # 2026-10-27：LLM 以 Hailo NPU 為主，CPU Ollama 只是備援。原本 "24h" 讓 qwen2.5:3b 一旦被用到就常駐 2.4 GB
            # 到隔天 (e2e RAM 峰值 4.1 GB、swap +247 MB)；改為用完即卸載。
            payload["keep_alive"] = 0
            options["num_thread"] = 4
        return payload


HAILO_BACKEND = LLMBackend("hailo", HAILO_URL, HAILO_MODEL, True)
CPU_BACKEND = LLMBackend("cpu", CPU_URL, CPU_MODEL, False)
CPU_FALLBACK_BACKEND = LLMBackend("cpu", CPU_URL, CPU_FALLBACK_MODEL, False)

_health_cache: Dict[str, Tuple[float, bool]] = {}
_health_lock = threading.Lock()


def _probe_backend(backend: LLMBackend, timeout: float = 1.5) -> bool:
    try:
        resp = requests.get(f"{backend.base_url}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return False
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        return backend.model in names
    except (requests.RequestException, ValueError) as e:
        logger.debug(f"backend {backend.name} probe failed: {e}")
        return False


def is_backend_healthy(backend: LLMBackend, force_refresh: bool = False) -> bool:
    key = f"{backend.base_url}|{backend.model}"
    now = time.monotonic()
    with _health_lock:
        cached = _health_cache.get(key)
        if cached and not force_refresh and now - cached[0] < _HEALTH_TTL_SEC:
            return cached[1]
    ok = _probe_backend(backend)
    with _health_lock:
        _health_cache[key] = (now, ok)
    return ok


def backends_in_order(force_refresh: bool = False) -> List[LLMBackend]:
    """Preferred backend first (only if healthy), then CPU fallbacks."""
    order: List[LLMBackend] = []
    if PREFERRED_BACKEND == "hailo" and is_backend_healthy(HAILO_BACKEND, force_refresh):
        order.append(HAILO_BACKEND)
    order.append(CPU_BACKEND)
    order.append(CPU_FALLBACK_BACKEND)
    return order


def get_backend() -> LLMBackend:
    return backends_in_order()[0]


def build_messages(prompt: str, system: Optional[str] = None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def extract_token(chunk: Dict[str, Any]) -> str:
    """Token text from a chat (or generate) NDJSON chunk."""
    msg = chunk.get("message")
    if isinstance(msg, dict):
        return msg.get("content", "") or ""
    return chunk.get("response", "") or ""


# ------------------------------------------------------------------ sync (requests)

def chat_complete(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    num_predict: int = 200,
    timeout: float = 60.0,
    hailo_keep_alive: Optional[int] = None,
) -> Tuple[str, LLMBackend]:
    """Blocking completion. Tries backends in order; raises LLMError if all fail."""
    last_err: Optional[Exception] = None
    messages = build_messages(prompt, system)
    for backend in backends_in_order():
        if backend.is_hailo:
            _free_npu_for_llm()
        payload = backend.build_chat_payload(messages, stream=False, temperature=temperature, num_predict=num_predict, hailo_keep_alive=hailo_keep_alive)
        try:
            resp = requests.post(backend.chat_url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise LLMError(f"{backend.label} HTTP {resp.status_code}: {resp.text[:200]}")
            text = extract_token(resp.json()).strip()
            return text, backend
        except (requests.RequestException, ValueError, LLMError) as e:
            last_err = e
            logger.warning(f"[LLM] {backend.label} failed, trying next backend: {e}")
    raise LLMError(f"all LLM backends failed: {last_err}")


def iter_chat_stream(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    num_predict: int = 200,
    top_p: Optional[float] = None,
    timeout: float = 60.0,
    hailo_keep_alive: Optional[int] = None,
) -> Iterator[Tuple[str, bool, LLMBackend]]:
    """
    Blocking token generator yielding (token, is_done, backend).
    Falls back to the next backend only if the current one failed before emitting any token.
    """
    last_err: Optional[Exception] = None
    messages = build_messages(prompt, system)
    for backend in backends_in_order():
        if backend.is_hailo:
            _free_npu_for_llm()
        payload = backend.build_chat_payload(messages, stream=True, temperature=temperature, num_predict=num_predict, top_p=top_p, hailo_keep_alive=hailo_keep_alive)
        emitted = False
        try:
            with requests.post(backend.chat_url, json=payload, stream=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    raise LLMError(f"{backend.label} HTTP {resp.status_code}: {resp.text[:200]}")
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    chunk = json.loads(raw.decode("utf-8"))
                    tok = extract_token(chunk)
                    done = bool(chunk.get("done", False))
                    if tok:
                        emitted = True
                    yield tok, done, backend
                    if done:
                        return
            return
        except (requests.RequestException, ValueError, LLMError) as e:
            last_err = e
            if emitted:
                raise LLMError(f"{backend.label} stream interrupted: {e}") from e
            logger.warning(f"[LLM] {backend.label} stream failed before first token, trying next backend: {e}")
    raise LLMError(f"all LLM backends failed: {last_err}")


def prewarm(timeout: float = 45.0) -> Optional[LLMBackend]:
    """Load weights of the active backend. Returns the backend used, None on failure."""
    try:
        _, backend = chat_complete("Hello", system="Reply with one word.", num_predict=4, timeout=timeout)
        return backend
    except LLMError as e:
        logger.warning(f"[LLM] prewarm failed: {e}")
        return None


# ------------------------------------------------------------------ async (httpx)

async def achat_complete(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    num_predict: int = 200,
    timeout: float = 60.0,
    hailo_keep_alive: Optional[int] = None,
    hailo_retries: int = 0,
) -> Tuple[str, LLMBackend]:
    """hailo_retries：Hailo 失敗時先等 3 s 重試幾次再換下一個後端 (hailo-ollama 偶發 500 後通常重載即恢復)。"""
    last_err: Optional[Exception] = None
    messages = build_messages(prompt, system)
    for backend in backends_in_order():
        if backend.is_hailo:
            await asyncio.to_thread(_free_npu_for_llm)
        payload = backend.build_chat_payload(messages, stream=False, temperature=temperature, num_predict=num_predict, hailo_keep_alive=hailo_keep_alive)
        for attempt in range(1 + (hailo_retries if backend.is_hailo else 0)):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(backend.chat_url, json=payload)
                    if resp.status_code != 200:
                        raise LLMError(f"{backend.label} HTTP {resp.status_code}: {resp.text[:200]}")
                    return extract_token(resp.json()).strip(), backend
            except (httpx.HTTPError, ValueError, LLMError) as e:
                last_err = e
                logger.warning(f"[LLM] {backend.label} failed (attempt {attempt + 1}): {e}")
                if backend.is_hailo and attempt < hailo_retries:
                    await asyncio.sleep(3.0)
        logger.warning(f"[LLM] {backend.label} failed, trying next backend: {last_err}")
    raise LLMError(f"all LLM backends failed: {last_err}")


async def aiter_chat_stream(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    num_predict: int = 200,
    timeout: float = 90.0,
    hailo_keep_alive: Optional[int] = None,
) -> AsyncIterator[Tuple[str, bool, LLMBackend]]:
    last_err: Optional[Exception] = None
    messages = build_messages(prompt, system)
    for backend in backends_in_order():
        if backend.is_hailo:
            await asyncio.to_thread(_free_npu_for_llm)
        payload = backend.build_chat_payload(messages, stream=True, temperature=temperature, num_predict=num_predict, hailo_keep_alive=hailo_keep_alive)
        emitted = False
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", backend.chat_url, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise LLMError(f"{backend.label} HTTP {resp.status_code}: {body[:200]!r}")
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        chunk = json.loads(line)
                        tok = extract_token(chunk)
                        done = bool(chunk.get("done", False))
                        if tok:
                            emitted = True
                        yield tok, done, backend
                        if done:
                            return
            return
        except (httpx.HTTPError, ValueError, LLMError) as e:
            last_err = e
            if emitted:
                raise LLMError(f"{backend.label} stream interrupted: {e}") from e
            logger.warning(f"[LLM] {backend.label} stream failed before first token, trying next backend: {e}")
    raise LLMError(f"all LLM backends failed: {last_err}")
