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
_HEALTH_TTL_SEC = 20.0


_HAILO_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_for_hailo(text: str) -> str:
    """Pre-escape text so hailo-ollama's internal JSON prompt rendering does not choke on it."""
    out = text.replace("\\", "\\\\")      # backslash -> literal \\  (re-parsed to a single backslash)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("\n", "\\n")           # newline  -> literal \n
    out = out.replace("\t", "\\t")           # tab      -> literal \t
    return _HAILO_CONTROL_RE.sub("", out)


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
        else:
            payload["keep_alive"] = "24h"
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
) -> Tuple[str, LLMBackend]:
    """Blocking completion. Tries backends in order; raises LLMError if all fail."""
    last_err: Optional[Exception] = None
    messages = build_messages(prompt, system)
    for backend in backends_in_order():
        payload = backend.build_chat_payload(messages, stream=False, temperature=temperature, num_predict=num_predict)
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
) -> Iterator[Tuple[str, bool, LLMBackend]]:
    """
    Blocking token generator yielding (token, is_done, backend).
    Falls back to the next backend only if the current one failed before emitting any token.
    """
    last_err: Optional[Exception] = None
    messages = build_messages(prompt, system)
    for backend in backends_in_order():
        payload = backend.build_chat_payload(messages, stream=True, temperature=temperature, num_predict=num_predict, top_p=top_p)
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
) -> Tuple[str, LLMBackend]:
    last_err: Optional[Exception] = None
    messages = build_messages(prompt, system)
    for backend in backends_in_order():
        payload = backend.build_chat_payload(messages, stream=False, temperature=temperature, num_predict=num_predict)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(backend.chat_url, json=payload)
                if resp.status_code != 200:
                    raise LLMError(f"{backend.label} HTTP {resp.status_code}: {resp.text[:200]}")
                return extract_token(resp.json()).strip(), backend
        except (httpx.HTTPError, ValueError, LLMError) as e:
            last_err = e
            logger.warning(f"[LLM] {backend.label} failed, trying next backend: {e}")
    raise LLMError(f"all LLM backends failed: {last_err}")


async def aiter_chat_stream(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    num_predict: int = 200,
    timeout: float = 90.0,
) -> AsyncIterator[Tuple[str, bool, LLMBackend]]:
    last_err: Optional[Exception] = None
    messages = build_messages(prompt, system)
    for backend in backends_in_order():
        payload = backend.build_chat_payload(messages, stream=True, temperature=temperature, num_predict=num_predict)
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
