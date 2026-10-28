# code_dispatcher/ws_manager.py
"""
WebSocket Connection Manager for LocalNote
Handles real-time status push, client subscriptions, and broadcast distribution.
"""

import asyncio
import json
import logging
from typing import Set, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("ws_manager")

# 2026-10-28 4 小時 e2e 實測：一個被瀏覽器凍結的分頁不再讀取 WebSocket，伺服器送出佇列卡住 192 KB，
# 而 broadcast() 的 await send_text() 沒有逾時 → 會議紀要流程在推送進度時永久卡住 (卡了 1 小時)。
# 每次送出最多等 SEND_TIMEOUT_S，逾時就視為斷線並移除，其他客戶端與呼叫端都不再被拖住。
SEND_TIMEOUT_S = 3.0


class ConnectionManager:
    """Singleton manager controlling active WebSocket sessions and telemetry broadcasts."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
        self._send_locks: Dict[WebSocket, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
            self._send_locks[websocket] = asyncio.Lock()
            self.subscriptions[websocket] = {
                "*", "system", "status", "bt", "vu",
                "audio_ready", "transcript_stream", "transcript_ready", "summary_stream"
            }
        logger.info(f"[WS] Client connected. Active clients: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
            self._send_locks.pop(websocket, None)
            self.subscriptions.pop(websocket, None)
        logger.info(f"[WS] Client disconnected. Active clients: {len(self.active_connections)}")

    def subscribe(self, websocket: WebSocket, topic: str, enable: bool = True):
        if websocket in self.subscriptions:
            if enable:
                self.subscriptions[websocket].add(topic)
            else:
                self.subscriptions[websocket].discard(topic)

    async def _send(self, websocket: WebSocket, payload: str) -> None:
        """送出單一訊息；同一連線的送出以鎖排序，含等鎖在內最多 SEND_TIMEOUT_S 秒 (逾時 raise asyncio.TimeoutError)。"""
        lock = self._send_locks.get(websocket)

        async def _do():
            if lock:
                async with lock:
                    await websocket.send_text(payload)
            else:
                await websocket.send_text(payload)

        await asyncio.wait_for(_do(), SEND_TIMEOUT_S)

    async def _drop(self, connections, reason: str = "") -> None:
        async with self._lock:
            connections = [ws for ws in connections if ws in self.active_connections]   # 已移除的不重複處理
            for ws in connections:
                self.active_connections.discard(ws)
                self._send_locks.pop(ws, None)
                self.subscriptions.pop(ws, None)
        if not connections:
            return
        for ws in connections:
            asyncio.create_task(self._close_quietly(ws))
        logger.warning(f"[WS] Dropped {len(connections)} unresponsive client(s) ({reason}). Active clients: {len(self.active_connections)}")

    @staticmethod
    async def _close_quietly(websocket: WebSocket) -> None:
        try:
            await asyncio.wait_for(websocket.close(code=1011), 1.0)
        except Exception:
            pass

    async def send_personal(self, websocket: WebSocket, event_type: str, data: Any):
        if websocket not in self.active_connections:   # 已因逾時被移除的連線不再送 (避免每次都再卡 SEND_TIMEOUT_S)
            return
        try:
            payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
            await self._send(websocket, payload)
        except asyncio.TimeoutError:
            await self._drop([websocket], f"send timeout {SEND_TIMEOUT_S:g}s")
        except Exception as e:
            logger.warning(f"[WS] Failed to send to single client: {e}")

    async def broadcast(self, event_type: str, data: Any, topic: Optional[str] = None):
        """Broadcast payload to all subscribed connected clients serialized non-blockingly."""
        if not self.active_connections:
            return

        payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
        target_topic = topic or event_type

        # Snapshot current connections
        async with self._lock:
            targets = [
                ws for ws in self.active_connections
                if "*" in self.subscriptions.get(ws, set()) or target_topic in self.subscriptions.get(ws, set())
            ]

        if not targets:
            return

        # 各客戶端並行送出：一個卡住的連線最多拖 SEND_TIMEOUT_S，且不影響其他連線
        results = await asyncio.gather(*(self._send(ws, payload) for ws in targets), return_exceptions=True)
        dead_connections = [ws for ws, r in zip(targets, results) if isinstance(r, BaseException)]
        if dead_connections:
            reasons = sorted({f"send timeout {SEND_TIMEOUT_S:g}s" if isinstance(r, asyncio.TimeoutError) else type(r).__name__
                              for r in results if isinstance(r, BaseException)})
            await self._drop(dead_connections, ", ".join(reasons))


# Global singleton instance
ws_manager = ConnectionManager()
