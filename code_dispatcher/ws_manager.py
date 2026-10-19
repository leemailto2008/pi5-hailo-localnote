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

    async def send_personal(self, websocket: WebSocket, event_type: str, data: Any):
        lock = self._send_locks.get(websocket)
        try:
            payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
            if lock:
                async with lock:
                    await websocket.send_text(payload)
            else:
                await websocket.send_text(payload)
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

        dead_connections = []
        for ws in targets:
            lock = self._send_locks.get(ws)
            try:
                if lock:
                    async with lock:
                        await ws.send_text(payload)
                else:
                    await ws.send_text(payload)
            except Exception:
                dead_connections.append(ws)

        if dead_connections:
            async with self._lock:
                for ws in dead_connections:
                    self.active_connections.discard(ws)
                    self._send_locks.pop(ws, None)
                    self.subscriptions.pop(ws, None)


# Global singleton instance
ws_manager = ConnectionManager()
