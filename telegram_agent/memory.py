# f:\12_prj_raspi5\telegram_agent\memory.py
"""
Three-Tier Memory Hierarchy for Raspberry Pi 5 AI Agent:
- Tier 1: In-Memory Working Context Buffer (Recent 8 dialogue turns)
- Tier 2: User Profile & Fact Key-Value Store (Persistent SQLite)
- Tier 3: Long-Term Episodic Vector Memory (SQLite BLOB + nomic-embed-text + Cosine Similarity)
"""

import os
import json
import sqlite3
import struct
from datetime import datetime
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import requests

from config import (
    MEMORY_DB_PATH,
    OLLAMA_API_URL,
    DEFAULT_EMBEDDING_MODEL,
    SHORT_TERM_MEMORY_MAX_ROUNDS,
    VECTOR_TOP_K,
    VECTOR_SIMILARITY_THRESHOLD
)


class ThreeTierMemoryManager:
    """
    Manages three tiers of memory: Working Buffer, User Profile, and Vector Memory.
    Designed for zero extra memory-leak and fast retrieval on ARM Cortex-A76.
    """

    def __init__(self, db_path: str = MEMORY_DB_PATH):
        self.db_path = db_path
        # Tier 1: In-memory working buffers keyed by chat_id
        # stores list of {"role": "user"|"assistant", "content": "..."}
        self.working_memories: Dict[int, deque] = {}
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize SQLite tables for Tier 2 and Tier 3 storage."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Tier 2: User profile facts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_facts (
                    chat_id INTEGER,
                    fact_key TEXT,
                    fact_value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chat_id, fact_key)
                )
            """)
            # Tier 3: Long-term episodic vector memory
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # =========================================================================
    # Tier 1: Working Memory Buffer (即時短期對話記憶)
    # =========================================================================

    def get_working_memory(self, chat_id: int) -> List[Dict[str, str]]:
        """Retrieve recent dialogue turns for the given chat_id."""
        if chat_id not in self.working_memories:
            self.working_memories[chat_id] = deque(maxlen=SHORT_TERM_MEMORY_MAX_ROUNDS * 2)
        return list(self.working_memories[chat_id])

    def append_turn(self, chat_id: int, user_text: str, assistant_text: str):
        """Append a user/assistant dialogue turn to working memory."""
        if chat_id not in self.working_memories:
            self.working_memories[chat_id] = deque(maxlen=SHORT_TERM_MEMORY_MAX_ROUNDS * 2)
        self.working_memories[chat_id].append({"role": "user", "content": user_text})
        self.working_memories[chat_id].append({"role": "assistant", "content": assistant_text})

    def clear_working_memory(self, chat_id: int):
        """Reset the working memory buffer."""
        if chat_id in self.working_memories:
            self.working_memories[chat_id].clear()

    # =========================================================================
    # Tier 2: User Profile Facts (使用者事實特徵 KV)
    # =========================================================================

    def set_user_fact(self, chat_id: int, key: str, value: str):
        """Store or update a user profile fact."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_facts (chat_id, fact_key, fact_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(chat_id, fact_key) DO UPDATE SET
                    fact_value = excluded.fact_value,
                    updated_at = CURRENT_TIMESTAMP
            """, (chat_id, key.strip(), value.strip()))
            conn.commit()

    def get_user_facts(self, chat_id: int) -> Dict[str, str]:
        """Retrieve all profile facts for the given user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fact_key, fact_value FROM user_facts WHERE chat_id = ?", (chat_id,))
            rows = cursor.fetchall()
            return {row["fact_key"]: row["fact_value"] for row in rows}

    def delete_user_fact(self, chat_id: int, key: str):
        """Delete a specific user profile fact."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_facts WHERE chat_id = ? AND fact_key = ?", (chat_id, key.strip()))
            conn.commit()

    def format_profile_prompt(self, chat_id: int) -> str:
        """Format user facts into a clean system instruction block."""
        facts = self.get_user_facts(chat_id)
        if not facts:
            return ""
        lines = ["\n[User Profile & Persona Facts (永久個人化記憶)]:"]
        for k, v in facts.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines) + "\n"

    # =========================================================================
    # Tier 3: Long-term Episodic Vector Memory (語意長期向量記憶庫)
    # =========================================================================

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute 768-dim float32 vector embedding via local Ollama nomic-embed-text."""
        try:
            url = f"{OLLAMA_API_URL}/api/embeddings"
            payload = {
                "model": DEFAULT_EMBEDDING_MODEL,
                "prompt": text
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                vector = resp.json().get("embedding", [])
                if vector:
                    return np.array(vector, dtype=np.float32)
        except Exception as e:
            print(f"[Memory] Failed to compute embedding: {e}")
        return None

    def store_memory(self, chat_id: int, content: str) -> bool:
        """Store a semantic text snippet into Tier 3 vector memory."""
        if not content.strip():
            return False
        vec = self.get_embedding(content)
        if vec is None:
            return False
        
        # Serialize float32 numpy array to binary blob
        vec_bytes = vec.tobytes()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO episodic_memories (chat_id, content, embedding)
                VALUES (?, ?, ?)
            """, (chat_id, content.strip(), vec_bytes))
            conn.commit()
        return True

    def retrieve_memories(self, chat_id: int, query: str, top_k: int = VECTOR_TOP_K, threshold: float = VECTOR_SIMILARITY_THRESHOLD) -> List[Tuple[str, float]]:
        """
        Perform fast Vector Cosine Similarity Search over user's episodic memories.
        Returns list of (content, similarity_score).
        """
        query_vec = self.get_embedding(query)
        if query_vec is None:
            return []

        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec_norm = query_vec / query_norm

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, embedding FROM episodic_memories WHERE chat_id = ?", (chat_id,))
            rows = cursor.fetchall()
            if not rows:
                return []

            scored_memories = []
            for row in rows:
                blob = row["embedding"]
                doc_vec = np.frombuffer(blob, dtype=np.float32)
                doc_norm = np.linalg.norm(doc_vec)
                if doc_norm > 0:
                    similarity = float(np.dot(query_vec_norm, doc_vec / doc_norm))
                    if similarity >= threshold:
                        scored_memories.append((row["content"], similarity))

            # Sort descending by cosine similarity
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            return scored_memories[:top_k]

    def format_retrieved_memories_prompt(self, chat_id: int, query: str) -> str:
        """Retrieve and format semantic memories for injection into the system prompt."""
        memories = self.retrieve_memories(chat_id, query)
        if not memories:
            return ""
        lines = ["\n[Retrieved Long-Term Semantic Memories (長期歷史記憶庫檢索)]:"]
        for content, score in memories:
            lines.append(f"- (相關度 {score:.2f}): {content}")
        return "\n".join(lines) + "\n"

    # =========================================================================
    # Complete Prompt Assembly (三層記憶融合組裝)
    # =========================================================================

    def build_prompt_messages(self, chat_id: int, user_query: str, base_system_instruction: str = "") -> List[Dict[str, str]]:
        """
        Assemble the final message list for Ollama Chat API:
        1. System Prompt + User Facts (Tier 2) + Retrieved Memories (Tier 3)
        2. Dialogue History from Working Memory (Tier 1)
        3. Current User Query
        """
        system_parts = []
        if base_system_instruction:
            system_parts.append(base_system_instruction.strip())

        # Inject Tier 2: User profile facts
        profile_block = self.format_profile_prompt(chat_id)
        if profile_block:
            system_parts.append(profile_block)

        # Inject Tier 3: Retrieved semantic memories
        memory_block = self.format_retrieved_memories_prompt(chat_id, user_query)
        if memory_block:
            system_parts.append(memory_block)

        final_system_prompt = "\n\n".join(system_parts)

        messages = [
            {"role": "system", "content": final_system_prompt}
        ]

        # Inject Tier 1: Working buffer turns
        working_turns = self.get_working_memory(chat_id)
        messages.extend(working_turns)

        # Current turn
        messages.append({"role": "user", "content": user_query})

        return messages
