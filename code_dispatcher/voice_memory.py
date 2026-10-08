# f:\12_prj_raspi5\code_dispatcher\voice_memory.py
"""
In-Memory Ring Buffer & Zero-Disk Audio I/O Engine for Raspberry Pi 5.
Directly harnesses Linux POSIX shared memory (/dev/shm/audio_cache) and
in-memory PCM streaming to eliminate MicroSD/NVMe flash wear and sub-second I/O latency.
"""

import os
import sys
import shutil
import subprocess
from typing import Dict, Any, Optional, Union, List

import numpy as np

# Maximum RAM quota allocated to audio sessions (Default: 64 MB)
MAX_RAM_CACHE_BYTES = 64 * 1024 * 1024
# Maximum number of concurrent audio sessions preserved in RAM ring buffer
MAX_RETAINED_SESSIONS = 15


def get_in_memory_cache_dir() -> str:
    """
    Get ultra-fast in-memory cache path.
    Prioritizes Linux POSIX shared memory (/dev/shm/audio_cache) for 0 disk wear.
    Gracefully falls back to local workspace cache on non-Linux environments.
    """
    if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
        ram_dir = "/dev/shm/audio_cache"
        os.makedirs(ram_dir, exist_ok=True)
        return ram_dir

    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cache")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def audio_to_pcm_array(input_source: Union[str, bytes]) -> Optional[np.ndarray]:
    """
    Zero-Disk In-Memory PCM Audio Decoder.
    Directly decodes any audio format (OGG, MP3, WAV, OPUS) into a 16kHz Mono Float32
    NumPy array using an ffmpeg stdout pipe, completely bypassing physical disk writes.
    """
    if isinstance(input_source, str):
        if not os.path.exists(input_source) or os.path.getsize(input_source) < 500:
            return None
        cmd = [
            "ffmpeg", "-y",
            "-i", input_source,
            "-f", "s16le",
            "-ac", "1",
            "-ar", "16000",
            "-"
        ]
        stdin_data = None
    elif isinstance(input_source, bytes):
        if len(input_source) < 500:
            return None
        cmd = [
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-f", "s16le",
            "-ac", "1",
            "-ar", "16000",
            "-"
        ]
        stdin_data = input_source
    else:
        return None

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        raw_pcm, _ = proc.communicate(input=stdin_data)

        if proc.returncode == 0 and raw_pcm and len(raw_pcm) > 0:
            # Convert 16-bit signed integer PCM to normalized float32 [-1.0, 1.0]
            audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_np
        return None
    except Exception as e:
        print(f"[VoiceMemory] Error decoding audio to in-memory PCM: {e}")
        return None


def evict_audio_ring_buffer(
    max_bytes: int = MAX_RAM_CACHE_BYTES,
    max_sessions: int = MAX_RETAINED_SESSIONS
) -> Dict[str, Any]:
    """
    Maintain FIFO Ring Buffer in RAM.
    Evicts oldest audio sessions or standalone files when RAM quota or count threshold is exceeded.
    """
    cache_dir = get_in_memory_cache_dir()
    if not os.path.exists(cache_dir):
        return {"evicted_count": 0, "reclaimed_bytes": 0}

    entries: List[Dict[str, Any]] = []
    total_size = 0

    try:
        for name in os.listdir(cache_dir):
            full_path = os.path.join(cache_dir, name)
            try:
                mtime = os.path.getmtime(full_path)
                if os.path.isdir(full_path):
                    # Directory size
                    dir_size = sum(
                        os.path.getsize(os.path.join(root, f))
                        for root, _, files in os.walk(full_path)
                        for f in files
                        if os.path.exists(os.path.join(root, f))
                    )
                    entries.append({"path": full_path, "is_dir": True, "size": dir_size, "mtime": mtime})
                    total_size += dir_size
                elif os.path.isfile(full_path):
                    file_size = os.path.getsize(full_path)
                    entries.append({"path": full_path, "is_dir": False, "size": file_size, "mtime": mtime})
                    total_size += file_size
            except Exception:
                pass
    except Exception as e:
        print(f"[VoiceMemory] Error reading ring buffer directory: {e}")
        return {"evicted_count": 0, "reclaimed_bytes": 0}

    # Sort oldest first (FIFO eviction)
    entries.sort(key=lambda x: x["mtime"])

    evicted_count = 0
    reclaimed_bytes = 0

    session_count = len(entries)
    for entry in entries:
        if total_size <= max_bytes and session_count <= max_sessions:
            break

        path_to_remove = entry["path"]
        try:
            if entry["is_dir"]:
                shutil.rmtree(path_to_remove)
            else:
                os.remove(path_to_remove)
            total_size -= entry["size"]
            session_count -= 1
            evicted_count += 1
            reclaimed_bytes += entry["size"]
            print(f"[VoiceMemory RingBuffer] Evicted: {os.path.basename(path_to_remove)} ({entry['size']} bytes)")
        except Exception as e:
            print(f"[VoiceMemory] Failed to evict {path_to_remove}: {e}")

    return {
        "evicted_count": evicted_count,
        "reclaimed_bytes": reclaimed_bytes,
        "remaining_bytes": total_size,
        "remaining_count": session_count,
        "is_ramfs": cache_dir.startswith("/dev/shm")
    }


def get_cache_stats() -> Dict[str, Any]:
    """Get active in-memory audio ring buffer stats."""
    cache_dir = get_in_memory_cache_dir()
    is_ramfs = cache_dir.startswith("/dev/shm")
    total_size = 0
    item_count = 0

    if os.path.exists(cache_dir):
        for root, _, files in os.walk(cache_dir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total_size += os.path.getsize(fp)
                    item_count += 1
                except Exception:
                    pass

    return {
        "cache_dir": cache_dir,
        "is_ramfs": is_ramfs,
        "storage_type": "POSIX Shared Memory (RAM-Disk, 0 Flash Wear)" if is_ramfs else "Physical Storage",
        "total_bytes": total_size,
        "total_mb": round(total_size / (1024 * 1024), 2),
        "item_count": item_count,
        "quota_mb": round(MAX_RAM_CACHE_BYTES / (1024 * 1024), 1)
    }
