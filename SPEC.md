# 規格文件 (SPEC.md): Task 05 - 記憶體環形緩衝區與零磁碟 I/O (In-Memory Ring Buffer & Zero-Disk I/O)

> **版本 (Version):** 1.0.0  
> **模組路徑 (Module Path):** `code_dispatcher/voice_memory.py`, `code_dispatcher/voice_pipeline.py`, `code_dispatcher/voice_streaming.py`, `code_dispatcher/main.py`  
> **建立日期 (Date):** 2026-10-08  
> **架構師 (Architect):** Senior Staff Software Engineer  

---

## 1. 背景與現狀問題 (Background & Problem Statement)

目前系統錄音與語音管線在處理音訊時，大量依賴實體快閃儲存 (MicroSD / eMMC / NVMe)：
1. **實體磁碟過度寫入與耗損 (Flash Storage Wear)：**
   - 每次錄音生成 `master.ogg`、ffmpeg 轉出 `master.mp3`、切出 `chunk_000.ogg`，全寫入 `/home/pi/code_dispatcher/audio_cache`。
   - `voice_pipeline.py` 在執行 STT 前，又透過 ffmpeg 將音訊轉存為暫存實體檔案 `temp_whisper_xxx.wav`，辨識完再手動刪除。
   - 頻繁的建立與刪除造成快閃記憶體區塊磨損 (Write Amplification)，長期運行極易引發檔案系統壞軌。
2. **I/O 延遲開銷 (I/O Latency Overhead)：**
   - 磁碟寫入與同步 (fsync) 在 MicroSD 上產生 200~400ms 的無謂等待。

---

## 2. 目標與驗收標準 (Objectives & Acceptance Criteria)

### 2.1 核心目標
1. **純記憶體 RAM-Disk 快取 (`/dev/shm/audio_cache`)：**
   - 將音訊暫存目錄掛載/重定向至 Linux 原生 POSIX 共享記憶體 `tmpfs` (`/dev/shm/audio_cache`)。
   - 所有即時音訊檔讀寫 100% 在 RAM 中完成，**實體磁碟寫入量降為 0**。
2. **純記憶體音訊串流解碼 (Zero-Disk In-Memory PCM Decoding)：**
   - 徹底移除 `temp_whisper_xxx.wav` 實體暫存檔機制。
   - 透過 ffmpeg 管道 (`pipe:1`) 直接將音訊解碼為 16kHz s16le PCM 位元組流至記憶體中，轉為 `np.float32` 陣列直接餵入 `WhisperModel.transcribe()`。
3. **容量感知環形緩衝區 (Capacity-Aware Ring Buffer)：**
   - 實作 FIFO 環形配額管理 (`AudioMemoryRingBuffer`)，上限設為 64MB 或最近 15 筆音訊。超過閥值時自動淘汰最舊音訊，嚴防 RAM 洩漏。
4. **驗收標準：**
   - 語音轉錄流程全線無任何暫存 `.wav` 檔案落地。
   - 音訊轉換階段延遲縮減 >150ms。

---

## 3. 系統架構設計 (System Architecture)

```mermaid
graph TD
    subgraph 🎙️ 語音收音與切片 (In-Memory POSIX RAM-Disk)
        MIC["藍牙耳麥輸入"] --> FFMPEG["ffmpeg (pulse)"]
        FFMPEG --> RAMFS["/dev/shm/audio_cache (tmpfs in RAM)"]
        RAMFS --> M_OGG["master.ogg (RAM)"]
        RAMFS --> CHUNKS["chunk_*.ogg (RAM)"]
        RAMFS --> RING["AudioMemoryRingBuffer (Max 64MB FIFO Eviction)"]
    end

    subgraph ⚡ 零磁碟 STT 串流解碼 (Zero-Disk In-Memory PCM)
        CHUNKS --> PIPE["ffmpeg pipe:1 (stdout)"]
        PIPE --> PCM["raw s16le PCM bytes (In-Memory)"]
        PCM --> NUMPY["np.float32 (16kHz Mono Array)"]
        NUMPY --> WHISPER["faster-whisper (Direct Array Inference)"]
    end

    subgraph 🌐 靜態音訊回放 (Web Player Serving)
        M_OGG --> MP3["master.mp3 (RAM)"]
        MP3 --> FASTAPI["FastAPI /audio mount (/dev/shm/audio_cache)"]
        FASTAPI --> BROWSER["📱 瀏覽器直接試聽 (極速微秒級加載)"]
    end
```

---

## 4. 詳細技術規格 (Technical Specifications)

### 4.1 Linux POSIX 共享記憶體架構
- 基礎路徑：優先採用 `/dev/shm/audio_cache`。若在非 Linux 系統環境則平滑回退至本機快取目錄。
- 權限管理：`0755`，確保 FastAPI (Uvicorn) 具備完整讀寫與子目錄建立權限。

### 4.2 零磁碟 PCM 轉換演算法 (In-Memory PCM Streaming)
```python
cmd = [
    "ffmpeg", "-y", "-i", input_audio,
    "-f", "s16le", "-ac", "1", "-ar", "16000",
    "-"
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
raw_pcm, _ = proc.communicate()
audio_np = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
# 直接傳入 WhisperModel.transcribe(audio_np)
```

### 4.3 環形緩衝區淘汰策略 (Ring Buffer FIFO Eviction)
- 監控總快取目錄容量。
- 當容量超過 `MAX_RAMFS_BYTES = 64 * 1024 * 1024` (64MB) 或目錄數超過 `MAX_SESSIONS = 15` 時，依最後修改時間 (mtime) 刪除最舊的 session 目錄。

---

## 5. 實作檢查清單 (Implementation Checklist)

- [ ] **Phase 1: 記憶體環形緩衝區模組實裝 (`code_dispatcher/voice_memory.py`)**
  - 實作 `init_memory_cache()`、`evict_ring_buffer()`。
  - 封裝 `audio_to_pcm_array()` 零磁碟記憶體轉換函式。
- [ ] **Phase 2: 語音管線零磁碟改造 (`code_dispatcher/voice_pipeline.py`)**
  - 改造 `speech_to_text()`：徹底移除實體 `wav_path` 寫入/刪除邏輯，全面接入 `audio_to_pcm_array()`。
- [ ] **Phase 3: 串流與 API 目錄重定向至 RAMFS (`code_dispatcher/main.py`, `voice_streaming.py`)**
  - 將 `AUDIO_CACHE_DIR` 導向 `/dev/shm/audio_cache`。
  - 在每次 session 完成時觸發環形緩衝區配額維護。
- [ ] **Phase 4: 本機實測與部署驗證**
  - 檢查 `/dev/shm/audio_cache` 運作與 `lsof` / `df -h /dev/shm` 佔用。
  - 驗證無任何檔案寫入 MicroSD 實體磁區。
