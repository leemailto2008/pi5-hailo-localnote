# 規格文件 (SPEC.md): Task 33 - 30分鐘長時會議錄音與階層式會議紀要引擎
# (30-Minute Long Recording & Hierarchical Map-Reduce Minutes Engine)

> **版本 (Version):** 1.0.0  
> **模組路徑 (Module Path):** `code_dispatcher/meeting_minutes.py`, `code_dispatcher/main.py`, `code_dispatcher/voice_pipeline.py`, `code_dispatcher/ui_client.html`  
> **建立日期 (Date):** 2026-10-19  
> **架構師 (Architect):** Senior Staff Software Engineer & System Architect  

---

## 1. 背景與硬體邊界評估 (Background & Hardware Feasibility)

### 1.1 記憶體與磁碟邊界 (Memory & Disk Footprint)
- 錄音格式為 16kHz 16-bit Mono PCM：
  - 每秒數據量: $16,000 \times 2 = 32,000\text{ Bytes} \approx 31.25\text{ KB/s}$
  - 30 分鐘（1,800 秒）未壓縮 PCM: $32\text{ KB} \times 1,800 \approx 57.6\text{ MB}$
  - 轉為 Float32 音訊矩陣: $115.2\text{ MB}$
  - Opus/OGG 壓縮母帶 ($64\text{ kbps}$): 約 $14.4\text{ MB}$
- 樹莓派 5 擁有 8GB LPDDR4X 實體記憶體，目前可用 RAM 約 $4,770\text{ MB}$。
- 本專案音訊全數掛載於 Linux POSIX 記憶體磁碟 (`/dev/shm/audio_cache/`)，30 分鐘僅耗費 RAM 的 **1.2% ~ 2.4%**，完全達成 **零磁碟寫入 (Zero-Disk I/O)** 原則，硬體完全可行。

### 1.2 串流 STT 吞吐平衡 (Streaming STT & Concurrency)
- 採用每 8 秒切片即時解算（`_chunk_stt_worker`），配合 VAD 靜音快篩（Peak < 0.035 & RMS < 0.0025 直接跳過）。
- 錄音 30 分鐘期間，絕大部分語音段落已在背後完成轉譯。30 分鐘錄音結束時，尾端等待時間小於 2 秒。

---

## 2. 核心功能規格 (Core Functional Specifications)

### 2.1 提前結束收音 (Graceful Early Finish)
- **痛點:** 使用者若設定 30 分鐘，但在第 12 分鐘會議結束，現有系統僅有「取消 (Cancel)」按鈕（會拋棄音訊與結果）。
- **架構方案:**
  - 新增非同步事件 `finish_recording_event: asyncio.Event`。
  - 新增 API 端點: `POST /api/record/finish`。
  - 當收到提早結束請求時，向 ffmpeg 行程發送 `SIGTERM`（在 Linux 下使 ffmpeg 正常關閉檔案寫入並 flush container header），保留前面錄製的所有音訊，自動計算實際錄音時長，並立即轉入轉譯母帶與會議總結。

### 2.2 階層式 Map-Reduce 會議紀要提煉 (Hierarchical Summary for Long Transcripts)
- **痛點:** 30 分鐘中文語音逐字稿通常達 3,000 ~ 6,000 字。Ollama 邊緣小模型（如 `qwen2.5:3b-opt`）之 Context Window 預設為 2,048 ~ 4,096 tokens。直接塞入可能引發 Context 截斷或遺忘前段討論。
- **架構方案:**
  - **閾值判定:** 當逐字稿字數 $\le 2,500$ 字時，走單次提煉 (Fast-Path Direct Summary)；
  - **當字數 $> 2,500$ 字時，自動切換至 Map-Reduce 階層提煉:**
    1. **Map (段落分塊提煉):** 依據自然段落或約 1,200 ~ 1,500 字進行切塊，調用模型快速萃取各段「核心討論點與行動要項」；
    2. **Reduce (全局彙總綜合):** 將各區塊要點聚合成單一上下文，生成標準三段式會議紀要（`## 摘要`、`## 待辦 (表格)`、`## 決議`）。

### 2.3 API 與資料模型更新 (API & Data Schema)
- `DemoRecordRequest.seconds`: 上限從 `300` 放寬至 `1800` (`ge=10, le=1800`)。
- 新增 `POST /api/record/finish`: 觸發提前結算。
- WebSocket / Status 更新: 錄音時長 `total` 動態調整為實際錄音秒數。

### 2.4 前端 UI 增強 (Frontend Enhancements)
- `<select id="dur">`: 加入 `300` (5分鐘)、`600` (10分鐘)、`900` (15分鐘)、`1800` (30分鐘-完整會議)。
- 錄音進行中動態顯示 `⏹️ 說完了，立即整理` 按鈕。

---

## 3. 實作步驟 (Implementation Steps)

1. **`code_dispatcher/voice_pipeline.py`:**
   - 擴充 `record_audio_from_mic()` 參數，新增 `finish_early_event: Optional[asyncio.Event] = None`。
   - 監聽 `finish_early_event`，觸發時執行 `proc.terminate()` 並等待結束，回傳錄音成功 `True`。
2. **`code_dispatcher/meeting_minutes.py`:**
   - 類別 `MeetingJobManager` 新增 `self._finish_early_event: asyncio.Event` 與 `finish_recording_early()` 方法。
   - `_generate_summary_llm()` 加入長文本檢查，字數 $> 2500$ 字時自動分塊執行 Map-Reduce。
3. **`code_dispatcher/main.py`:**
   - 修改 `DemoRecordRequest.seconds` 上限為 `1800`。
   - 新增 `@app.post("/api/record/finish")` 路由。
4. **`code_dispatcher/ui_client.html` & `static/index.html`:**
   - 更新選單與按鈕，綁定 `finishRecordingEarly()`。
5. **編譯測試與遠端推送 (Verification & Deployment):**
   - 執行本地語法驗證 `py_compile`。
   - 同步至樹莓派重啟服務。
   - 推送至 GitLab 與 GitHub 雙遠端。
