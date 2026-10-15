# 規格文件 (SPEC.md) - 樹莓派 5 邊緣雙工錄音與 AI NPU 邊錄邊辨架構
# Concurrent Recording & Streaming STT Pipeline via AI NPU

> **文件狀態 (Status):** 實作中 (In Implementation)  
> **架構師 (Architect):** Antigravity (Google Senior Staff Engineer)  
> **模組路徑 (Module Path):** `AI_HAT_plus/` & `code_dispatcher/`  
> **目標硬體 (Target Hardware):** Raspberry Pi 5 (16GB RAM) + Official Raspberry Pi AI HAT+ (Hailo-8 M.2 26 TOPS)  

---

## 1. 背景與需求動機 (Background & Motivation)

### 1.1 現狀瓶頸
在現行系統中，系統採用「**全軌錄音完成後，才啟動全軌 STT (Post-Recording Full-Track STT)**」模式：
* 使用者錄音 60 秒，必須在錄音結束後等待約 25~45 秒才能看到整篇逐字稿，接著再等待約 30 秒的 LLM 摘要，形成嚴重的階段式等待感。

### 1.2 核心目標
1. **體感無延遲 (Perceptual Zero-Latency):** 錄音開始後 **5~6 秒**，前端逐字稿卡片立即秒開，並即時動態滾動跳出第一句辨識出的繁體中文！
2. **錄完即完工 (Instant Finish on Stop):** 使用者按下「停止錄音」時，90% 的音訊早己在背景轉錄完畢，尾端剩餘時間 < 1 秒即可完成全篇整合。
3. **藍牙物理層零爭搶 (Zero-Contention Bluetooth SCO):**
   * 樹莓派 5 藍牙 mSBC (16kHz) 走實體硬體 UART 總線 (`/dev/ttyAMA0`)，每 7.5ms 一幀。
   * 必須透過 CPU 核心綁定 (`taskset`) 隔離藍牙音訊守護行程與 NPU 推論調度，確保 100% 不丟幀、不破音。

---

## 2. 系統架構設計 (System Architecture)

```
                       [藍牙耳麥 SCO 16kHz]
                                │ (每 7.5ms 一幀)
                                ▼
         ┌─────────────────────────────────────────────┐
         │          WirePlumber / PipeWire             │ (鎖定 CPU 0,1 / Priority -18)
         └──────────────────────┬──────────────────────┘
                                │
                                ▼
         ┌─────────────────────────────────────────────┐
         │      單一持續 ffmpeg 麥克風收音守護行程       │
         │ (aresample=16000, highpass=80, lowpass=7600)│
         └──────────────┬──────────────────────────────┘
                        │
         ┌──────────────┴──────────────────────────────┐
         ▼                                             ▼
[母帶寫入記憶體磁碟]                             [50ms PCM 標準輸出 (pipe:1)]
/dev/shm/audio_cache/m_xxx.ogg                         │
(保證完整母帶供試聽與下載)                               ▼
                                         [即時 VU 與 5s 滑動分塊隊列]
                                         (PCM Chunk Queue: 5s / 160KB)
                                                       │ (非同步派發)
                                                       ▼
                                         ┌───────────────────────────┐
                                         │  AI NPU 聽寫工作線程池    │
                                         │ (鎖定 CPU 2,3 / PCIe Gen3)│
                                         └─────────────┬─────────────┘
                                                       │
                                                       ▼
                                         [WebSocket: transcript_stream]
                                         (前端第 6 秒起逐段動態噴字)
```

---

## 3. 關鍵實作規格 (Technical Specifications)

### 3.1 PCM 記憶體滑動切片佇列 (Sliding PCM Buffer)
* 不切斷現有 ffmpeg 錄音進程，避免頻繁重啟 ffmpeg 造成的藍牙 UART 總線抖動與銜接爆音。
* 從 ffmpeg `stdout` 的 16kHz mono PCM 串流中，每累積 5.0 秒（80,000 個 16-bit 採樣點，約 160,000 位元組）自動切出一個 `AudioSegmentChunk`。
* 透過靜音檢測器 (Voice Activity Detector / RMS Peak Check) 過濾純空白或呼吸聲片段。

### 3.2 CPU 核心隔離策略 (CPU Core Pinning & Affinity)
* **核心 0、1 (專屬音訊):**
  - WirePlumber、PipeWire、BlueZ、ffmpeg 行程設定 CPU 親和度 `0,1`，優先級 `renice -n -18`。
* **核心 2、3 (推論與應用服務):**
  - FastAPI 主進程、HailoRT 驅動、STT 背景線程設定 CPU 親和度 `2,3`，優先級 `0`。

### 3.3 容錯與尾端平滑縫合 (Tail Stitching & Context Merging)
1. 錄音期間每個 chunk 獨立由 NPU/Whisper 轉錄，取得即時預覽文本。
2. 當使用者手動點擊「停止」或時長到達時：
   - 立即清空 queue，將最後一段尾音（例如 1.2 秒）進行極速辨識。
   - 使用雙換行 (`\n\n`) 自然段落自動串接，並以 OpenCC 統一轉為台灣繁體中文標準。
3. 觸發 `transcript_ready` 事件，無縫接軌進入 LLM 會議紀要打字機階段。

---

## 4. 驗收標準 (Acceptance Criteria)

1. **首字延遲 (Time-To-First-Segment, TTFS):** 錄音開始後 **6.5 秒以內**，前端逐字稿即時展示第 1 區塊轉錄文本。
2. **錄音結束等待時間 (Tail Latency):** 錄音停止後，完整逐字稿就緒時間 **< 1.5 秒**。
3. **音訊穩定度 (Zero Contention):** 錄音全程日誌不得出現任何 `spa.bluez5.source.sco: decode failed: -3`。
4. **母帶完整性:** `/dev/shm/audio_cache/` 依然產出高品質完整的 OGG 母帶供波形圖拖曳與試聽。
