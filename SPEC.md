# 規格文件 (SPEC.md): Task 32 - 雙層環形緩衝區與核心隔離分塊音訊管線
# (Two-Layer Ring Buffer & Core-Isolated Chunking Voice Pipeline)

> **版本 (Version):** 1.2.0  
> **模組路徑 (Module Path):** `code_dispatcher/two_layer_audio.py`, `code_dispatcher/voice_pipeline.py`, `code_dispatcher/meeting_minutes.py`  
> **建立日期 (Date):** 2026-10-13  
> **架構師 (Architect):** Senior Staff Software Engineer & System Architect  

---

## 1. 背景與硬體物理邊界 (Background & Hardware Constraints)

樹莓派 5 (BCM2712 SoC) 的藍牙晶片透過實體 UART 串列總線 (`/dev/ttyAMA0`) 通訊，其硬體接收暫存器 (Rx FIFO) 僅有 **32 個位元組**。藍牙語音 (mSBC 16kHz) 走 SCO 通道，每 7.5ms 即送入 60 Bytes。
若背景執行 CPU 全核心張量乘法 (GEMM)，記憶體匯流排與快取爭搶將導致中斷延遲超過 3ms，觸發硬體 FIFO 溢位 (`decode failed: -3`)。

為實現「邊錄邊切塊即時辨識」且「100% 不破壞硬體音訊流」，必須透過**雙層緩衝區 (Two-Layer Buffer)** 與 **CPU 核心硬隔離 (Core Affinity)** 徹底解耦。

---

## 2. 系統架構設計 (System Architecture)

```
[Bluetooth Mic (SCO 16kHz)]
         │ (7.5ms frame)
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: 硬體對接環形緩衝區 (Real-Time Hardware Ring Buffer)    │
│  - 綁定 CPU 0 (taskset -c 0)                                │
│  - 實時排程 SCHED_FIFO 優先級 80 (chrt -f 80)                 │
│  - 寫入 Linux POSIX 記憶體磁碟 (/dev/shm/audio_cache)         │
│  - 輸出: master.ogg (全軌母帶) + 環形 PCM Ring Buffer 佇列     │
└─────────────────────────────────────────────────────────────┘
         │
         │ (每 10 秒或 VAD 靜音點切塊抽取)
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: 分段分塊推論器 (Throttled Chunk Inference Engine)    │
│  - 綁定 CPU 2,3 (taskset -c 2,3)                            │
│  - 算力油門限制: cpu_threads=2 (保留 50% CPU 與匯流排餘裕)       │
│  - 模型: faster-whisper (ARM NEON int8)                      │
│  - 輸出: 逐塊即時轉錄文字，WebSocket 串流廣播至前端             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: 零延遲波形圖預提取 (Pre-computed Peaks Visualizer)   │
│  - 後端快速萃取 200 點峰值 JSON (Peaks Array, ~2KB)           │
│  - 前端 wavesurfer.load(url, peaks) 0.01 秒免解碼瞬間繪製      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 實作規範 (Implementation Specifications)

1. **`two_layer_audio.py` (核心模組)：**
   - `TwoLayerAudioPipe` 類別：
     - 管理 Layer 1 ffmpeg 捕捉行程，綁定 CPU 0 與 Linux RT 排程。
     - 以 16000Hz 16-bit Mono PCM 格式寫入記憶體環形緩衝區。
     - 定期萃取音訊塊 (Audio Chunks，預設 8~10 秒) 並送入非同步佇列。
     - 提供 `extract_waveform_peaks(ogg_path)` 函式，快速產生正規化浮點峰值陣列。
2. **`voice_pipeline.py` (推論調度調校)：**
   - 在多工作業時提供安全模式推論器：`cpu_threads=2`，並透過 `os.sched_setaffinity` 鎖定在 CPU 2, 3。
3. **`meeting_minutes.py` 與前端介面整合：**
   - 錄音進行中，每產生一個 chunk 即時推播轉錄文字。
   - 錄音結束時回傳 `peaks`，前端 Wavesurfer 0 延遲瞬開波形圖。

---

## 4. 驗收標準 (Acceptance Criteria)

- [ ] 錄音進行中分塊推論時，`dmesg` 與 `journalctl` 0 筆 `decode failed: -3` 錯誤。
- [ ] 3 分鐘長音訊能按 10 秒區間逐塊輸出即時文字。
- [ ] 錄音結束時波形圖於 0.05 秒內渲染完畢，無需等待 Web Audio 解碼。
- [ ] 程式碼符合 KISS 原則，嚴格處理例外狀況與 Process Lifecycle 釋放。
