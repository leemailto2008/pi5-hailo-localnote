# 🎙️ SPEC: Hailo-10H NPU Whisper 語音轉錄引擎部署與架構整合規格書

- **專案名稱 (Project):** Pi 5 Local Note - Hailo-10H Whisper NPU Acceleration
- **模組路徑 (Module Path):** `f:\13_prj_raspi5_localnote\hailo_whisper\`
- **建立日期 (Created Date):** 2026-10-19
- **作者 (Author):** Senior Staff Software Engineer & System Architect

---

## 1. 系統目標與工程效益 (Engineering Objectives)

1. **CPU 負載徹底歸零 (Zero-CPU Contention):**
   - 原架構：樹莓派 5 四核心 CPU 在執行 `faster-whisper (small)` 時佔用率達 360%~380%，造成系統瞬態溫度飆升，並干擾藍牙 UART SCO 封包傳輸。
   - 目標架構：將 Whisper Encoder/Decoder 完整下放至 Hailo-10H NPU (40 TOPS INT4) 與板載 8GB LPDDR4X 記憶體執行，CPU 使用率降至 **< 5%**（僅負責 Log-Mel DMA 搬移與 Token 串流）。
2. **極致即時率 (Real-Time Factor, RTF):**
   - `Whisper-Small (244M)` 推論 RTF 目標由現行 CPU 的 `0.65`（推論時間約音訊的 65%）縮短至 **`0.05 ~ 0.08`**（推論 6 秒切片僅需 0.3 ~ 0.5 秒）。
3. **無縫退避機制 (Graceful Fallback & High Availability):**
   - 提供統一介面 `WhisperEngineInterface`，若檢測到 Hailo-10H 裝置未載入或 HEF 異常，自動無感退避至 CPU `faster-whisper`，保證系統高可用性。

---

## 2. 硬體與軟體架構層次 (Architecture Hierarchy)

```
+-----------------------------------------------------------------------------+
|                           FastAPI / Web UI Layer                            |
|             (main.py / meeting_minutes.py / voice_pipeline.py)              |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                       hailo_whisper 整合介面層                             |
|               (Unified Engine: HailoWhisperEngine / Fallback)               |
+-----------------------------------------------------------------------------+
        |                                                             |
        v (Hailo-10H Available)                                       v (Fallback)
+-----------------------------------------------+             +---------------+
|          Hailo-10H NPU 推論核心               |             | CPU 推論核心  |
|  1. 音訊前處理: Log-Mel 80-band (NumPy / SciPy)|             | faster-whisper|
|  2. DMA 傳輸: PCIe Gen3 x1 (8.0 Gbps)         |             | CTranslate2   |
|  3. Encoder: 80x3000 -> 1500x768 (HEF)        |             +---------------+
|  4. Decoder: Auto-regressive Loop (8GB LPDDR4X)|
|  5. BPE 解碼與 OpenCC 繁體對齊                |
+-----------------------------------------------+
```

---

## 3. 資料管線與規格 (Data Pipeline Specification)

### 3.1 音訊輸入規範 (Audio Input)
- **取樣率 (Sample Rate):** 16,000 Hz, 16-bit Mono PCM (相容目前 `/dev/shm/audio_cache/` 產出)。
- **切片規格 (Chunk Window):** 6.0 秒固定窗口 (動態 Zero-Padding 至 30.0 秒或 Hailo 支援之可變長度)。
- **Log-Mel 頻譜特徵:**
  - FFT Window: 400, Hop Size: 160 (10ms 步進)。
  - Mel Filter Banks: 80 bands, 範圍 0 ~ 8,000 Hz。
  - 特徵尺寸: `(80, 3000)` FP16 / INT8。

### 3.2 關鍵模組劃分 (Module Breakdown)

1. `hailo_whisper/audio_processor.py`:
   - 輕量化純 Python/NumPy Log-Mel 計算器，杜絕 PyTorch/Librosa 等重型相依庫（啟動延遲 < 10ms）。
2. `hailo_whisper/tokenizer.py`:
   - Whisper 專屬 Tiktoken BPE 解碼器與特殊 Token (`<|startoftranscript|>`, `<|notimestamps|>`) 處理。
   - 整合 `opencc-python-reimplemented` 保證繁體中文輸出。
3. `hailo_whisper/engine.py`:
   - `HailoWhisperEngine` 實作：
     - 使用 `hailort.pyhailort` 載入 `.hef` 模型檔案。
     - 配置 VDevice (Virtual Device) 與 ConfiguredNetwork。
     - 封裝非同步推論 (Asynchronous Inference) 與自動排空 (Drain Queue)。
4. `hailo_whisper/benchmark.py`:
   - 壓測工具：測量首字延遲 (TTFT)、單塊 RTF、記憶體占用與 PCIe 頻寬利用率。

---

## 4. 階段實施里程碑 (Phased Milestones)

| 階段 (Phase) | 項目名稱 | 輸出產物 | 預估工時 |
|:---|:---|:---|:---:|
| **Phase 1** | 架構規格與介面定義 | `SPEC_HAILO10_WHISPER.md`, `engine.py` 介面 | 0.5 天 |
| **Phase 2** | 純 NumPy 輕量級 Log-Mel 前處理器 | `audio_processor.py`, 單元測試 | 0.5 天 |
| **Phase 3** | BPE Tokenizer 與繁體中文轉譯器 | `tokenizer.py`, 驗證指令碼 | 0.5 天 |
| **Phase 4** | HailoRT-GenAI / HEF 載入與推論管線 | `engine.py` 核心, HEF 載入測試 | 1.0 天 |
| **Phase 5** | 系統管線整併與 E2E 雙模式退避切換 | `voice_pipeline.py` 整合, 實測驗證 | 0.5 天 |
