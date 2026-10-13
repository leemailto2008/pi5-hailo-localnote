# 規格文件：Task 11 & Task 12 - 異步工作佇列、取消權杖與音訊波形視覺化剪輯
# (SPEC_TASK11_12.md)

> **版本 (Version):** 1.0.0  
> **系統環境:** Raspberry Pi 5 (8GB) | Debian 13 (Trixie) | FastAPI | faster-whisper | Wavesurfer.js  
> **涵蓋任務:**  
> - **Task 11: 異步工作佇列與取消權杖 (Job Queue & Cancellation Token)**  
> - **Task 12: 互動式音訊波形視覺化與剪輯 (Waveform Visualizer & Trimming - Wavesurfer.js)**  

---

## 一、 需求背景與架構目標 (Background & Objectives)

### 1.1 現存痛點
1. **推論無法即時中斷 (Unstoppable Inference):**  
   使用者若誤觸錄音或錄音過長想中斷時，舊推論（Whisper 4 核心運算或 Ollama 3B 模型）會無情霸佔樹莓派 5 全 CPU 資源數十秒甚至一分鐘，導致前端卡死無法開啟新錄音。
2. **錄音設備無法即時釋放 (Audio Device Lock):**  
   若錄音未結束時使用者離開或想重錄，ffmpeg 仍在佔用 PipeWire 藍牙 mSBC 音訊流，引起硬體爭搶。
3. **音訊試聽體驗陽春 (Basic Audio Player):**  
   錄音結束後僅有原生 `<audio controls>` 標籤，無法得知音訊能量波形分佈、哪一段有說話、哪一段是無效雜音。
4. **缺乏局部剪輯轉錄能力 (Lack of Region Trimming & Re-transcription):**  
   若一段 60 秒的語音只有第 10~30 秒是重點，使用者無法單獨圈選該段落進行二次高精度轉錄。

### 1.2 改良目標
1. **取消權杖機制 (Cancellation Token):**  
   實作後端 `cancellation_token`，支援錄音階段瞬間中斷 ffmpeg、Whisper 迭代階段中途跳出、LLM 階段終止串流，並提供 `/api/record/cancel` 端點。
2. **前端取消操作動效 (Cancel UX):**  
   在錄音與推論狀態下提供顯著的「⛔ 中斷任務」按鈕，WebSocket 即時同步中斷狀態，瞬間歸位為 IDLE。
3. **離線波形視覺化 (Offline Wavesurfer Visualizer):**  
   本機集成 `wavesurfer.min.js`（確保 PWA 100% 離線可用），在母帶生成後以霓虹漸層波形圖呈現，具備游標拖曳、時間軸刻度與播放頭。
4. **互動式區間剪輯與局部重錄 (Audio Trimming & Region STT):**  
   支援拖曳選取特定時間區間 (`start_time` ~ `end_time`)，提供「播放選取範圍」與「選取範圍轉錄 (`/api/transcribe/region`)」。

---

## 二、 核心架構與 API 介面設計 (Architecture & Interfaces)

### 2.1 取消權杖 (Cancellation Token Pattern)
- 在 `voice_pipeline.py` 與 `meeting_minutes.py` 傳入 `cancellation_token: threading.Event / asyncio.Event`。
- **錄音中斷:** `proc.terminate()` / `proc.kill()`，強制終止 ffmpeg 並釋放 PipeWire bluez_input 節點。
- **Whisper 中斷:** 在 `model.transcribe` 的 generator 迴圈中，每段 segment 迭代時檢驗 `if cancellation_token.is_set(): raise InterruptedError`。
- **LLM 中斷:** 取消 async generator 與 requests 連線。

### 2.2 後端 API 端點
- `POST /api/record/cancel`:
  - 觸發取消權杖，狀態重設為 `idle`，並向 WebSocket 廣播 `{ "state": "idle", "msg": "任務已取消" }`。
- `POST /api/transcribe/region`:
  - 參數: `{ "job_id": "...", "start_sec": float, "end_sec": float }`
  - 使用 ffmpeg 或 NumPy 切片直接提取特定秒數區間的 16kHz PCM，送入 Whisper 進行局部轉錄。

### 2.3 前端 Wavesurfer.js 與剪輯介面
- 檔案路徑: `code_dispatcher/static/vendor/wavesurfer.min.js`
- 樣式: 漸層紫/青霓虹色波形 (`#6366f1` -> `#06b6d4`)，Cursor 顏色 `#f43f5e`。
- 支援功能:
  - Play / Pause 波形播放。
  - 時間軸指示器 (`00:12 / 01:00`)。
  - 選區剪輯控制桿 (Start / End Trim Handles)。
  - 局部試聽按鈕與局部聽寫按鈕。

---

## 三、 實施計畫與檢查清單 (Implementation Plan)

- [ ] **步驟 1:** 下載/建立離線 `wavesurfer.min.js` 至 `code_dispatcher/static/vendor/`。
- [ ] **步驟 2:** 在 `voice_pipeline.py` 改造 `record_audio_from_mic` 與 `speech_to_text`，支援 `cancellation_event` 與局部時間切片轉錄。
- [ ] **步驟 3:** 在 `meeting_minutes.py` 實作 `cancel_current_job()` 與 `transcribe_region()`。
- [ ] **步驟 4:** 在 `main.py` 暴露 `/api/record/cancel` 與 `/api/transcribe/region` 端點。
- [ ] **步驟 5:** 在 `ui_client.html` 與 `index.html` 整合 Wavesurfer.js 波形播放器、取消按鈕與區間剪輯功能。
- [ ] **步驟 6:** SCP 部署至樹莓派 5 實體機並重啟服務驗證。
- [ ] **步驟 7:** 瀏覽器端驗收、更新清單並同步至 GitHub / GitLab。
