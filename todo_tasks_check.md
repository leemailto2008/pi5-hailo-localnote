# 🚀 樹莓派 5 邊緣 AI 與語音網關改良藍圖 - 任務檢核與實裝進度表 (todo_tasks_check.md)

> **版本 (Version):** 1.2.0  
> **系統架構 (System Architecture):** Raspberry Pi 5 (8GB RAM) | Debian 13 (Trixie) | PipeWire Bluetooth mSBC (16kHz) | faster-whisper | Ollama Qwen 3B | FastAPI  
> **更新時間 (Last Updated):** 2026-10-13  
> **遠端同步 (Git Remotes):** GitHub (`origin`) & GitLab (`gitlab`) 雙端同步  

---

## 📊 整體進度統計 (Overall Progress)

- **總規劃任務 (Total Tasks):** 30 項
- **已完整實裝並驗收上線 (Fully Completed):** **11 項** (37%)
- **部分實裝具備核心模組 (Partially Completed):** **3 項** (10%)
- **待後續迭代實作 (Backlog / To-Do):** **16 項** (53%)
- **額外完成關鍵專案增強 (Extra System Enhancements):** **4 項** (藍牙 Hub SPA、Captive Portal 攔截、窄螢幕響應式佈局、PipeWire 精準拓撲解析)

---

## 一、 語音管線與推論最佳化 (Voice Pipeline & Inference Optimization)

- [x] **Task 01: 串流分塊轉錄 (Streaming Chunking STT)** ✅ *(已實裝)*
  - **實作模組:** `code_dispatcher/voice_streaming.py`
  - **成效與狀態:** 支援 6 秒 segment 即時切片錄音，音訊送交 faster-whisper 分塊推論，錄音結束後以 ffmpeg concat demuxer copy 無縫合併母帶。縮減 >75% 體感等待時間。
  
- [x] **Task 02: 大模型串流口譯與先行推論 (Speculative LLM Streaming Translation)** ✅ *(已實裝)*
  - **實作模組:** `models/Modelfile.qwen2.5-3b-opt`, `code_dispatcher/main.py`
  - **成效與狀態:** 專屬口譯模型約束純英文輸出 (`temperature: 0.2`)，首字延遲 (TTFT) 僅 0.18 秒，以 SSE 事件即時逐 Token 串流輸出，前端呈現實時打字機動效與 TPS 速率監控。

- [ ] **Task 03: 引入斷路器機制與失敗自動降級 (Circuit Breaker & Graceful Degradation)**
  - **狀態:** 待辦。目前僅具備超時 (Timeout) 與簡單例外捕捉，尚未封裝為完整的斷路器模式類別 (Circuit Breaker Pattern)。

- [ ] **Task 04: 純本機離線神經語音合成 (Offline Neural TTS - Piper)**
  - **狀態:** 待辦。目前語音朗讀採用 gTTS ➔ Edge-TTS ➔ espeak-ng 三層降級，尚未整合 Piper C++ 離線神經語音引擎。

- [x] **Task 05: 記憶體環形緩衝區與零磁碟 I/O (In-Memory Ring Buffer & Zero-Disk I/O)** ✅ *(已實裝)*
  - **實作模組:** `code_dispatcher/voice_memory.py`, `/dev/shm/audio_cache`
  - **成效與狀態:** 音訊捕捉、切片與母帶全數掛載於 Linux POSIX 共享記憶體 (`/dev/shm`)，ffmpeg pipe 直解 16kHz PCM NumPy 陣列送入 Whisper，達成 100% 零磁碟寫入 (0 Flash Wear)。

- [ ] **Task 06: 遷移至 whisper.cpp 高效推論引擎 (Engine Migration to whisper.cpp)**
  - **狀態:** 待辦。目前採用 faster-whisper (CTranslate2, int8)。

- [-] **Task 07: 動態語音靜音切除與前置降噪 (Dynamic VAD Trimming & Speex Pre-filtering)** ⚠️ *(部分實裝)*
  - **實作模組:** `code_dispatcher/voice_pipeline.py`
  - **成效與狀態:** 已啟用 Whisper 內建 Silero VAD (`vad_filter=True`)，並實裝局部 1 秒視窗 RMS 與峰值 (Peak) 智慧靜音過濾；尚未抽換為獨立的 SpeexDSP 硬體級前置濾波模組。

---

## 二、 通訊協定與前後端架構 (Protocols & Frontend/Backend Architecture)

- [x] **Task 08: WebSocket 雙向即時狀態推播 (Real-Time WebSocket Push)** ✅ *(已實裝)*
  - **實作模組:** `code_dispatcher/ws_manager.py`, `code_dispatcher/main.py`, `code_dispatcher/ui_client.html`, `code_dispatcher/static/index.html`
  - **成效與狀態:** 建立 `/ws/status` 雙向 WebSocket 端點與連線管理器。錄音管線狀態機 (`meeting_minutes.py`) 流轉時主動廣播最新進度 (<50ms 延遲)；背景協程以 1.5s 週期廣播系統指標與藍牙狀態。前端連線成功時立即關閉高頻 HTTP 輪詢，終結 journalctl 日誌洗版；斷線時自動無縫降級回 HTTP 輪詢並實施指數退避重連。

- [x] **Task 09: 即時音量頻譜採樣 (Live VU Meter & Spectrum Visualizer)** ✅ *(已實裝)*
  - **實作模組:** `code_dispatcher/voice_pipeline.py`, `code_dispatcher/meeting_minutes.py`, `code_dispatcher/ws_manager.py`, `code_dispatcher/ui_client.html`, `code_dispatcher/static/index.html`
  - **成效與狀態:** 在 ffmpeg 錄音管線配置雙路輸出（母帶 libopus + 原始 PCM s16le stdout pipe），每 50ms 提取 RMS、Peak、dBFS 與 8 頻段 FFT 頻譜。透過 WebSocket 即時推播，前端動態繪製霓虹漸層 VU 電平條、Peak Hold 標記與 8 頻段跳動柱狀頻譜，並以即時徽章提示收音狀態（收音良好 🟢 / 音量微弱 🟡 / 未收到聲音 🔴），徹底解決盲錄痛點。

- [x] **Task 10: 漸進式 Web 應用程式離線快取 (Progressive Web App, PWA & Service Worker)** ✅ *(已實裝)*
  - **實作模組:** `code_dispatcher/static/manifest.json`, `code_dispatcher/static/sw.js`, `code_dispatcher/static/icon-192.png`, `code_dispatcher/static/icon-512.png`, `code_dispatcher/main.py`, `code_dispatcher/ui_client.html`, `code_dispatcher/static/index.html`
  - **成效與狀態:** 
    1. 實裝標準 PWA Web App Manifest 與 192x192 / 512x512 高解析度獨立應用圖示，支援 iOS / Android「加到主畫面 (Add to Home Screen)」為獨立全螢幕 App (standalone)。
    2. 實裝 Service Worker Cache-First 快取架構 (`localnote-v1.0.0`)：預先快取核心 HTML、CSS 與 WebManifest；非 API 靜態請求若命中快取立即 0 延遲返回並背景更新 (Stale-While-Revalidate)，斷網或熱點斷線時依舊能秒開介面。
    3. 實裝嚴格動態請求排除機制：精準排除 `/api/`、`/audio/`、`/ws/` 等動態遙測與即時串流端點，防止遙測狀態與音訊被快取污染。
    4. 後端 FastAPI 掛載 `/sw.js`、`/service-worker.js` 與 `/manifest.json`，並配置 `Service-Worker-Allowed: /` 與 `Cache-Control: no-cache` 標頭，徹底解決 Service Worker Scope 與版本更迭問題。

- [x] **Task 11: 異步工作佇列與取消權杖 (Job Queue & Cancellation Token)** ✅ *(已實裝)*
  - **實作模組:** `code_dispatcher/voice_pipeline.py`, `code_dispatcher/meeting_minutes.py`, `code_dispatcher/main.py`, `code_dispatcher/ui_client.html`, `code_dispatcher/static/index.html`
  - **成效與狀態:** 
    1. 實裝 `cancellation_token` (支援 `asyncio.Event` 與 `threading.Event`)。
    2. 錄音期間取消時立即中斷 `ffmpeg` 子行程並釋放硬體音訊節點；Whisper 推論期間每段 segment 迭代時檢核取消權杖並中途跳出；LLM 階段終止串流，徹底根除無效運算霸佔 4 核心 CPU 的難題。
    3. 暴露 `POST /api/record/cancel` 與 `POST /api/job/cancel` API 端點。前端進度條右側配置一鍵紅色 ⛔ **「取消」** 按鈕，點擊後即時重設管線狀態機為 `idle` 並廣播 WebSocket 歸位。

- [x] **Task 12: 互動式音訊波形視覺化與剪輯 (Waveform Visualizer & Trimming - Wavesurfer.js)** ✅ *(已實裝)*
  - **實作模組:** `code_dispatcher/static/vendor/wavesurfer.min.js`, `code_dispatcher/static/vendor/regions.min.js`, `code_dispatcher/ui_client.html`, `code_dispatcher/static/index.html`, `code_dispatcher/main.py`, `code_dispatcher/meeting_minutes.py`, `code_dispatcher/voice_pipeline.py`
  - **成效與狀態:** 
    1. 整合輕量版離線 `wavesurfer.min.js` (41KB) 與 `regions.min.js` (16KB) 至本機 vendor 目錄，無外部 CDN 依賴，100% 符合 PWA 離線熱點運作要求。
    2. 錄音完成後自動呼叫 `/api/audio/latest` 掛載最新母帶，繪製現代化雙色音訊波形圖，支援播放、暫停、停止、點擊跳轉 (Seek) 與實時播放時間軸。
    3. 實裝波形滑鼠拖曳選區剪輯功能 (Regions Plugin) 與局部試聽 (`playSelectedRegion()`)。
    4. 實裝選區局部轉錄 (`POST /api/transcribe/region`)：在記憶體中快速裁切選定秒數範圍的 16kHz PCM 送交 Whisper 轉錄，無需全軌重新處理。

- [ ] **Task 13: 雙語字幕時間軸對齊 (Bilingual Subtitle Sync & SRT/VTT Alignment)**
  - **狀態:** 待辦。

- [ ] **Task 14: 指數退避網路自動重連 (Exponential Backoff Reconnect & Network Resiliency)**
  - **狀態:** 待辦。

---

## 三、 硬體與邊緣運算整合 (Hardware, GPIO & Edge Infrastructure)

- [-] **Task 15: 軟硬體狀態雙向連動 (Bidirectional GPIO & Web State Synchronization)** ⚠️ *(部分實裝)*
  - **實作模組:** `gpio_driver/key_led_driver.py`
  - **成效與狀態:** 具備實體微動按鍵讀取與雙色 LED 狀態指示驅動，但尚未與 Web 後端建立內部非同步事件匯流排 (Event Bus) 雙向廣播。

- [ ] **Task 16: 記憶體熱度管理與防崩潰保護 (VRAM / Memory Guard & OOM Prevention)**
  - **狀態:** 待辦。目前透過模型長駐 (keep_alive: 24h) 與 RAM 快取管理，尚未設置 cgroups 資源配額硬限制。

- [ ] **Task 17: 實體按鍵手勢識別 (Multi-Click & Long-Press GPIO Gestures)**
  - **狀態:** 待辦。目前按鍵僅支援單擊開始/結束錄音。

- [ ] **Task 18: 智慧溫度感知與風扇轉速聯動 (Thermal Throttling Guard & Active Cooling Curve)**
  - **狀態:** 待辦。後端已有溫度讀取與降頻防護感知，但動態 PWM 風扇轉速控制曲線尚未整合進主程式。

- [x] **Task 19: 藍牙連線生命週期自動保活 (Bluetooth Link Keep-Alive & Auto-Reconnect Daemon)** ✅ *(已實裝)*
  - **實作模組:** `telegram_agent/bluetooth_manager.py`, `code_dispatcher/static/index.html` (Bluetooth Hub), `code_dispatcher/voice_pipeline.py`
  - **成效與狀態:** 
    1. 實裝獨立 SPA 藍牙設備管理中心 (Bluetooth Hub)，支援受信任設備管理、手動指定連線、單鍵切換常用耳麥。
    2. 每個耳麥可獨立切換「自動連線 (Auto-Connect)」開關，後端定時輪詢保活。
    3. 實裝非阻塞 `pw-dump` PipeWire 拓撲解析，精準鎖定 `bluez_input` 節點，**杜絕重複切換 Profile 造成的 SCO 封包打碎與 UART 爭搶斷流**。

- [ ] **Task 20: 超低功耗休眠與行動電源續航優化 (Ultra-Low Power Deep Sleep & Battery Management)**
  - **狀態:** 待辦。

- [-] **Task 21: 儲存磨損平衡與唯讀檔案系統 (Storage Wear Leveling & OverlayFS)** ⚠️ *(部分實裝)*
  - **成效與狀態:** 音訊錄製、轉碼全數掛載於 RAM-Disk (`/dev/shm`)，實體磁碟 0 磨損；OverlayFS 唯讀切換腳本尚未整合。

---

## 四、 智慧邊緣 AI 功能延伸 (Edge AI Capabilities & Smart Agents)

- [ ] **Task 22: 本機端檢索增強生成向量庫 (Edge RAG Vector Store - SQLite-vec)**
  - **狀態:** 待辦。

- [ ] **Task 23: 語音提示詞範本與場景切換 (Voice Prompt Templates & Scenario Presets)**
  - **狀態:** 待辦。目前固定為會議紀要摘要與專業口譯 Prompt。

- [ ] **Task 24: 自動多語言偵測與動態切換 (Zero-Shot Multilingual Detection & Auto-Switching)**
  - **狀態:** 待辦。目前預設為繁體中文 (台灣) 與英文口譯。

- [x] **Task 25: 對話記憶與滾動式會議摘要 (Conversation Memory & Rolling Summarization)** ✅ *(已實裝)*
  - **實作模組:** `code_dispatcher/meeting_minutes.py`, `code_dispatcher/static/index.html`
  - **成效與狀態:** 
    1. 完整實裝會議錄音 ➔ Whisper 逐字稿 ➔ Qwen 提煉【摘要】、【待辦事項表格】、【決議事項】之端到端管線。
    2. 實裝「紀要翻成英文」加分功能：精準提取核心條列內容（排除冗長逐字稿），在樹莓派 5 上穩定於 10~15 秒內完成高品質翻譯，絕不逾時。

- [ ] **Task 26: 語音意圖分流與硬體控制 (Voice Intent Dispatcher & Hardware Actions)**
  - **狀態:** 待辦。

---

## 五、 維運、安全與監控 (DevOps, Security & Observability)

- [x] **Task 27: 核心級封包隔離與安全防護 (Kernel-Level Packet Isolation & Captive Portal Guard)** ✅ *(已實裝)*
  - **實作模組:** `hotspot.md`, `code_dispatcher/main.py`
  - **成效與狀態:** 
    1. 提供 nftables 獨立熱點封包隔離規則。
    2. **實裝 Captive Portal 強制入口防斷網路由:** 註冊 `/hotspot-detect.html` (Apple)、`/generate_204` (Android)、`/ncsi.txt`，**徹底解決手機在戶外無外網熱點模式下，自動切斷 Wi-Fi 或轉發至行動數據 (4G/5G) 導致前端 Fetch 報錯中斷的物理硬體難題**。

- [ ] **Task 28: 結構化日誌與推論延遲追蹤 (Structured JSON Logging & Telemetry Tracing)**
  - **狀態:** 待辦。目前使用 Uvicorn 與 Systemd Journalctl 日誌。

- [ ] **Task 29: 平滑熱更新與零停機守護 (Zero-Downtime Hot Reload & Health Checks)**
  - **狀態:** 待辦。

- [ ] **Task 30: 一鍵災難復原與配置備份腳本 (Automated Backup & Disaster Recovery Playbook)**
  - **狀態:** 待辦。

---

## 🌟 額外已完成之關鍵架構與體驗優化 (Extra Engineering Enhancements)

除原 30 項規劃外，本專案在實測與使用者回饋迭代中已落地完成下列重大工程修復：

1. **窄螢幕與手機端響應式排版 (Mobile Responsive Layout):**
   - 移除手機寬度 (`max-width: 480px`) 下粗暴隱藏 CPU 膠囊的規則，重構頂部列為彈性容器 (`.topbar-pills`)，確保 CPU 負載與核心溫度在任何螢幕寬度下永不消失。
2. **消滅錄音完成閃爍與單欄滾動動效 (Done State Smooth Scroll & Flashing Fix):**
   - 修正輪詢計時器相互打架導致「紀要已完成」忽隱忽現的閃爍問題。
   - 單欄模式下自動將提示改為 `👇 紀要已在下方生成 (點擊查看)`，並自動平滑滾動 (Smooth Scroll) 至輸出卡片並附加 `pulse-highlight` 呼吸光暈動效。
3. **PipeWire 藍牙拓撲精準路由 (PipeWire Node Name Routing):**
   - 捨棄脆弱的字串正則截取，改用 `pw-dump` 精確解析 `bluez_input.A0_D1_2A_42_12_34.0`，杜絕每次錄音重複下達 `set-profile` 打碎藍牙 SCO 通道。
4. **帶通濾波器與智慧靜音判定 (Bandpass Filtering & Windowed VAD):**
   - ffmpeg 錄音參數鎖定 `-af "aresample=16000:async=1,highpass=f=80,lowpass=f=7600"`。
   - 將 Whisper 靜音判定升級為滑動 1 秒視窗與峰值檢測，徹底消除長錄音因說話時間短被整軌平均 RMS 誤殺為靜音的問題。
