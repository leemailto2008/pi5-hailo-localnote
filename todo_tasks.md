# 🚀 樹莓派 5 邊緣 AI 與語音網關改良藍圖 (Pi 5 Edge AI Arena Improvement Roadmap)

> **版本 (Version):** 1.0.0  
> **系統架構 (Architecture):** Raspberry Pi 5 (8GB) | Debian 13 Bookworm/Trixie | PipeWire Bluetooth mSBC | faster-whisper | Ollama Qwen 3B | FastAPI  
> **最後更新 (Last Updated):** 2026-10-08  

本清單依據系統架構深度、可維護性 (Maintainability)、使用者體驗與極致效能，規劃 30 個具體、結構化且可落地的改良任務。

---

## 一、 語音管線與推論最佳化 (Voice Pipeline & Inference Optimization)

- [x] **Task 01: 串流分塊轉錄 (Streaming Chunking STT)** ✅ *(已實裝 - SPEC.md / voice_streaming.py / SSE 串流)*
  - **現狀問題：** 30 秒或 60 秒錄音必須等待整段錄音結束才開始推論，使用者面臨長達數十秒的尾端等待。
  - **改良方案：** 透過 ffmpeg 邊錄音邊進行 6 秒分塊 (Chunking)，各分塊即時送交 Whisper 轉錄，並以 SSE 即時推播進度至 Web。
  - **效益：** 縮減 >75% 體感等待時間，錄音結束時尾段僅需 2~3 秒即可完成全文。

- [x] **Task 02: 大模型串流口譯與先行推論 (Speculative LLM Streaming Translation)** ✅ *(已實裝 - SPEC.md / qwen2.5:3b-opt / SSE 打字機動效)*
  - **現狀問題：** STT 必須全文辨識完成後才一次性呼叫 Qwen，首字延遲 (Time to First Token, TTFT) 明顯。
  - **改良方案：** 在分塊產出時即時先行預熱 (Speculative Pre-warm)，並透過 SSE 逐 Token 串流輸出，TTFT 僅 0.18 秒，前端呈現實時打字機效果。
  - **效益：** 消除轉譯等待空白，首字輸出延遲縮短 95%，大幅提升口譯順暢度。

- [ ] **Task 03: 引入斷路器機制與失敗自動降級 (Circuit Breaker & Graceful Degradation)**
  - **現狀問題：** 若 Ollama 或音訊播放器因瞬時高負載無回應，前端將陷入無止境掛起 (Hang)。
  - **改良方案：** 在後端實作斷路器模式 (Circuit Breaker Pattern)。單一推論步驟超過閥值 (如 12 秒) 立即熔斷，精確回報故障階段並支援一鍵單步重試。
  - **效益：** 根絕前端卡死假死現象，提升系統自我修復能力 (Self-healing)。

- [ ] **Task 04: 純本機離線神經語音合成 (Offline Neural TTS - Piper)**
  - **現狀問題：** 目前 Edge-TTS 依賴網際網路連線 (Internet Connection)，在離線熱點模式 (AP Mode) 或離網環境無法發聲。
  - **改良方案：** 整合輕量級 Piper TTS（C++ 引擎，針對 ARM NEON 優化，即時率 RTF < 0.2），建立完全離線的中英雙語神經發音庫。
  - **效益：** 實現 100% 離網環境語音互動。

- [x] **Task 05: 記憶體環形緩衝區與零磁碟 I/O (In-Memory Ring Buffer & Zero-Disk I/O)** ✅ *(已實裝 - SPEC.md / /dev/shm / Zero-Disk PCM Streaming)*
  - **現狀問題：** 流程中歷經多次實體磁碟讀寫（錄音寫 OGG ➔ ffmpeg 轉 MP3 ➔ Whisper 轉 WAV）。
  - **改良方案：** 將音訊快取掛載至 Linux POSIX 共享記憶體 (`/dev/shm/audio_cache`)，並以 ffmpeg pipe 直解 16kHz PCM NumPy 陣列直接送入 Whisper，徹底消除暫存 WAV 檔。
  - **效益：** 達成 100% 零磁碟寫入 (0 Flash Wear)，消除磁碟 I/O 延遲，延長樹莓派 MicroSD / NVMe 壽命。

- [ ] **Task 06: 遷移至 whisper.cpp 高效推論引擎 (Engine Migration to whisper.cpp)**
  - **現狀問題：** Python faster-whisper 仍受限於 Python 全域直譯器鎖 (Global Interpreter Lock, GIL) 與 runtime 開銷。
  - **改良方案：** 評估遷移至純 C/C++ 實作的 `whisper.cpp`，搭配 Q5_K_M 量化模型與 ARM 向量指令集加速。
  - **效益：** 記憶體佔用降低 35%，CPU 推論速度提升約 30%。

- [ ] **Task 07: 動態語音靜音切除與前置降噪 (Dynamic VAD Trimming & Speex Pre-filtering)**
  - **現狀問題：** 錄音首尾常包含空白靜音或環境背景雜訊，浪費 Whisper 計算資源。
  - **改良方案：** 在音訊送交 STT 前，以輕量級 SpeexDSP / WebRTC VAD 進行硬體級前置降噪與靜音切除。
  - **效益：** 減少 20~40% 無效推論時間，提升嘈雜環境辨識率。

---

## 二、 通訊協定與前後端架構 (Protocols & Frontend/Backend Architecture)

- [ ] **Task 08: WebSocket 雙向即時狀態推播 (Real-Time WebSocket Push)**
  - **現狀問題：** Web 介面依賴定時 HTTP 輪詢 (Polling) 來取得藍牙狀態、電量與管線進度，消耗網路與 CPU。
  - **改良方案：** 建立 WebSocket 雙向通道。藍牙耳麥連線/斷線、電量變化、STT 階段完成時由後端主動推播。
  - **效益：** 狀態更新 0 延遲，消除輪詢造成的無謂資源開銷。

- [ ] **Task 09: 即時音量頻譜採樣 (Live VU Meter & Spectrum Visualizer)**
  - **現狀問題：** 使用者在錄音中無法確定耳麥麥克風是否有收到聲音，若靜音需等失敗後才發覺。
  - **改良方案：** 錄音進行時從 PipeWire 提取即時均方根 (RMS Amplitude) 音量，以 50ms 週期推播至前端繪製動態跳動音量條。
  - **效益：** 錄音視覺回饋即時，大幅降低無效錄音與重錄成本。

- [ ] **Task 10: 漸進式 Web 應用程式離線快取 (Progressive Web App, PWA & Service Worker)**
  - **現狀問題：** 離線熱點模式下首次載入靜態資源時若快取失效可能導致頁面不完整。
  - **改良方案：** 實作標準 PWA Manifest 與 Service Worker Cache-First 策略，將 CSS、圖示與 JS 100% 本地快取，並支援加入手機主畫面。
  - **效益：** 手機端如原生 App 般瞬間開啟，離網環境穩定度達 100%。

- [ ] **Task 11: 異步工作佇列與取消權杖 (Job Queue & Cancellation Token)**
  - **現狀問題：** 當使用者發起一次長時間轉錄後若想中斷或重新錄音，舊任務仍會佔滿 CPU 直至完成。
  - **改良方案：** 導入任務排程佇列 (In-process Task Queue) 與取消權杖 (Cancellation Token)，前端點擊取消時能瞬間終止背景推論。
  - **效益：** 避免無效運算霸佔 CPU，提升連續操作彈性。

- [ ] **Task 12: 互動式音訊波形視覺化與剪輯 (Waveform Visualizer & Trimming - Wavesurfer.js)**
  - **現狀問題：** 錄音試聽僅能透過簡易 HTML5 audio 播放器，無法得知音訊段落分布。
  - **改良方案：** 整合 Wavesurfer.js，提供互動式音訊波形可視化，並允許使用者圈選特定秒數進行局部轉錄。
  - **效益：** 介面現代化，提升精準校對與試聽體驗。

- [ ] **Task 13: 雙語字幕時間軸對齊 (Bilingual Subtitle Sync & SRT/VTT Alignment)**
  - **現狀問題：** 轉錄與翻譯文字僅以單一純文字區塊呈現，無法得知何時說了哪句話。
  - **改良方案：** 利用 Whisper 輸出的 word-level / segment-level timestamps，生成含時間軸的 SRT/VTT 雙語字幕，並在試聽音訊時高亮當前語句。
  - **效益：** 提升長段落錄音的閱讀性與專業商務口譯價值。

- [ ] **Task 14: 指數退避網路自動重連 (Exponential Backoff Reconnect & Network Resiliency)**
  - **現狀問題：** 手機從家用 Wi-Fi 切換到樹莓派熱點時，網路瞬斷容易導致前端 JavaScript 報錯停止。
  - **改良方案：** 在前端 API 呼叫層封裝指數退避重試演算法 (Exponential Backoff)，並監聽 `navigator.onLine` 事件平滑恢復連線。
  - **效益：** 網路漫遊切換無感，大幅提升行動端體驗。

---

## 三、 硬體與邊緣運算整合 (Hardware, GPIO & Edge Infrastructure)

- [ ] **Task 15: 軟硬體狀態雙向連動 (Bidirectional GPIO & Web State Synchronization)**
  - **現狀問題：** 實體按鍵 (GPIO Key) 與 Web UI 操作各自獨立運作。
  - **改良方案：** 透過內部事件匯流排 (Event Bus) 串聯。按下實體鍵錄音時 Web 儀表板同步啟動；Web 點擊錄音時硬體雙色 LED 同步進入呼吸閃爍。
  - **效益：** 打造軟硬體一體化的多模態互動體驗。

- [ ] **Task 16: 記憶體熱度管理與防崩潰保護 (VRAM / Memory Guard & OOM Prevention)**
  - **現狀問題：** Whisper Small (約 1GB) + Qwen 3B (約 2.2GB) 同時運作易導致記憶體碎片化，極端狀況可能觸發 Linux OOM Killer。
  - **改良方案：** 建立動態釋放策略 (LRU Cache Eviction) 與 cgroups 資源限制，監控系統可用 RAM 低於 800MB 時主動卸載非核心模組。
  - **效益：** 保障系統 7x24 高度穩定運作。

- [ ] **Task 17: 實體按鍵手勢識別 (Multi-Click & Long-Press GPIO Gestures)**
  - **現狀問題：** 目前 GPIO 實體按鍵僅支援單一單擊行為。
  - **改良方案：** 在 `key_led_driver.py` 加入短按 (開始/停止錄音)、雙擊 (切換家用 Wi-Fi / AP 熱點)、長按 3 秒 (安全關機) 等多重手勢。
  - **效益：** 無需螢幕與手機即可完全操控樹莓派核心功能。

- [ ] **Task 18: 智慧溫度感知與風扇轉速聯動 (Thermal Throttling Guard & Active Cooling Curve)**
  - **現狀問題：** 連續執行 Whisper STT 與 LLM 會導致 CPU 滿載產熱，可能觸發 80°C 降頻 (Thermal Throttling)。
  - **改良方案：** 整合 CPU 溫度讀取與動態風扇 PWM 控制，並在溫度過高時動態調配 OpenMP 執行緒數。
  - **效益：** 預防硬體降頻，維持推論效能穩定於峰值。

- [ ] **Task 19: 藍牙連線生命週期自動保活 (Bluetooth Link Keep-Alive & Auto-Reconnect Daemon)**
  - **現狀問題：** 耳麥閒置過久或短暫離身可能自動休眠斷線，重連時常需要手動操作。
  - **改良方案：** 撰寫 D-Bus 藍牙守護行程 (Bluetooth Daemon)，偵測配對耳麥廣播時自動秒級連線，並送出無聲保活封包防止耳麥自動關機。
  - **效益：** 拿起耳麥隨時可用，免去重複配對困擾。

- [ ] **Task 20: 超低功耗休眠與行動電源續航優化 (Ultra-Low Power Deep Sleep & Battery Management)**
  - **現狀問題：** 樹莓派 5 預設待機功耗約 2.5~3W，使用 10000mAh 行動電源連續運作時間有限。
  - **改良方案：** 在無網路連線且無錄音 10 分鐘後，自動關閉 HDMI、關閉未用 USB 埠與降頻 CPU，待按鍵觸發時微秒級喚醒。
  - **效益：** 延長野外行動續航力達 2~3 倍。

- [ ] **Task 21: 儲存磨損平衡與唯讀檔案系統 (Storage Wear Leveling & OverlayFS)**
  - **現狀問題：** 長期頻繁寫入日誌與音訊快取對 MicroSD / 快閃儲存顆粒造成磨損。
  - **改良方案：** 將音訊快取目錄 (`/tmp/audio_cache`) 掛載為 `tmpfs` (RAM Disk)，並提供可切換的 OverlayFS 唯讀根檔案系統選項。
  - **效益：** 徹底避免異常斷電引發的檔案損毀與儲存卡耗損。

---

## 四、 智慧邊緣 AI 功能延伸 (Edge AI Capabilities & Smart Agents)

- [ ] **Task 22: 本機端檢索增強生成向量庫 (Edge RAG Vector Store - SQLite-vec)**
  - **現狀問題：** 目前口譯僅依賴單句直譯，缺乏專有名詞、特定領域上下文與公司知識庫支援。
  - **改良方案：** 整合超輕量級 `sqlite-vec` 或 ChromaDB 本地嵌入 (Embeddings)，錄音辨識後自動檢索特定語料庫補充專用名詞翻譯。
  - **效益：** 專有名詞與專業術語口譯準確率大幅提升。

- [ ] **Task 23: 語音提示詞範本與場景切換 (Voice Prompt Templates & Scenario Presets)**
  - **現狀問題：** 翻譯提示詞 (System Prompt) 固定為通用口譯，缺乏風格彈性。
  - **改良方案：** 介面提供「商務正式」、「輕鬆口語」、「逐字速記摘要」、「中翻日/西」等多場景預設樣版切換。
  - **效益：** 拓寬邊緣 AI 適用場景與實用性。

- [ ] **Task 24: 自動多語言偵測與動態切換 (Zero-Shot Multilingual Detection & Auto-Switching)**
  - **現狀問題：** 目前 STT 固定強制指定中文 (`language="zh"`)。
  - **改良方案：** 啟用 Whisper 語言自動偵測 (`language=None`) 或輕量級 FastText 語種分類，支援自動判斷英語、日語、台語並動態雙向互譯。
  - **效益：** 實現真正跨語言隨說隨翻。

- [ ] **Task 25: 對話記憶與滾動式會議摘要 (Conversation Memory & Rolling Summarization)**
  - **現狀問題：** 每次錄音獨立處理，缺乏對話脈絡記憶。
  - **改良方案：** 實作本機滾動式上下文快取 (Rolling Window Buffer)，將多次錄音組織為會議紀錄，並支援「一鍵生成重點摘要與待辦事項」。
  - **效益：** 由單純翻譯工具躍升為個人 AI 邊緣秘書。

- [ ] **Task 26: 語音意圖分流與硬體控制 (Voice Intent Dispatcher & Hardware Actions)**
  - **現狀問題：** 語音僅用於文字翻譯，未與樹莓派作業系統功能連動。
  - **改良方案：** 增加輕量級意圖分類器 (Intent Classifier)。當說出「切換熱點」、「系統關機」、「現在溫度」、「查看剩餘電量」時，自動分流執行本機指令。
  - **效益：** 實現完全用語音控制的無人邊緣主機。

---

## 五、 維運、安全與監控 (DevOps, Security & Observability)

- [ ] **Task 27: 核心級封包隔離與安全防護 (Kernel-Level Packet Isolation & nftables Guard)**
  - **現狀問題：** 手機連上獨立熱點時，背景常有大量 App 嘗試向外連網，佔用 Wi-Fi 空口時間 (Airtime)。
  - **改良方案：** 持續完善 `nftables.hotspot_guard.nft`，針對非 10.20.0.1 請求進行快速 TCP RST / ICMP Unreachable 拒絕，不浪費 Pi 5 CPU 與頻寬。
  - **效益：** 保障熱點連線極致純淨與零頻寬浪費。

- [ ] **Task 28: 結構化日誌與推論延遲追蹤 (Structured JSON Logging & Telemetry Tracing)**
  - **現狀問題：** 除錯需透過 `journalctl` 翻查純文字 Log，難以量化統計各階段精確耗時分佈。
  - **改良方案：** 導入結構化 JSON 日誌與輕量級指標收集中介軟體 (Prometheus / JSON Metric Middleware)，量化追蹤 P50/P95/P99 延遲。
  - **效益：** 提供客觀效能基線，快速鎖定效能退化點。

- [ ] **Task 29: 平滑熱更新與零停機守護 (Zero-Downtime Hot Reload & Health Checks)**
  - **現狀問題：** 更新程式碼重啟服務時，現有連線可能中斷。
  - **改良方案：** 設定 Uvicorn 雙 Worker 平滑轉移或使用 Systemd Socket Activation，部署新版本時無縫接軌。
  - **效益：** 系統維護零服務中斷。

- [ ] **Task 30: 一鍵災難復原與配置備份腳本 (Automated Backup & Disaster Recovery Playbook)**
  - **現狀問題：** 包含 Systemd 服務、nftables 設定、Bluetooth Profile 與 Python 虛擬環境分佈於多個路徑，缺乏一體化災難備份。
  - **改良方案：** 撰寫宣告式 (Declarative) 部署與備份腳本 (`backup_restore.sh`)，將系統狀態壓成加密封存檔並支援 3 分鐘全新機器還原。
  - **效益：** 消除單點故障隱憂，提升專案工程標準。
