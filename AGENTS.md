# 🤖 Pi 5 Local Note - Antigravity Agent Project Memory & Context
# 樹莓派 5 邊緣雙語口譯、語音筆記與藍牙網關專案記憶檔案 (AGENTS.md)

> 本文件為 Antigravity AI Agent 專案記憶中樞。當您在此專案目錄（`F:\13_prj_raspi5_localnote`）展開工作時，必須嚴格遵守並繼承下列所有工程規範、歷史經驗、避坑指引與部署設定。

---

## 1. 遠端版本控制與推送規範 (Git Remotes & Push Workflow)

本專案配置有 GitHub 與 GitLab 雙遠端倉庫，任何程式碼更新必須保持雙端同步：

- **GitLab (主倉庫 / Remote: gitlab):**
  - URL: `http://andrew.lee@gitlab.mytekcentral.com:8081/andrew.lee/pi5-hailo-localnote.git`
  - 授權身分: `andrew.lee`
  - 推送指令: `git push gitlab main`
- **GitHub (Remote: origin):**
  - URL: `https://alntu2@github.com/leemailto2008/pi5-hailo-localnote.git`
  - 授權身分: `alntu2` (Windows Credential Manager / PAT)
  - 推送指令: `git push origin main`
- **歷史相容倉庫 (Legacy Remotes):**
  - `origin-legacy`: `https://alntu2@github.com/leemailto2008/pi5-llm-localnote.git`
  - `gitlab-legacy`: `http://andrew.lee@gitlab.mytekcentral.com:8081/andrew.lee/pi5-llm-localnote.git`
- **一鍵雙推規範：**
  - 提交訊息遵循 Conventional Commits（`feat:`, `fix:`, `docs:`, `chore:`）。
  - 驗收完成後執行：`git push gitlab main && git push origin main`。

---

## 2. 樹莓派主機環境與 SSH 連線資訊 (Raspberry Pi 5 Environment)

- **硬體平台:** Raspberry Pi 5 (8GB RAM), Debian 13 (Trixie)
- **網路 IP 位址:**
  - 家用 Wi-Fi 模式 (STA): `192.168.50.228` (Web UI: `http://192.168.50.228:8000/`)
  - 獨立熱點模式 (AP Mode): `10.20.0.1` (SSID: `raspi543_AI`, Web UI: `http://10.20.0.1:8000/`)
- **SSH 遠端連線 (Paramiko / OpenSSH):**
  - 主機: `192.168.50.228` (或熱點時 `10.20.0.1`)
  - 帳號: `pi`
  - 私鑰路徑: `C:\Users\Andrew\.ssh\id_rsa_pi5`
  - 連線命令: `ssh -i C:\Users\Andrew\.ssh\id_rsa_pi5 pi@192.168.50.228`
- **遠端專案運行目錄:**
  - 主目錄: `/home/pi/code_dispatcher/`
  - 音訊快取 (RAM-Disk): `/dev/shm/audio_cache/`
- **核心 Systemd 服務:**
  - `pi5-dispatcher-api.service`: FastAPI 後端主服務 (Uvicorn 8000 port)
  - 重啟指令: `sudo systemctl restart pi5-dispatcher-api.service`

---

## 3. 核心架構決策與重大避坑記憶 (Critical Architectural Gotchas & Decisions)

### ⚠️ [重點一] 樹莓派 5 藍牙 mSBC 16kHz 硬體 UART 總線特性與音訊零爭搶 (Zero-Contention)
- **硬體物理限制:** 樹莓派 5 的藍牙晶片共用單一實體硬體 UART 串列總線 (`/dev/ttyAMA0`)。藍牙寬頻語音 (mSBC, 16kHz) 採用 SCO 通道，嚴格每 7.5ms 傳輸一幀。
- **嚴格禁忌 1 (禁止錄音中執行推論):** 
  - 錄音進行中（`is_recording == True`），**絕對禁止**在背景啟動 faster-whisper 或 CTranslate2 等 CPU/矩陣計算密集模型。OpenMP 多核心運算會引起記憶體匯流排與 CPU 爭搶，導致 `wireplumber` 錯過 7.5ms 時間窗口，噴發大量 `spa.bluez5.source.sco: decode failed: -3` 造成嚴重視訊碎裂與斷流。
  - **解決方案:** 錄音期間將 CPU 0/1 與記憶體頻寬 100% 獨占保留給音訊守護行程；待錄音結束退出 ffmpeg 後，再釋放全 4 核心（`cpu_threads=4`）進行高精度全軌 STT。
- **嚴格禁忌 2 (禁止錄音中執行 HCI 指令):**
  - 錄音期間**絕對禁止**呼叫 `hcitool lq`、`hcitool rssi` 或高頻 `bluetoothctl info` 輪詢。HCI 控制指令會直接阻塞 UART 佇列，導致 SCO 封包丟失引發斷流與 DC 偏移跳變暴音 (Transient Click Artifact)。
  - **解決方案:** 在 `voice_streaming.py` 實裝 `is_any_recording_active()` 全域守衛；API 端點若處於錄音中直接回傳記憶體快取。

### ⚠️ [重點二] 瞬態平滑帶通濾波器 (Bandpass Filter)
- ffmpeg 錄音參數固定採用 `-af "highpass=f=80,lowpass=f=7600"`，濾除 80Hz 以下低頻共振與 7.6kHz 以上的高頻階躍毛刺，杜絕末端暴音。

### ⚠️ [重點三] 零磁碟寫入原則 (Zero-Disk I/O)
- 音訊捕捉全數掛載於 Linux POSIX 記憶體磁碟 (`/dev/shm/audio_cache/`)，不產生任何實體磁碟 WAV/MP3 寫入 (0 Flash Wear)，極致保護樹莓派 MicroSD / NVMe 壽命。

### ⚠️ [重點四] 即時串流分塊轉譯與打字機動效 (Streaming STT & Speculative LLM)
- 單路 segment 切片錄音 (6s/chunk)，錄音結束後以 ffmpeg concat demuxer copy 合併母帶。
- Ollama `qwen2.5:3b-opt` 專為雙語口譯最佳化（System Prompt 約束僅輸出英文、temperature 0.2），首字延遲 TTFT 僅 0.18 秒，以 SSE 事件即時逐 Token 串流輸出並顯示 TPS 速率。

### ⚠️ [重點五] 手機端 Web 控制台最佳化 (Mobile Responsive Web UX)
- **文字捲軸 (Scrollbar):** 語音轉錄與即時英文雙語卡片設定 `max-height: 165px; overflow-y: auto;`，並在 SSE 更新時自動 `scrollTop = scrollHeight` 平滑跟隨最新文字。
- **藍牙卡片折疊 (Collapsible Hub):** 支援一鍵折疊，收合時僅留設備名稱、連線狀態與 🔋 電量膠囊，狀態由 `localStorage` 持久化。
- **受信任耳麥清單 (Trusted Devices):** 支援多耳麥環境，每支耳麥可獨立切換「自動連線 (Auto-Connect)」開關，並支援指定耳麥手動連線與設為預設。

---

## 4. 目錄與模組結構對照 (Module Directory Mapping)

- `code_dispatcher/`: FastAPI 後端 (`main.py`)、語音串流核心 (`voice_streaming.py`)、音訊記憶體緩衝區 (`voice_memory.py`) 與前端儀表板 (`ui.html`)。
- `models/`: Ollama 口譯模型定義檔 (`Modelfile.qwen2.5-3b-opt`)、自動預熱腳本 (`setup_models.sh`) 與模型 Manifests 清單。
- `gpio_driver/`: 樹莓派 5 實體微動按鍵與雙色 LED 狀態指示驅動程式 (`key_led_driver.py`)。
- `telegram_agent/`: Telegram 語音助理 (`agent.py`)、無螢幕藍牙配對管理模組 (`bluetooth_manager.py`) 與自我進化引擎。
- `hotspot.md`: 戶外獨立熱點方案與 nftables 防火牆封包隔離設定指引。
- `todo_tasks.md` & `SPEC.md`: 系統工程規格文件與 30 項模組改良任務追蹤清單。

---

## 5. Agent 行為規範 (Agent Rules & Behavior)

1. **語言與語氣:** 嚴格使用繁體中文 (Traditional Chinese) 回應；關鍵技術詞彙必須採中英對照格式：`中文 (English Term)`。
2. **KISS & No DIY 原則:** 優先採用現成穩定標準模組，避免過度設計；嚴禁留下 TODO 或虛構程式碼。
3. **驗證與自我修復:** 程式碼產出後自我檢查，若遇日誌報錯直接分析原因並自動修正。
