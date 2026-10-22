# 🤖 Pi 5 Local Note - Claude Code Instructions & Project Memory (CLAUDE.md)
# 樹莓派 5 邊緣雙語口譯、語音筆記與藍牙網關專案記憶檔案

> 本文件為 Claude Code / Claude Desktop 專案記憶與行為準則中樞。在此專案目錄（`F:\13_prj_raspi5_localnote`）工作時，必須嚴格遵守下列所有工程規範、歷史經驗、避坑指引與部署設定。

---

## 1. 角色身分與行為準則 (Identity & Core Rules)

- **Role:** 資深主任工程師與架構師 (Senior Staff Software Engineer & Architect)。極度重視效率、正確性與可維護性 (Maintainability)。
- **Tone:** 直接、精簡、專業。不說廢話、不進行無意義的客套。
- **Language:** **嚴格使用繁體中文 (Traditional Chinese)** 進行回應。
- **Terminology:** 關鍵技術詞彙必須同時提供中英文對照，格式為：`中文 (English Term)`。
  - *範例:* 「請進行重構 (Refactor) 並確認依賴注入 (Dependency Injection)」。
- **KISS 原則 (Keep It Simple, Stupid) & 拒絕自幹 (No DIY):**
  - 最簡邏輯、可讀性至上。避免過度設計 (Over-engineering)。
  - 優先使用現成標準套件或工具，絕不重新發明輪子 (Don't Reinvent the Wheel)。
- **No Hallucinations:** 嚴禁編造代碼或不存在的 API 參數；不確定時明確告知。
- **No Placeholders:** 嚴禁留下 `// TODO` 或未完成區塊，程式碼產出必須完整且可執行 (Production-ready)。
- **Error Handling:** 顯式錯誤處理 (Explicit Error Handling)，嚴禁使用 bare `try: ... except: pass`。
- **Async First:** 涉及 I/O (網路請求、DB) 一律使用 `async/await`。

---

## 2. 遠端版本控制與推送規範 (Git Remotes & Push Workflow)

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

## 3. 樹莓派主機環境與 SSH 連線資訊 (Raspberry Pi 5 Environment)

- **硬體平台:** Raspberry Pi 5 (raspi543: 16GB RAM / raspi678: 8GB RAM), Debian 13 (Trixie), kernel 6.18.50 (raspi543 於 2026-10-22 升級；Hailo 驅動 `hailo1x_pci` 走 DKMS)
- **網路 IP 位址 (兩台主機):**
  - `raspi543` 家用 Wi-Fi (STA): `192.168.50.229` (Web UI: `http://192.168.50.229:8000/`，Uvicorn 實際監聽 80，由 `pi5-port8000-proxy.service` 轉發)
  - `raspi678` 家用 Wi-Fi (STA): `192.168.50.228`（常處於熱點模式供測試人員使用）
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

## 4. 核心架構決策與重大避坑記憶 (Critical Architectural Gotchas & Decisions)

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
- **LLM 後端統一走 `code_dispatcher/llm_backend.py`**（2026-10-22 起）：`hailo`（hailo-ollama `127.0.0.1:11435` / `qwen3:1.7b` on Hailo-10H，CPU 歸零）優先，健康檢查 20s 快取，失敗自動退避 `cpu`（Ollama `11434` / `qwen2.5:3b-opt` → `qwen2.5:3b`）。環境變數 `PI5_LLM_BACKEND=cpu` 可強制切回。`/health` 回傳 `llm_backend` 顯示當前後端。
- 實測效能：Hailo qwen3:1.7b TTFT ≈ 0.6s、≈ 4.5 tok/s；CPU qwen2.5:3b-opt TTFT ≈ 0.2s、≈ 6 tok/s。**Hailo 版的價值是 CPU 零占用（藍牙零爭搶），不是速度。**
- 所有 LLM 呼叫一律用 `chat_complete / iter_chat_stream / achat_complete / aiter_chat_stream`，嚴禁再直接 `requests.post("…11434/api/generate")`。

### ⚠️ [重點五] 手機端 Web 控制台最佳化 (Mobile Responsive Web UX)
- **文字捲軸 (Scrollbar):** 語音轉錄與即時英文雙語卡片設定 `max-height: 165px; overflow-y: auto;`，並在 SSE 更新時自動 `scrollTop = scrollHeight` 平滑跟隨最新文字。
- **藍牙卡片折疊 (Collapsible Hub):** 支援一鍵折疊，收合時僅留設備名稱、連線狀態與 🔋 電量膠囊，狀態由 `localStorage` 持久化。
- **受信任耳麥清單 (Trusted Devices):** 支援多耳麥環境，每支耳麥可獨立切換「自動連線 (Auto-Connect)」開關，並支援指定耳麥手動連線與設為預設。

### ⚠️ [重點六] 藍牙耳麥掃描與配對：BR/EDR 專掃 + 單一會話 + 守護行程不得碰未配對裝置 (2026-10-22 實測)
- **症狀:** Web 掃描看不到 RichChipAN（iPhone 看得到）、配對回 500、耳麥一被「連線」就退出配對模式。
- **根因 1 — `scan on` 交錯輪詢:** HFP 耳麥走傳統藍牙 (BR/EDR)，`transport auto` 在 LE/BR-EDR 間交錯，8 秒窗口經常錯過；同一支耳麥的 BLE 分身（隨機位址 `C0:…`，名稱 `Air Shutter`）卻每次都看得到，使用者點它 → `trust` 成功 `pair` 失敗 → 留下幽靈。**一律用 `scan bredr`，未配對的 `(random)` 位址不列出。**
- **根因 2 — BlueZ 暫存設備即時清除:** discovery 結束後 `info MAC` 立刻 `not available`，分開呼叫 `scan` 再 `pair` 必失敗。**`scan → pair → trust → connect` 必須在同一個 `bluetoothctl` 會話內完成**（`BluetoothctlSession`，以 pty 驅動；plain pipe 會被 readline 全緩衝）。
- **根因 3 — 自動重連守護行程搶連:** `auto_reconnect_tick()` 對 `default_mac` 無條件 `connect`，耳麥一進配對模式就被它連上又斷開，直接踢出配對模式。**只對 `devices Paired` 的裝置重連；`_bt_exclusive` 鎖住期間（掃描/配對中）一律跳過。**
- **順序原則:** `pair` 成功後才 `trust`，杜絕 trusted-but-unpaired 幽靈。
- **除錯工具:** `tools/pi_run.cmd <ip> <script.sh>`、`tools/pi_push.cmd <ip> <local> <remote> [script.sh]`，輸出在 `tools/pi_probe_out/`（已 gitignore）。

### ⚠️ [重點七] hailo-ollama (0.5.1) API 相容性陷阱 (2026-10-22 實測)
- 只有 `/api/chat` 會遵守 system message；`/api/generate` 的 `system` 欄位被忽略。
- `keep_alive` 必須是整數或省略，Ollama 風格字串 `"24h"` → HTTP 500 `Value is NOT a Primitive type`。
- 回應沒有 `eval_duration / prompt_eval_duration`，TPS 必須在客戶端計時。
- **prompt 內含原始換行 / Tab / 反斜線 → HTTP 500 `Failed to render prompt (status=8)`**（它內部組 JSON 只跳脫引號）。`llm_backend._sanitize_for_hailo()` 把 `\n`→字面`\n`、`\t`→字面`\t`、`\`→`\\`，模型端會還原成真正的換行。
- 並行請求會在 server 端排隊（不會錯），但會拉長 TTFT。
- `curl` 測試務必帶 `-H "Content-Type: application/json"`，否則 oatpp 回 `No suitable mapper`。

### ⚠️ [重點八] 藍牙耳麥錄音靜音/破碎的真正根因：mSBC (EDR 2-EV3) 位元錯誤 → 強制 CVSD (2026-10-22 實測結案)
- **症狀:** 錄音檔 20–70% 樣本為數位零值（100–500ms 整段空洞），wireplumber 大量 `spa.bluez5.source.sco: decode failed: -3`，Whisper 輸出幻覺。
- **已排除（全部實測）:** CPU 爭搶、Wi-Fi 共存（含完全關閉）、PipeWire/bluez/核心版本、Hailo HAT（拆除仍壞）、PCIe Gen1/ASPM、sniff 模式、SCO 雙向保活、AFH 頻道圖、耳麥本身（iPhone 錄音乾淨）。
- **硬證據:** btmon 顯示 SCO 封包 1333/10s **一個不少**地到達主機；但開啟 Erroneous Data Reporting (`hcitool cmd 0x03 0x005B 01`) 後控制器標記 **65% 封包 pkt_status=1（資料損壞）**；payload 的 mSBC H2 標頭有 1–3 bit 隨機錯誤、SBC CRC 失敗。即此 CYW43455 ↔ 耳麥鏈路的 **EDR π/4-DQPSK (2-EV3)** 封包位元錯誤率過高，mSBC 只能走 2-EV3，解碼失敗即補零。
- **解法:** `deploy/wireplumber.conf.d/51-bluez-cvsd.conf` → `bluez5.enable-msbc = false`，HFP 固定 **CVSD（基本速率 GFSK EV3，8kHz）**。profile 名稱仍為 `headset-head-unit`，`voice_pipeline.ensure_bluetooth_recording_ready()` 不需修改。實測零值降至 2–20%，逐字稿恢復可讀。
- **注意:** `hcitool rssi` 對 BR/EDR 回傳的是相對黃金接收範圍的差值，不是 dBm；此板子回報值不隨距離變化，不可作為訊號強度依據。
- **殘餘限制:** CVSD 下仍有 10–20% 封包 `pkt_status=2`（無資料）→ 零值空洞最長可達 1s。這是該射頻鏈路的物理上限；要再改善只能換射頻路徑（USB 藍牙 dongle）或換耳麥。
- **除錯工具:** `tools/pi_scostat` 系列腳本（btsnoop 解析 H2 標頭/CRC/旗標）、`/tmp/analyze.py`（零值統計）。

---

## 5. 目錄與模組結構對照 (Module Directory Mapping)

- `code_dispatcher/`: FastAPI 後端 (`main.py`)、語音串流核心 (`voice_streaming.py`)、音訊記憶體緩衝區 (`voice_memory.py`)、LLM 後端選擇層 (`llm_backend.py`)、藍牙管理 (`bluetooth_manager.py`) 與前端儀表板 (`ui.html`)。
- `models/`: Ollama 口譯模型定義檔 (`Modelfile.qwen2.5-3b-opt`)、自動預熱腳本 (`setup_models.sh`) 與模型 Manifests 清單。
- `gpio_driver/`: 樹莓派 5 實體微動按鍵與雙色 LED 狀態指示驅動程式 (`key_led_driver.py`)。
- `telegram_agent/`: Telegram 語音助理 (`agent.py`)、無螢幕藍牙配對管理模組 (`bluetooth_manager.py`) 與自我進化引擎。
- `deploy/`: 樹莓派系統層設定檔（WirePlumber CVSD 強制、systemd 日誌 drop-in），安裝方式見各檔案開頭註解。
- `hotspot.md`: 戶外獨立熱點方案與 nftables 防火牆封包隔離設定指引。
- `todo_tasks.md` & `SPEC.md`: 系統工程規格文件與 30 項模組改良任務追蹤清單。
