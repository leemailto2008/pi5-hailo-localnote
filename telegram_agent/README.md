# 🎙️ Raspberry Pi 5 專屬 Telegram 雙向語音 AI 助理與三層記憶金字塔
# (Pi 5 Telegram Voice AI Agent with Three-Tier Memory Hierarchy)

> **邊緣硬體環境 (Target Environment):** Raspberry Pi 5 (ARM Cortex-A76 四核心 @ 2.40 GHz, 16GB LPDDR4X RAM)  
> **核心定位:** 100% 本地端私有化、零雲端推論費用、具備雙向語音 (Voice-In / Voice-Out) 與長期向量記憶之個人 AI 助理。

---

## 🌟 核心特色 (Key Highlights)

1. **雙向語音極速閉環 (Bi-directional Voice Pipeline):**
   - **語音轉文字 (STT):** Telegram Voice (.oga) $\to$ `ffmpeg` 轉碼 $\to$ `faster-whisper` (ARM NEON int8 量化)，1 秒語音僅需 0.35 秒本地轉錄。
   - **文字轉語音 (TTS):** LLM 回應文字 $\to$ `edge-tts` (微軟台灣繁體中文神經女聲 `zh-TW-HsiaoChenNeural`)，生成自然流暢的 Telegram 語音訊息 (.ogg)。
2. **三層記憶金字塔架構 (Three-Tier Memory Hierarchy):**
   - **Tier 1 (工作記憶 Working Buffer):** 記憶體滑動視窗保留最近 8 輪對話，無縫銜接話題。
   - **Tier 2 (個人事實畫像 User Profile KV):** SQLite 持久化儲存使用者名稱、偏好與重要事實，永久常駐 System Prompt。
   - **Tier 3 (長期語意向量記憶庫 Vector DB):** 本地 Ollama `nomic-embed-text` (768 維度) + SQLite Cosine KNN 向量檢索，按需精準召回歷史知識。
3. **無縫綁定 10 款調優模型:**
   - 預設調用 `qwen2.5:3b-opt` (繁中最佳，6 tok/s，單次推論僅 10~15 秒)。
   - 支援指令動態切換 `deepseek-r1:1.5b-opt` (數理思考，12 tok/s) 或 `qwen2.5-coder:7b-opt` (深度代碼審查)。
4. **硬體安全防護與白名單 (Hardware & Whitelist Security):**
   - 內建 `ALLOWED_CHAT_IDS` 白名單，拒絕公網未授權人員存取。
   - 即時監控 Pi 5 核心溫度 (`vcgencmd measure_temp`) 與記憶體餘裕。

---

## 🛠️ 安裝與快速啟動 (Quickstart on Raspberry Pi 5)

### 1. 安裝系統相依套件 (System Packages)
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libopus-dev
```

### 2. 安裝 Python 模組
```bash
pip install -r requirements.txt --break-system-packages
```

### 3. 設定環境變數 (`.env`)
在 `telegram_agent/` 目錄建立 `.env` 檔案：
```ini
# 向 Telegram @BotFather 申請的 Token
TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"

# 允許存取的 Telegram User ID 白名單 (多個請用逗號隔開)
ALLOWED_CHAT_IDS="123456789"

# 本機 Ollama 端點
OLLAMA_API_URL="http://127.0.0.1:11434"
```

### 4. 啟動 Agent 服務
```bash
# 前台直接測試運行
python3 agent.py

# 或配置為 24/7 Linux systemd 守護服務
sudo cp pi5-telegram-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pi5-telegram-agent.service
```

---

## 💬 常用 Telegram 互動指令 (Commands)

| 指令 (Command) | 功能描述 (Description) |
| :--- | :--- |
| **直接傳送語音訊息** | 自動語音轉錄 $\to$ 三層記憶檢索 $\to$ AI 思考 $\to$ 語音訊息朗讀回覆。 |
| `/status` | 檢視樹莓派 5 當前 CPU 溫度、可用 RAM 與記憶層級狀態。 |
| `/model <名稱>` | 動態切換大腦模型 (例如 `/model deepseek-r1:1.5b-opt`)。 |
| `/remember <內容>` | 主動將特定記憶或重要筆記寫入 Tier 3 向量記憶庫。 |
| `/facts` | 檢視系統已為您記錄的個人事實特徵 (Tier 2)。 |
| `/clear` | 清空當前短期對話記憶 (Tier 1 Working Buffer)。 |
