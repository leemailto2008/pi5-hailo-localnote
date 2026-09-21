# 規格說明書 (SPEC.md): Raspberry Pi 5 Telegram 雙向語音 AI 助理與三層記憶金字塔
# (Specification: Pi 5 Telegram Voice AI Agent with Three-Tier Memory Hierarchy)

## 1. 系統目標與概述 (System Objectives & Overview)
本規格定義在 **Raspberry Pi 5 (ARM Cortex-A76 四核心 @ 2.40 GHz, 16GB LPDDR4X RAM)** 上構建之 24/7 常駐個人 AI 代理系統：
- **主要互動介面 (Interface):** Telegram 雙向語音與文字互動 (Voice-to-Voice & Text-to-Text)。
- **核心推理引擎 (LLM Engine):** 本地 Native Ollama REST API (`http://127.0.0.1:11434`)，優先調用已調優模型 `qwen2.5:3b-opt` (繁中最佳)、`deepseek-r1:1.5b-opt` (極速思考)。
- **長期記憶系統 (Memory Hierarchy):** 三層記憶金字塔 (Three-Tier Memory Pyramid: 即時緩衝 + 用戶畫像 KV + 本地語意向量庫)。
- **完全隱私與開銷控制 (Privacy & Efficiency):** 語音轉錄、向量化、記憶檢索與推論 100% 在樹莓派本地閉環運作，記憶體常駐開銷控制於 250 MB 以內。

---

## 2. 系統架構與資料流 (System Architecture & Data Flow)

```mermaid
flowchart TD
    User([使用者手機 Telegram]) -->|語音 .oga / 文字| TG[python-telegram-bot 長輪詢閘道]
    
    subgraph Voice Pipeline [雙向語音處理管線]
        TG -->|語音輸入| FFMPEG[ffmpeg 音訊轉碼 16kHz WAV]
        FFMPEG --> STT[faster-whisper 本地轉錄 / Groq 備援]
        STT --> TextPrompt[使用者提問文字]
        TG -->|純文字輸入| TextPrompt
    end

    subgraph Memory Hierarchy [三層記憶金字塔]
        TextPrompt --> Retr[記憶檢索調度器]
        Retr <--> Tier1[Tier 1: 短期對話緩衝記憶體最近 8 輪]
        Retr <--> Tier2[Tier 2: 使用者個人事實特徵 SQLite/JSON]
        Retr <--> Tier3[Tier 3: 語意長期記憶庫 sqlite-vec + nomic-embed-text]
        Retr --> PromptAssembler[動態 Prompt 注入組裝器]
    end

    subgraph Brain [本機推論大腦]
        PromptAssembler --> OllamaAPI[Native Ollama REST API :11434]
        OllamaAPI --> ModelInfer[qwen2.5:3b-opt / deepseek-r1:1.5b-opt]
        ModelInfer --> TextResponse[LLM 生成文字解答]
    end

    subgraph Outbound Voice [語音合成回傳]
        TextResponse --> Choice{使用者是否用語音提問?}
        Choice -->|是| TTS[Piper TTS / Edge-TTS 合成語音 .ogg]
        TTS -->|發送語音訊息| TG
        Choice -->|否 (純文字)| TextOut[發送 Markdown 文字訊息]
        TextOut --> TG
    end

    TG -->|即時語音/文字串流| User
```

---

## 3. 三層記憶金字塔詳細設計 (Three-Tier Memory Hierarchy Specification)

### 3.1 Tier 1: 即時工作對話記憶 (Short-Term Working Memory)
- **存放媒介:** 記憶體中的 Python `collections.deque(maxlen=16)`（保留最近 8 輪 User & Assistant 對話）。
- **重置策略:** 使用者發送指令 `/new` 或 `/clear` 時清空。
- **邊界保護:** 總 Token 數嚴格控制在 1,000 Tokens 以內，避免擠佔 4k Context。

### 3.2 Tier 2: 使用者事實畫像特徵 (User Profile & Fact Key-Value)
- **存放媒介:** 本地 SQLite 資料庫 `telegram_agent/memory.db` 中的 `user_profile` 表格。
- **欄位結構:**
  - `key` (TEXT, PRIMARY KEY): 特徵鍵值（如 `user_name`, `preferred_language`, `tech_stack`, `current_project`）。
  - `value` (TEXT): 特徵內容。
  - `updated_at` (TIMESTAMP): 最後更新時間。
- **常駐注入:** 每次發送給 LLM 的 System Prompt 底部均動態附加：
  ```
  [User Profile Information]:
  - Name: Andrew
  - Preferred Language: Traditional Chinese (繁體中文)
  - Active Project: pi5-llm-arena on Raspberry Pi 5
  ```

### 3.3 Tier 3: 語意長期記憶庫 (Long-Term Episodic Vector Database)
- **存放媒介:** `sqlite-vec` 向量資料表或 SQLite + Cosine Distance 檢索庫。
- **嵌入模型 (Embedding Model):** 本地 Ollama `nomic-embed-text` (768 維度，單次向量化耗時 ~15ms)。
- **記憶寫入機制 (Consolidation):**
  - 當短期對話被移出 Tier 1 緩衝區時，觸發背景非同步 Worker。
  - 提取對話中的關鍵摘要，計算 768 維 Embedding，寫入 `episodic_memories` 表。
- **記憶檢索機制 (Retrieval):**
  - 使用者提問時，即時計算提問的 Embedding 向量。
  - 在 SQLite 執行 KNN 余弦相似度檢索（閾值 > 0.65），擷取 Top-3 相關歷史記憶注入 Prompt。

---

## 4. 語音雙向管線設計 (Voice Pipeline Specification)

### 4.1 語音辨識 (STT)
- **套件:** `faster-whisper` (基於 CTranslate2，支援 ARM64 NEON)。
- **模型:** `base` 或 `small` (量化 `int8`，佔用 RAM < 200MB)。
- **流程:** Telegram 下載 `.oga` $\to$ `ffmpeg` 轉碼為 16kHz 單聲道 WAV $\to$ `faster-whisper.transcribe()` $\to$ 輸出文字。

### 4.2 語音合成 (TTS)
- **引擎選型:**
  - **首選 (高品質免連線):** `edge-tts` (微軟神經語音，繁體中文台灣女聲 `zh-TW-HsiaoChenNeural` / 男聲 `zh-TW-YunJheNeural`)。
  - **本地離線備援 (純離線 C++):** `piper` (`zh_CN-huayan-medium`)。
- **輸出格式:** OGG / OPUS 封裝，發送為 Telegram 原生 Voice Message (語音泡泡)，支援倍速播放。

---

## 5. 安全性與生產環境部署 (Security & Production Daemon)

1. **白名單授權防護 (Access Control):**
   - 伺服器設定 `ALLOWED_TELEGRAM_USER_IDS`（如使用者的個人 Telegram ID）。
   - 未經授權之 Telegram 使用者發起訊息一律直接拒絕並記錄日誌。
2. **24/7 Systemd 守護進程:**
   - 服務檔案: `/etc/systemd/system/pi5-telegram-agent.service`。
   - 支援開機自動啟動、異常崩潰自動重啟 (`Restart=always`, `RestartSec=5s`)。
