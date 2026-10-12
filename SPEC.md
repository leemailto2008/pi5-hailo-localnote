# 規格文件 (SPEC.md): Task 31 - LocalNote 聽記盒 Demo 版 API 適配層與狀態機規格
# (Meeting Minutes Engine & Demo UI Backend Adapter)

> **版本 (Version):** 1.1.0  
> **模組路徑 (Module Path):** `code_dispatcher/meeting_minutes.py`, `code_dispatcher/main.py`  
> **建立日期 (Date):** 2026-10-12  
> **架構師 (Architect):** Senior Staff Software Engineer  

---

## 1. 背景與目標 (Background & Objectives)

為滿足「LocalNote 聽記盒」面向客戶與投資人的展示需求，後端需提供一套符合前端 `LocalNote-工程版-index.html` 規格的極簡、高效且符合硬體零爭搶規範的 API 接口層。

### 核心原則：
1. **零爭搶 (Zero-Contention SCO Guard):** 錄音進行中嚴禁推論與高頻 HCI 指令，保障 7.5ms SCO 音訊傳輸不掉幀。
2. **非阻塞離線探測 (Non-blocking Offline Probing):** 後端背景排程非同步 ping 外網，快取離線狀態，避免輪詢 `/api/system` 造成延遲卡頓。
3. **非同步狀態機 (Async State Machine):** 管理 `idle -> recording -> processing (transcribe -> summarize) -> done / error` 生命週期。
4. **結構化紀要解析與降級保護 (Structured Minutes Parser & Graceful Degradation):** 將本地 LLM (`qwen2.5:3b-opt`) 生成的 Markdown 紀要精確拆解為 `summary[]`, `todos[]`, `decisions[]`，同時提供完整 Markdown 文本。

---

## 2. API 介面規格 (API Interface Specifications)

| 方法 | 路徑 | 請求主體 (Payload) | 回傳格式 (Response) | 說明 |
|---|---|---|---|---|
| `GET` | `/api/system` | — | `{offline, cpu, temp, mode, ssid, host, wifi_qr}` | 每 2 秒輪詢；讀取記憶體快取 |
| `GET` | `/api/bt` | — | `{connected, name}` | 每 5 秒輪詢；錄音時讀取快取 |
| `POST` | `/api/bt/{action}` | — | `{ok, msg}` | 支援 `scan`, `connect`, `disconnect` |
| `POST` | `/api/record` | `{seconds: int}` | `{job: str}` | 啟動非同步錄音與後續處理任務 |
| `GET` | `/api/status` | — | `{state, phase, progress, elapsed, total, eta, job, msg}` | 狀態輪詢（1.5s/次） |
| `GET` | `/api/result` | — | `{job, title, started_at, duration, summary[], todos[], decisions[], transcript, markdown}` | 取得最終或階段性產出 |
| `POST` | `/api/regenerate` | — | `{job: str}` | 僅重新執行 LLM 紀要生成，不重錄 |
| `POST` | `/api/translate` | `{text: str, to: "en"\|"zh"}` | `{text: str}` | 紀要雙向翻譯 |
| `POST` | `/api/tts` | `{text: str}` | `{ok: bool}` | 耳麥語音朗讀合成 |
| `POST` | `/api/net/switch` | `{mode: "ap"\|"sta"}` | `{ok: bool}` | 熱點/家用 Wi-Fi 雙向原子切換 |

---

## 3. 狀態機與處理流水線設計 (State Machine Pipeline)

```mermaid
stateDiagram-v2
    [*] --> idle: 系統就緒
    idle --> recording: POST /api/record (時長 60~180s)
    
    state recording {
        [*] --> MicRecording: ffmpeg 收音至 /dev/shm (CPU 獨占)
    }
    
    recording --> processing: 錄音完成
    
    state processing {
        [*] --> transcribe: 釋放 4 核心 faster-whisper (int8)
        transcribe --> summarize: Ollama qwen2.5:3b-opt 生成繁體中文紀要
        summarize --> [*]: Regex 抽取結構化待辦/摘要
    }
    
    processing --> done: 寫入結果並通知前端
    done --> processing: POST /api/regenerate (僅重跑 summarize)
    done --> recording: 再次按開始錄音
    
    recording --> error: 麥克風異常或中斷
    processing --> error: 模型超時或異常
    error --> idle: 使用者重試
```

---

## 4. 實作檢查清單 (Implementation Checklist)

- [x] **Phase 1-A: 撰寫架構規格與驗證避坑條件 (SPEC.md)**
- [x] **Phase 1-B: 實作會議錄音與紀要管理引擎 (`code_dispatcher/meeting_minutes.py`)**
- [x] **Phase 1-C: 在 `code_dispatcher/main.py` 整合並註冊 10 個標準 API 端點**
- [x] **Phase 1-D: 靜態 Wi-Fi QR Code 產生與路由掛載 (`static/wifi.png`)**
- [x] **Phase 1-E: 單元測試與 API 模擬驗證 (`test_meeting_minutes.py`, `test_api_endpoints.py`)**
- [x] **Phase 1-F: 實體機部署驗收與雙遠端推送 (Dual Git Push)**

---

## 5. 規格文件 (Task 32): 獨立藍牙管理頁面 (Bluetooth Device Manager UI View)

### 5.1 需求背景
使用者在 LocalNote 首頁點擊「找耳麥」按鈕時，需平滑切換至專屬的藍牙耳麥管理介面，具備以下能力：
1. **周圍耳麥搜尋 (Scan Nearby Devices):** 呼叫 `/api/bt/scan`，條列可配對之藍牙音訊設備與 RSSI/名稱。
2. **一鍵配對與連線 (Pair & Connect):** 支援 `/api/bt/pair` 與 `/api/bt/connect`，自動信任設備並切換 PipeWire 音訊輸出 (Audio Sink)。
3. **信任設備管理 (Trusted Headsets Hub):** 讀取 `/api/bt/status` 的 `trusted_devices` 清單，支援切換自動連線 (Auto-Connect)、設為預設耳麥 (Set Default) 與中斷連線 (Disconnect)。
4. **無縫返回 (Back to Notes):** 頂部提供導航按鈕，隨時一鍵返回會議聽記主畫面。

### 5.2 實作檢查清單
- [ ] **Task 32-A: 在 `ui_client.html` 構建 SPA 視圖切換器 (Main View vs Bluetooth View)**
- [ ] **Task 32-B: 整合豐富的藍牙操作 API (Scan, Pair, Connect, Disconnect, Auto-connect, Set Default)**
- [ ] **Task 32-C: 在 `main.py` 支援 `/bluetooth` 與 `/bt` 路由直達**
- [ ] **Task 32-D: 實體機部署與驗收驗證**

