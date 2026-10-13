# 規格文件 (SPEC.md): Task 08 - WebSocket 雙向即時狀態推播
# (Real-Time Bidirectional WebSocket Status Push)

> **版本 (Version):** 1.0.0  
> **關聯任務 (Task Reference):** [todo_tasks.md](file:///f:/13_prj_raspi5_localnote/todo_tasks.md) Task 08  
> **模組路徑 (Module Path):** `code_dispatcher/ws_manager.py`, `code_dispatcher/main.py`, `code_dispatcher/meeting_minutes.py`, `code_dispatcher/static/index.html`  
> **建立日期 (Date):** 2026-10-13  
> **架構師 (Architect):** Senior Staff Software Engineer & Architect  

---

## 1. 背景與現狀問題 (Background & Problem Analysis)

目前 LocalNote 前端儀表板高度依賴多個定時器發起 HTTP 短輪詢 (HTTP Short-Polling)：
1. 每 1.5s ~ 4.5s 輪詢 `/api/status`（錄音與整理進度）。
2. 每 2.0s 輪詢 `/api/system`（CPU 負載、SoC 核心溫度、網路狀態）。
3. 每 5.0s 輪詢 `/api/bt`（藍牙耳麥連線狀態）。
4. 在藍牙中心 (Bluetooth Hub) 開啟時每 2.5s 輪詢 `/api/bt/status`。

### 存在缺陷：
- **資源浪費與總線壓力：** 每個 HTTP 請求皆需 TCP 握手/排程，在樹莓派 5 邊緣運算環境下佔用 CPU 與網路。
- **日誌洗版 (Log Spamming)：** 系統日誌充斥每秒數行的 `GET /api/system 200 OK`，妨礙問題排查。
- **狀態更新延遲 (Latency)：** 狀態變化（如錄音完成、耳麥斷線）需等待下一個輪詢週期 (1.5~5 秒) 才能呈現在畫面上。

---

## 2. 改良目標與效益 (Objectives & Benefits)

1. **零延遲狀態推播 (Zero-Latency Event Push)：** 錄音狀態機流轉或藍牙狀態改變時，後端主動推播至前端，體感延遲由 2 秒降至 <50ms。
2. **終結高頻 HTTP 輪詢 (Eliminate Polling Overhead)：** 當 WebSocket 建立成功時，暫停定時 HTTP 輪詢，僅保留 WebSocket 心跳與推播。
3. **優雅降級 (Graceful Degradation)：** 若網路環境或代理伺服器阻擋 WebSocket，自動無縫降級回既有 HTTP 短輪詢，確保 100% 可用性。
4. **雙向通訊 (Bidirectional)：** 支援前端發送 `ping` 保活與 `refresh` 手動請求立即狀態回傳。

---

## 3. WebSocket 通訊協定規格 (WebSocket Protocol Schema)

### 連線端點 (Endpoint):
`ws://<host>:<port>/ws/status`

### 後端 ➔ 前端推播訊息格式 (Server-to-Client Messages):

```jsonc
// 1. 系統硬體與網路狀態推播 (週期性 1.5s 或事件驅動)
{
  "type": "system",
  "data": {
    "offline": false,
    "cpu": 3.5,
    "temp": 46.8,
    "mode": "sta",
    "ssid": "raspi543_AI",
    "host": "localnote.local"
  }
}

// 2. 管線進度與狀態機推播 (事件驅動，狀態改變時立即發送)
{
  "type": "status",
  "data": {
    "state": "processing",       // idle / recording / processing / done / error
    "phase": "transcribe",       // record / transcribe / summarize
    "progress": 45,
    "elapsed": 60.0,
    "total": 60.0,
    "eta": 25.0,
    "job": "m_1791857536",
    "msg": "正在進行語音聽寫..."
  }
}

// 3. 藍牙耳麥基礎狀態推播 (事件驅動)
{
  "type": "bt",
  "data": {
    "connected": true,
    "name": "RichChipAN"
  }
}

// 4. 藍牙管理中心詳細狀態推播 (若前端訂閱藍牙中心檢視)
{
  "type": "bt_hub",
  "data": {
    "connected": true,
    "connected_name": "RichChipAN",
    "connected_mac": "A0:D1:2A:42:12:34",
    "battery_level": 90,
    "signal_level_pct": 95,
    "rssi": -48,
    "trusted_devices": [...]
  }
}
```

### 前端 ➔ 後端指令格式 (Client-to-Server Messages):

```jsonc
// 1. 心跳保活
{"action": "ping"} // 後端回覆 {"type": "pong", "ts": 1791859999}

// 2. 請求立即刷新全量狀態
{"action": "refresh"}

// 3. 訂閱/取消訂閱藍牙中心高頻詳細狀態
{"action": "subscribe_bt_hub", "enable": true}
```

---

## 4. 架構設計與實作步驟 (Implementation Architecture)

1. **新增 `code_dispatcher/ws_manager.py`:**
   - 實裝 `ConnectionManager` 類別，負責連線維護、廣播、例外過濾。
   - 封裝 `broadcast_event(event_type: str, data: dict)` 供各模組非同步調用。
2. **整合後端狀態機 (`code_dispatcher/meeting_minutes.py`):**
   - 在 `status` 更新、階段切換時註冊回調 (Callback)，主動觸發 `broadcast_event("status", status_data)`。
3. **新增主路由端點 (`code_dispatcher/main.py`):**
   - 註冊 `@app.websocket("/ws/status")`。
   - 啟動背景非同步推播協程 (Background Telemetry Task)，以 1.5 秒週期推播最新 `system` 負載指標，取代前端高頻 HTTP 請求。
4. **前端 WebSocket 客戶端實裝 (`code_dispatcher/static/index.html` & `ui_client.html`):**
   - 實裝 `initWebSocket()`，自動連線 `/ws/status`。
   - 支援斷線指數退避自動重連 (Exponential Backoff: 1s, 2s, 4s, 8s...)。
   - 連線建立後暫停原本高頻 HTTP 短輪詢；若斷線則無縫啟動 HTTP 短輪詢降級。
   - 收到 `status`、`system`、`bt` 訊息時立即更新對應 DOM 元素。
5. **部署至樹莓派 5 驗證並雙端推送:**
   - 驗證 WebSocket 連線與雙向推播。
   - 驗證錄音、狀態流轉時的秒級零延遲反應。
   - 驗證系統日誌不再被 HTTP 輪詢洗版。
