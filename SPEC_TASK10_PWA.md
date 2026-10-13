# SPEC_TASK10_PWA: 漸進式 Web 應用程式離線快取規格說明書 (PWA & Service Worker)

> **專案名稱:** Pi 5 Local Note (樹莓派 5 邊緣雙語口譯與語音筆記)  
> **模組目標:** 實作 Task 10 PWA 標準化與 Service Worker 離線快取機制，達成 100% 離網瞬間開啟與手機主畫面安裝 (A2HS)  
> **架構角色:** Senior Staff Software Engineer & Architect  
> **版本:** 1.0.0 (2026-10-13)

---

## 1. 現狀問題與設計背景 (Problem Statement & Context)

1. **離網弱網環境載入風險:**
   - 樹莓派 5 處於戶外獨立熱點 AP 模式 (`10.20.0.1`, SSID: `raspi543_AI`) 時，使用者手機連線至無對外網路 (Captive/Local Network) 環境。
   - 若瀏覽器未建立本地持久化快取，遇到 HTTP 連線延遲或微弱訊號時，頁面可能出現破圖、樣式遺失或白屏。
2. **缺乏原生 App 體驗:**
   - 目前缺少獨立 PWA 資訊清單與規格圖示，無法在 iOS Safari 或 Android Chrome 透過「加入主畫面 (Add to Home Screen, A2HS)」以沉浸式全螢幕 (Standalone) 開啟。

---

## 2. 解決方案架構 (Architecture & Solution)

### 2.1 Web App Manifest (`manifest.json`)
- 提供標準 W3C Web App Manifest，並於 `/manifest.json` 與 `/static/manifest.json` 暴露：
  - `name`: "LocalNote 聽記盒"
  - `short_name`: "LocalNote"
  - `start_url`: "/"
  - `scope`: "/"
  - `display`: "standalone"
  - `orientation`: "portrait"
  - `theme_color`: "#0f766e"
  - `background_color`: "#f3f6f8"
  - `icons`: 包含 192x192 與 512x512 圖示規格，以及 maskable 圖示。

### 2.2 Service Worker 離線快取策略 (`sw.js`)
- **檔案路徑:** `/sw.js` (服務於根路徑，具備全域 scope `/`)。
- **快取名稱:** `localnote-cache-v1`。
- **生命週期 (Lifecycle):**
  1. **`install` 事件:** 預快取關鍵靜態資源（Shell Resources）：
     - `/`
     - `/static/index.html`
     - `/static/manifest.json`
     - `/static/icon-192.png`
     - `/static/icon-512.png`
     - `/static/wifi.png`
     - 呼叫 `self.skipWaiting()` 確保立即就緒。
  2. **`activate` 事件:**
     - 遍歷 `caches.keys()`，自動清除舊版快取。
     - 呼叫 `self.clients.claim()` 立即接管已開啟的客戶端。
  3. **`fetch` 事件路由策略:**
     - **動態 API / 即時通訊 (Network-Only):**
       - 凡路徑包含 `/api/` 或協定為 WebSocket (`/ws/status`)，一律直接轉交網路請求，**絕對不快取**，杜絕資料過期。
     - **靜態資源 (Stale-While-Revalidate 或 Cache-First with Network Fallback):**
       - 先檢索 Cache，命中即刻回傳 (0ms 延遲)；
       - 若快取未命中則發送網路請求，並自動複製 (`response.clone()`) 存入快取供日後離網使用。
       - 即使網路完全中斷，快取依然提供 100% 完整的 UI。

### 2.3 後端路由與 HTTP 標頭配置
- 在 `main.py` 新增路由：
  - `@app.get("/manifest.json")`: 回傳 JSON，附帶正確 MIME 與快取標頭。
  - `@app.get("/sw.js")` / `@app.get("/service-worker.js")`: 回傳 JS，設定重要 HTTP 標頭：
    - `Service-Worker-Allowed: /`
    - `Content-Type: application/javascript; charset=utf-8`
    - `Cache-Control: no-cache, no-store, must-revalidate` (防止瀏覽器快取 SW 檔案本身，確保版本更迭即時生效)。

### 2.4 PWA 高質感圖示產生
- 撰寫 Python 腳本產生簡潔現代、符合深色綠底（`#0f766e`）之高解析度向量/點陣圖示：
  - `icon-192.png` (192x192)
  - `icon-512.png` (512x512)

---

## 3. 驗證清單 (Acceptance Criteria)

- [x] 後端正確提供 `/manifest.json` 與 `/sw.js`，標頭符合 PWA 規範。
- [x] 前端成功註冊 Service Worker，Console 印出 `[PWA] Service Worker registered`。
- [x] Chrome DevTools / 瀏覽器 Application 分頁檢視 Manifest 與 Service Worker 處於 Running 狀態。
- [x] 模擬離線模式 (Offline) 重新整理頁面，UI 完整呈現無任何資源載入失敗。
- [x] 雙端推送至 GitHub 與 GitLab。
