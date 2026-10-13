# SPEC_TASK09_VUMETER: 即時音量頻譜採樣規格說明書 (Live VU Meter & Spectrum Visualizer)

> **專案名稱:** Pi 5 Local Note (樹莓派 5 邊緣雙語口譯與語音筆記)  
> **模組目標:** 實作 Task 09 即時音量頻譜採樣，杜絕無效靜音錄音，提供 50ms 實時視覺回饋  
> **架構角色:** Senior Staff Software Engineer & Architect  
> **版本:** 1.0.0 (2026-10-13)

---

## 1. 現狀問題與設計背景 (Problem Statement & Context)

在藍牙耳麥錄音情境中，使用者經常遇到以下痛點：
1. **黑箱錄音 (Blind Recording):** 使用者按下錄音後，無法即時得知藍牙耳麥麥克風是否有開啟、是否處於靜音或物理開關未開。
2. **重錄成本高:** 往往等待整整 60~90 秒錄音結束、送入 Whisper 失敗報錯後，才發現完全沒有收到聲音。
3. **樹莓派 5 效能與 UART 總線限制 (Zero-Contention Rule):**
   - 樹莓派 5 藍牙共用 `/dev/ttyAMA0` 硬體 UART 總線。
   - 嚴禁額外啟動第二個獨立的 PipeWire 錄音程序或高頻輪詢 HCI 設備，否則會引發 SCO 丟包與斷流。

---

## 2. 解決方案架構 (Architecture & Solution)

### 2.1 單一 ffmpeg 進程雙路輸出 (Dual-Output Zero-Contention Pipeline)
不建立第二個錄音程序，而是在原本的 `record_audio_from_mic` ffmpeg 指令中，配置一進雙出：
1. **輸出 1 (File Out):** `-af "aresample=16000:async=1,highpass=f=80,lowpass=f=7600" -c:a libopus -b:a 64k <output_path>`（高音質語音母帶，掛載於 `/dev/shm/audio_cache` 零磁碟 I/O）。
2. **輸出 2 (Stdout Pipe Out):** `-f s16le -ac 1 -ar 16000 -c:a pcm_s16le pipe:1`（未經二次編碼的純 PCM 原始音訊串流）。

### 2.2 50ms 採樣視窗與特徵萃取 (Feature Extraction)
Python 異步協程從 `proc.stdout` 每次讀取 1600 位元組（50ms 視窗，800 個 16-bit 採樣點）：
1. **均方根音量 (RMS Amplitude):** 計算電平有效值，正規化為 0.0 ~ 1.0，並換算為 dBFS。
2. **峰值 (Peak Amplitude):** 擷取 50ms 視窗內的最大絕對值。
3. **8 頻段頻譜 (8-Band Spectrum):** 透過 `np.fft.rfft` 進行快速傅立葉變換，劃分為 8 個語音頻帶（80Hz ~ 7.6kHz），換算為柱狀高度 (0~100%)。
4. **即時廣播 (WebSocket Broadcast):** 透過 `ws_manager.broadcast("vu", vu_data)` 推送至前端。

### 2.3 前端動態 VU 表與 8-Band 頻譜 (Frontend Visualizer)
1. 在錄音進度卡片內嵌動態視覺化容器：
   - **VU Meter 漸層音量電平條:** 綠色 (正常) ➔ 黃色 (警告) ➔ 紅色 (過載/削頂)，並具備動態 Peak Hold 標記。
   - **語音活躍指示徽章 (VAD Badge):** 「收音良好 🟢」/「音量微弱 🟡」/「未偵測到聲音 🔴」。
   - **8-Band 動態頻譜條 (Equalizer Bars):** 8 根發光霓虹漸層柱狀條，根據頻帶能量即時跳動。
2. **動畫效能最佳化:** 透過 `requestAnimationFrame` 與 CSS transform 驅動，不產生任何重排 (Reflow)。

---

## 3. 介面與資料格式 (Data Contract)

```json
{
  "type": "vu",
  "data": {
    "rms": 0.185,
    "peak": 0.420,
    "db": -14.6,
    "level_pct": 52,
    "status": "active",
    "bands": [35, 68, 85, 72, 40, 28, 15, 10],
    "active": true
  }
}
```

---

## 4. 驗證清單 (Acceptance Checklist)
- [x] 後端 ffmpeg 雙輸出測試無死鎖且 OGG 母帶正常生成。
- [x] 50ms 週期 WebSocket 穩定推播 `vu` 數據。
- [x] 錄音結束後音量計與頻譜平滑歸零並收合。
- [x] 實機部署至樹莓派 5 驗證。
- [x] Git dual push 至 GitHub 與 GitLab。
