# 🎙️ Pi 5 Edge AI Local Note & Bluetooth Voice Gateway
### 樹莓派 5 邊緣雙語口譯、語音筆記與藍牙智慧網關系統

[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205%208GB%20%7C%2016GB-red.svg)](https://www.raspberrypi.com/)
[![NPU](https://img.shields.io/badge/NPU-Hailo--10H%2040%20TOPS-success.svg)](https://hailo.ai/)
[![OS](https://img.shields.io/badge/OS-Debian%2013.7%20Trixie%20%7C%20kernel%206.18.50-blue.svg)](https://www.debian.org/)
[![Audio](https://img.shields.io/badge/Audio-PipeWire%201.4.2%20mSBC%2016kHz-green.svg)](https://pipewire.org/)
[![HailoRT](https://img.shields.io/badge/HailoRT-5.3.0-orange.svg)](https://hailo.ai/developer-zone/)
[![STT](https://img.shields.io/badge/STT-Hailo%20Whisper--Small-orange.svg)](https://hailo.ai/)
[![LLM](https://img.shields.io/badge/LLM-hailo--ollama%200.5.1%20qwen3%3A1.7b-purple.svg)](https://hailo.ai/)

---

## 一、 專案架構概覽 (Architecture Overview)

`pi5-hailo-localnote` 是一套專為樹莓派 5 (Raspberry Pi 5) 與 Hailo-10H NPU (40 TOPS) 打造的低延遲、高隱私邊緣 AI 語音處理中樞。整合藍牙耳麥硬體串流、本機端硬體加速語音辨識 (STT)、即時大語言模型口譯 (LLM Streaming) 以及獨立離網熱點 (AP Mode)：

### 系統架構與資料流拓撲圖 (System Architecture & Datapath Topology)

<p align="center">
  <img src="./ARCHITECTURE_AND_DATAPATH_20261019.jpg" alt="系統架構與資料流拓撲圖 (System Architecture & Datapath Topology)" width="100%">
</p>

```mermaid
graph TD
    BT[🎧 Bluetooth Headset mSBC] -->|PipeWire 16kHz| CAP[🎙️ Zero-Disk /dev/shm Buffer]
    CAP -->|Zero-Contention| STT[⚡ Hailo-10H NPU Whisper-Small 40 TOPS]
    STT -->|SSE Stream / WebSocket| LLM[🌐 hailo-ollama qwen3:1.7b on Hailo-10H / CPU 備援 Ollama qwen2.5:3b-opt]
    LLM -->|Typewriter SSE| WEB[📱 Mobile Responsive Web UI]
    LLM -->|Dual-Audio Playback| TTS[🔊 Edge-TTS Neural Voice]
    KEY[🔘 GPIO Physical Key] -->|Hardware Event| CAP
    WIFI[📡 nmcli Hotspot & nftables] -->|Fail-Safe Guard| WEB
```

---

## 二、 核心特性 (Key Features)

### 1. 純淨音訊捕捉與抗爆音隔離 (Zero-Contention Audio Isolation)
- **硬體 UART 總線隔離：** 錄音期間完全靜默 `hcitool` HCI 輪詢與神經網絡推論，消除 `spa.bluez5.source.sco` 丟幀問題。
- **瞬態平滑濾波 (Bandpass Filter)：** 80Hz ~ 7.6kHz 帶通抑制，杜絕連線階躍爆音。
- **零磁碟寫入 (0 Flash Wear)：** 全流程於 Linux POSIX 記憶體磁碟 (`/dev/shm/audio_cache`) 完成，保護 MicroSD / NVMe 壽命。

### 2. 即時串流與分塊口譯 (Streaming STT & Speculative LLM)
- **即時分塊轉錄：** 錄音中約每 8 秒（在 4～9.5 秒間找停頓切段）送 Hailo-10H Whisper，並行推播轉錄進度，尾段辨識無感知。
- **NPU 共用模式：** Whisper 與 qwen3 同時常駐 Hailo-10H（hailo-ollama `HAILO_OLLAMA_VDEVICE_GROUP_ID`），STT 與 LLM 不必輪流開關 NPU；閒置時若 NPU 記憶體異常偏高會自動重置（`code_dispatcher/npu_share.py`）。
- **即時道地英文口譯：** 首字延遲 (TTFT) 僅 0.18 秒，提供即時 Token 速率 (TPS) 統計與打字機動態效果。
- **手機端流暢體驗：** 轉錄與口譯區塊支援現代化捲軸 (Scrollbar) 與自動平滑追隨滾動。

### 3. 多耳麥智慧管理與折疊控制台 (Smart Bluetooth Hub)
- **受信任耳麥清單 (Trusted Devices)：** 支援個別切換「自動連線 (Auto-Connect)」開關，多設備環境自由切換。
- **一鍵折疊 (Collapsible View)：** 平時高度僅 48px，保留名稱、連線狀態與 🔋 電量膠囊；展開可進行完整掃描與配對。

### 4. 戶外/飛航獨立熱點 (Offline Edge Hotspot)
- **方案 A 獨立熱點 (`raspi543_AI`)：** 搭配 `nftables` 封包隔離與看門狗防護 (Fail-Safe Watchdog)，離網環境下手機直連 10.20.0.1 即可操作。

---

## 三、 軟硬體版本 (Versions)

兩台實機（raspi678、raspi543）於 2026-10-28 對齊、10-29 更新後的實際版本。完整套件清單與升級評估見各機 `/home/pi/ln_e2e/versions_20261028/`。

| 類別 | 元件 | 版本 |
|---|---|---|
| 硬體 | Raspberry Pi 5 | raspi678 8 GB、raspi543 16 GB |
| | Hailo-10H（M.2，PCIe Gen3 x1，裝置記憶體 8 GB） | 韌體 5.3.0 |
| | 開機韌體 (bootloader EEPROM) | 2026-09-25（rpi-eeprom 28.34-1） |
| 作業系統 | Debian 13 Trixie | 13.7，kernel 6.18.50-1+rpt1（`linux-image-rpi-2712`） |
| Hailo 軟體 | HailoRT／PCIe 驅動 (`hailort`、`hailort-pcie-driver`) | 5.3.0 |
| | HailoRT Python 綁定 (`hailort` wheel，cp313) | 5.3.0 |
| | Hailo Model Zoo GenAI (`hailo-gen-ai-model-zoo`，含 hailo-ollama) | 5.3.0（hailo-ollama 0.5.1） |
| AI 模型 | STT：Whisper-Small HEF | `/home/pi/models/hailo10h/Whisper-Small.hef`；解碼：自動語言偵測、repetition_penalty 1.1（`PI5_WHISPER_LANGUAGE`／`PI5_WHISPER_REPETITION_PENALTY`） |
| | LLM（NPU）：`qwen3:1.7b` | Q4_0，2.9 GB，hailo-ollama `127.0.0.1:11435` |
| | LLM（CPU 備援）：`qwen2.5:3b-opt`（→ `qwen2.5:3b`） | Ollama 0.40.4，`127.0.0.1:11434` |
| | STT（CPU 備援）：faster-whisper | 1.2.1（ctranslate2 4.8.3） |
| 音訊／藍牙 | PipeWire | 1.4.2-1+rpt3 |
| | WirePlumber | 0.5.8-2 |
| | BlueZ | 5.82-1.1+rpt2 |
| | FFmpeg | 7.1.5（8:7.1.5-0+deb13u1+rpt2） |
| Python | CPython | 3.13.5 |
| | fastapi／starlette／uvicorn | 0.115.11／0.46.2／0.32.0 |
| | websockets／httpx／requests／pydantic | 17.2／0.28.1／2.34.2／2.10.6 |
| | numpy／onnxruntime／av | 2.5.4／1.31.0／19.0.1 |
| | OpenCC／edge-tts／gTTS／psutil | 1.1.9／7.3.1／2.5.4／7.0.0 |

- Hailo 元件（HailoRT、PCIe 驅動、晶片韌體、Model Zoo GenAI 的 HEF）必須同一個版號一起升級。Developer Zone 目前最新為 5.4.0（2026-08-16）；評估後暫不升級。
- Python 套件裝在 `/usr/local`（pip，`--break-system-packages`）；`hailort` wheel 只在 Hailo Developer Zone 提供，不在 PyPI。

---

## 四、 目錄結構 (Directory Structure)

```
pi5-hailo-localnote/
├── code_dispatcher/        # FastAPI 後端、語音串流核心與響應式 Web 控制台
│   ├── main.py             # 核心 API 端點與藍牙裝置管理
│   ├── voice_streaming.py  # 串流切片錄音、純淨音訊捕捉與全軌 STT
│   ├── voice_memory.py     # POSIX 記憶體磁碟環形緩衝區管理
│   ├── ui.html             # 繁體中文深色現代化響應式 Web UI
│   └── hotspot_watchdog.py # 熱點自動防呆復原看門狗
├── gpio_driver/            # 實體按鍵與雙色 LED 狀態連動驅動程式
│   └── key_led_driver.py   # GPIO 錄音觸發與狀態指示燈
├── telegram_agent/         # Telegram 智慧語音助理與自我進化引擎
│   ├── agent.py            # Raspi543bot 核心代理
│   ├── bluetooth_manager.py# 無螢幕 Headless 藍牙配對與連線守護行程
│   └── evolution_engine.py # 自我進化與排程追蹤引擎
├── models/                 # AI 模型設定、Modelfile 與自動化安裝腳本
│   ├── Modelfile.qwen2.5-3b-opt # 即時雙語口譯專用模型定義
│   ├── setup_models.sh     # 一鍵預熱與拉取腳本
│   └── manifests/          # 模型規格描述檔清單
├── hotspot.md              # 戶外熱點與 nftables 防火牆設定指引
├── SPEC.md                 # 系統工程規格文件
└── todo_tasks.md           # 30 項模組改良藍圖與進度追蹤
```

---

## 五、 快速啟動 (Quick Start)

### 1. 安裝 Python 依賴
```bash
pip install -r requirements.txt
```

### 2. 預熱與配置 AI 模型
```bash
chmod +x models/setup_models.sh
./models/setup_models.sh
```

### 3. 啟動 Web 服務
實機以 systemd 服務 `pi5-dispatcher-api.service` 執行（Uvicorn 監聽 port 80）：
```bash
sudo systemctl restart pi5-dispatcher-api.service
# 手動執行（開發用）：
cd code_dispatcher
python3 -m uvicorn main:app --host 0.0.0.0 --port 80
```
瀏覽器開啟：`http://<樹莓派IP>/` 或熱點模式 `http://10.20.0.1/`。

### 4. Hailo-10H NPU 共用模式（建議）
```bash
sudo install -D -m 644 deploy/systemd/hailo-ollama.service.d-10-shared-vdevice.conf \
    /etc/systemd/system/hailo-ollama.service.d/10-shared-vdevice.conf
sudo systemctl daemon-reload && sudo systemctl restart hailo-ollama.service pi5-dispatcher-api.service
```
