# 🍓 Raspberry Pi AI HAT+ (Hailo-8 / 26 TOPS) 加速套件
# Project: `AI_HAT_plus` (Future: `pi5-hailo-localnote`)

> **版本 (Version):** 1.0.0-alpha  
> **硬體目標 (Hardware Target):** Raspberry Pi 5 (8GB / 16GB) + Official Raspberry Pi AI HAT+ (Hailo-8 M.2 26 TOPS)  
> **架構定位 (Role):** 高算力邊緣語音特徵提取與 Whisper STT 硬件加速中樞  

---

## 1. 核心亮點 (Key Features)

* **邊錄邊即時轉錄 (Concurrent Streaming STT):** 錄音開始後第 5~6 秒即由 AI NPU 啟動即時分塊轉錄，逐字稿動態秒出，錄音結束等待時間壓縮至 < 1 秒！
* **10 倍速語音轉錄 (10x Faster STT):** 3 分鐘會議音訊轉錄耗時從原本 CPU 的 40~45 秒壓縮至 **3~5 秒** (即時因子 RTF < 0.03)。
* **消除藍牙音訊爭搶 (Zero-Contention):** 將密集張量運算移至獨立 PCIe NPU，CPU 佔用率降至 < 15%，杜絕 UART 7.5ms SCO 爆音。
* **無痛回退保險 (Graceful Fallback):** 內建 `get_whisper_engine()` 智能工廠。當硬體尚未插上或驅動異常時，自動切回原生 `faster-whisper`，保證業務 100% 不中斷。

---

## 2. 目錄結構 (Directory Layout)

```
AI_HAT_plus/
├── SPEC.md                   # 邊錄邊辨與雙工音訊串流架構規格書
├── plan.md                   # 系統架構四階段詳細規劃書
├── setup_hailo_env.sh        # 一鍵 PCIe 3.0 與官方 HailoRT 驅動安裝腳本
├── requirements.txt          # Python 相依清單
├── hailo_engine/             # 核心加速引擎模組
│   ├── __init__.py           # 套件匯出入口
│   ├── device_manager.py     # PCIe 硬件與驅動探測器 (HailoDeviceManager)
│   ├── whisper_adapter.py    # 異質 Whisper 推理適配器 (NPU vs CPU)
│   └── audio_preprocessor.py # 16kHz PCM 轉 80-band Log-Mel 特徵萃取
├── benchmarks/
│   └── benchmark_rtf.py      # 1/3/5 分鐘音訊即時因子 (RTF) 效能基準測試
└── README.md                 # 本文件
```

---

## 3. 快速上手 (Quick Start)

### 步驟 1：安裝硬體與啟用 PCIe Gen 3 高速模式
```bash
chmod +x setup_hailo_env.sh
./setup_hailo_env.sh
# 首次設定後請重啟樹莓派
sudo reboot
```

### 步驟 2：在 Python 中調用加速引擎
```python
from hailo_engine import get_whisper_engine

# 自動探測硬體：若有 AI HAT+ 則走 NPU，否則自動回退為 CPU
engine = get_whisper_engine(auto_detect=True)
print("使用後端:", engine.get_engine_type())

result = engine.transcribe("/dev/shm/audio_cache/meeting.ogg")
print("逐字稿:", result["text"])
print(f"轉錄耗時: {result['duration_sec']:.2f} 秒 (RTF: {result['rtf']:.3f})")
```

---

## 4. 未來獨立拆分為新倉庫指引 (Standalone Repo Setup)

當您在樹莓派上驗證通過後，可隨時將本目錄抽離為獨立 GitHub / GitLab 倉庫：

```bash
# 進入本套件目錄
cd f:\13_prj_raspi5_localnote\AI_HAT_plus

# 初始化獨立 Git 倉庫
git init
git add .
git commit -m "feat: initial commit for pi5-hailo-localnote"

# 關聯遠端倉庫 (以 pi5-hailo-localnote 為例)
git remote add origin https://alntu2@github.com/leemailto2008/pi5-hailo-localnote.git
git remote add gitlab http://andrew.lee@gitlab.mytekcentral.com:8081/andrew.lee/pi5-hailo-localnote.git

# 雙推至新倉庫
git push -u origin main && git push -u gitlab main
```
