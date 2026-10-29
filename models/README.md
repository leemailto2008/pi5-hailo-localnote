# 🧠 樹莓派 5 邊緣 AI 模型架構與配置說明 (Pi 5 Edge AI Models)

本專案 `pi5-hailo-localnote` 採用邊緣端離線／低延遲模型架構：語音轉錄與紀要優先在 Hailo-10H NPU 上執行（CPU 幾乎不占用，藍牙錄音零爭搶），CPU 模型作為備援。適用 Raspberry Pi 5（8 GB／16 GB）＋ Hailo-10H。各元件實際版本見根目錄 `README.md` 的「軟硬體版本」。

---

## 1. 模型清單與規格矩陣 (Models Specification Matrix)

| 類型 | 模型名稱 | 格式 / 引擎 | 量化精度 | 記憶體佔用 | 核心用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **STT（主要）** | Whisper-Small HEF（`/home/pi/models/hailo10h/Whisper-Small.hef`） | HailoRT 5.3.0 Speech2Text（Hailo-10H） | HEF | NPU 約 0.4 GB | 藍牙 16kHz mSBC 語音辨識，`stt_worker` 子行程常駐 |
| **LLM（主要，紀要）** | `qwen3:1.7b` | hailo-ollama 0.5.1（Hailo Model Zoo GenAI 5.3.0） | `Q4_0` HEF | NPU 約 2.9 GB | 會議紀要、分段整理；共用模式下與 Whisper 同時常駐 |
| **STT（CPU 備援）** | `Systran/faster-whisper-small` | faster-whisper 1.2.1 / CTranslate2 4.8.3 | `int8` (NEON) | 約 460 MB | NPU 不可用時的備援 |
| **LLM（CPU 備援／口譯）** | `qwen2.5:3b-opt`（→ `qwen2.5:3b`） | Ollama 0.40.4 (GGUF) | `Q4_K_M` | 約 1.9 GB | NPU 不可用時的備援；中翻英即時口譯 |
| **LLM (代碼審查)** | `qwen2.5-coder:7b` | Ollama (GGUF) | `Q4_K_M` | 約 4.4 GB | 程式碼技術審查與架構分析 (選擇性) |
| **TTS (神經發音)** | `Edge-TTS` (`HsiaoChenNeural`) | edge-tts 7.3.1（Microsoft Edge 線上服務） | 串流 | 0 MB (串流) | 繁中/美語自然語音耳麥喇叭朗讀 |

---

## 2. 快速部署與模型初始化 (Quick Setup)

在樹莓派環境中執行一鍵配置腳本：

```bash
chmod +x models/setup_models.sh
./models/setup_models.sh
```

### 手動客製化自訂模型 (Custom Modelfile)

若欲手動調整口譯溫度 (Temperature) 或系統提示詞 (System Prompt)：

```bash
ollama create qwen2.5:3b-opt -f models/Modelfile.qwen2.5-3b-opt
```

---

## 3. 模型 Manifests 清單 (Model Manifests)

目錄 `models/manifests/` 內包含本系統所支援之邊緣 LLM 官方 Manifest 描述檔 (JSON 格式，保留模型階層結構、層數與雜湊標籤)，方便進行版本鎖定與追蹤。
