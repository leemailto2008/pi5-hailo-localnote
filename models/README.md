# 🧠 樹莓派 5 邊緣 AI 模型架構與配置說明 (Pi 5 Edge AI Models)

本專案 `pi5-llm-localnote` 採用三位一體的邊緣端離線/低延遲神經網絡模型架構，專為 Raspberry Pi 5 (8GB) 的 ARM Cortex-A76 NEON 指令集與記憶體頻寬深度最佳化：

---

## 1. 模型清單與規格矩陣 (Models Specification Matrix)

| 類型 | 模型名稱 | 格式 / 引擎 | 量化精度 | 記憶體佔用 | 核心用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **STT (語音轉錄)** | `Systran/faster-whisper-small` | CTranslate2 / OpenMP | `int8` (NEON) | 約 460 MB | 藍牙 16kHz mSBC 繁體中文高精度語音辨識 |
| **LLM (雙語口譯)** | `qwen2.5:3b-opt` | Ollama (GGUF) | `Q4_K_M` | 約 1.9 GB | 中翻英即時道地口譯，首字延遲 TTFT < 0.2s |
| **LLM (代碼審查)** | `qwen2.5-coder:7b` | Ollama (GGUF) | `Q4_K_M` | 約 4.4 GB | 程式碼技術審查與架構分析 (選擇性) |
| **TTS (神經發音)** | `Edge-TTS` (`HsiaoChenNeural`) | Microsoft Azure Edge | WebRTC Stream | 0 MB (串流) | 繁中/美語自然語音耳麥喇叭朗讀 |

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
