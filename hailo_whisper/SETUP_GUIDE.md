# 🚀 Hailo-10H Whisper NPU (40 TOPS) 部署與設定指引 (SETUP_GUIDE.md)

本文件說明當樹莓派 5 接入 Wi-Fi 網路後，如何在系統端快速驗證 Hailo-10H 硬體、下載官方 Whisper HEF 模型，並啟動全 NPU 加速語音轉錄引擎。

---

## 1. 硬體探測與驅動驗證 (Hardware Verification)

登入樹莓派 5 終端機，執行以下診斷命令：

```bash
# 1. 檢查 PCIe 匯流排裝置識別碼 (確認 Hailo-10H M.2 模組已被列舉)
lspci -nn | grep -i hailo

# 預期輸出類似：
# 0000:01:00.0 Co-processor [0b40]: Hailo Technologies Ltd. Hailo-10H AI Processor [1e60:2864]

# 2. 檢查 HailoRT 驅動狀態與韌體控制
hailortcli fw-control identify

# 3. 檢查 PCIe Gen3 協商頻寬 (應為 8.0 GT/s, x1 鏈結)
sudo lspci -vv -s 01:00.0 | grep -E "LnkCap|LnkSta"
```

---

## 2. 下載官方 Whisper-Small / Base HEF 模型 (Model Download)

Hailo-10H 原生具備 8GB LPDDR4X 專屬記憶體，支援完整 Encoder + Decoder 端到端自回歸推論。

```bash
# 建立模型快取目錄
mkdir -p /home/pi/models/hailo10h/

# 下載官方 Hailo-10H Whisper-Small (244M) 預編譯 HEF
# (亦可選用 whisper_base_h10h.hef 以獲得更高 TPS)
cd /home/pi/models/hailo10h/

# 下載命令 (透過 Hailo Model Zoo 或專屬鏡像)
wget -O whisper_small_h10h.hef "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/GenAI/whisper_small_h10h.hef" \
  || echo "若 AWS 需驗證，請參閱 Hailo Developer Zone 取得直接下載連結。"
```

---

## 3. 本地推論與基準測試 (Run Local Benchmark)

在樹莓派上執行效能評測工具：

```bash
cd /home/pi/code_dispatcher
python3 -m hailo_whisper.benchmark
```

**預期效能指標：**
- **Log-Mel 萃取延遲 (Pure NumPy):** ~15ms
- **Whisper-Small 6 秒切片推論延遲:** 300ms ~ 500ms
- **實時因子 (RTF):** `0.05 ~ 0.08` (比 CPU 快 8~12 倍)
- **CPU 核心佔用率:** `< 5%` (徹底解放 Cortex-A76 四核心)

---

## 4. 自動退避保護與架構相容性 (Graceful Fallback)

本專案實作之 `hailo_whisper.engine.get_whisper_engine()` 具備智慧自動探測：
- **Hailo-10H 正常且 HEF 存在：** 啟用 40 TOPS NPU 全硬體推論。
- **Hailo-10H 離線或 HEF 缺漏：** 自動零中斷平滑切換至 CPU `faster-whisper`，保證 100% 聽記可用性。
