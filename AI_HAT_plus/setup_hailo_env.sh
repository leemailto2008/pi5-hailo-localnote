#!/usr/bin/env bash
# ==============================================================================
# AI_HAT_plus/setup_hailo_env.sh
# 一鍵配置樹莓派 5 PCIe Gen 3 與 Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L) 環境
# ==============================================================================
set -euo pipefail

echo "=========================================================================="
echo " 🚀 開始配置 Raspberry Pi 5 + AI HAT+ (Hailo-8 / 26 TOPS) 驅動與 PCIe Gen 3"
echo "=========================================================================="

# 1. 檢查硬體是否為 Raspberry Pi 5
if ! grep -q "Raspberry Pi 5" /proc/cpuinfo 2>/dev/null; then
    echo "⚠️ 警告：當前系統可能非 Raspberry Pi 5，請確認硬體平台。"
fi

# 2. 自動在 /boot/firmware/config.txt 啟用 PCIe Gen 3 模式
CONFIG_FILE="/boot/firmware/config.txt"
echo "--- [步驟 1/4] 配置 PCIe Gen 3 匯流排模式 ---"
if [ -f "$CONFIG_FILE" ]; then
    if ! grep -q "^dtparam=pciex1" "$CONFIG_FILE"; then
        echo "啟用 dtparam=pciex1..."
        echo "dtparam=pciex1" | sudo tee -a "$CONFIG_FILE"
    fi
    if ! grep -q "^dtparam=pciex1_gen=3" "$CONFIG_FILE"; then
        echo "啟用 PCIe 3.0 高速模式 (dtparam=pciex1_gen=3)..."
        echo "dtparam=pciex1_gen=3" | sudo tee -a "$CONFIG_FILE"
    fi
    echo "✅ PCIe 3.0 配置已寫入 $CONFIG_FILE。"
else
    echo "❌ 找不到 $CONFIG_FILE，請確認 Debian 13 / Raspberry Pi OS 檔案路徑。"
fi

# 3. 更新 apt 來源並安裝官方 hailo-all 套件
echo "--- [步驟 2/4] 安裝 Raspberry Pi 官方 Hailo 工具鏈與驅動 ---"
sudo apt-get update -y
sudo apt-get install -y hailo-all python3-hailort dkms pciutils

# 4. 驗證 PCIe 設備枚舉與 HailoRT 韌體狀態
echo "--- [步驟 3/4] 探測 PCIe 匯流排上的 Hailo 晶片 ---"
if lspci | grep -i "Hailo"; then
    echo "✅ 成功探測到 PCIe Hailo AI 加速器！"
    lspci -nnk -d 1e60:
else
    echo "⚠️ 尚未在 PCIe 匯流排上探測到 Hailo 晶片。若您剛插入硬體，請重啟系統以使 PCIe 生效。"
fi

# 5. 檢查 hailortcli 工具與 Python 綁定
echo "--- [步驟 4/4] 驗證 HailoRT 執行時期狀態 ---"
if command -v hailortcli &>/dev/null; then
    echo "HailoRT CLI 版本:"
    hailortcli --version || true
    echo "正在探測硬體裝置:"
    hailortcli scan || true
else
    echo "⚠️ hailortcli 尚未就緒，請檢查套件安裝日誌。"
fi

echo "=========================================================================="
echo " 🎉 AI HAT+ 基礎環境配置腳本執行完成！"
echo " 💡 提示：若剛剛首次啟用 PCIe Gen 3，強烈建議執行 'sudo reboot' 重啟樹莓派。"
echo "=========================================================================="
