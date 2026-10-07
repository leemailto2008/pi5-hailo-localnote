# 樹莓派 5 (Raspberry Pi 5) 5組按鍵與LED硬體驅動 (Key-LED Hardware Driver)

本驅動模組針對 **Raspberry Pi 5 (RP1 南橋晶片)** 進行最佳化，支援軟體消彈跳 (Software Debouncing)、硬體中斷事件回呼 (Interrupt Callback)、自檢跑馬燈 (Power-On Self-Test) 以及常駐背景守護服務 (Systemd Daemon)。

---

## 1. 硬體接線與腳位對照表 (Hardware Pinout)

| 功能編號 | 40-PIN 實體腳位 | BCM GPIO | 觸發極性 (Polarity) | 偏置電路 / 備註說明 | 對應 LED | 40-PIN 實體腳位 | BCM GPIO | 觸發極性 | 限流電阻 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KEY1** | **Pin 31** | **GPIO 6** | Active Low (低電位有效) | 內部上拉 (Pull-Up) 至 3.3V | **LED1** | **Pin 37** | **GPIO 26** | Active High (高電位點亮) | 串聯 330Ω |
| **KEY2** | **Pin 29** | **GPIO 5** | Active Low (低電位有效) | 內部上拉 (Pull-Up) 至 3.3V | **LED2** | **Pin 36** | **GPIO 16** | Active High (高電位點亮) | 串聯 330Ω |
| **KEY3** | **Pin 15** | **GPIO 22** | Active Low (低電位有效) | 內部上拉 (Pull-Up) 至 3.3V | **LED3** | **Pin 22** | **GPIO 25** | Active High (高電位點亮) | 串聯 330Ω |
| **KEY4** | **Pin 13** | **GPIO 27** | Active Low (低電位有效) | 內部上拉 (Pull-Up) 至 3.3V | **LED4** | **Pin 18** | **GPIO 24** | Active High (高電位點亮) | 串聯 330Ω |
| **KEY5** | **Pin 11** | **GPIO 17** | Active Low (低電位有效) | 內部上拉 (Pull-Up) 至 3.3V | **LED5** | **Pin 16** | **GPIO 23** | Active High (高電位點亮) | 串聯 330Ω |

> **電氣特性說明**：
> - 按鍵端：按下一瞬間接地 (GND)，釋放時由樹莓派內部上拉電阻拉至 3.3V。
> - LED 端：輸出 3.3V 高電位時電流經 330Ω 電阻驅動 LED 點亮；輸出 0V (LOW) 時熄滅。

---

## 2. 驅動運作模式 (Operation Modes)

1. **即按即亮模式 (Direct Follow Mode - 預設)**：
   - 按住 KEY，對應 LED 立即點亮。
   - 放開 KEY，對應 LED 立即熄滅。
2. **開關切換模式 (Toggle / Latch Mode)**：
   - 點按一下 KEY，對應 LED 開啟並保持。
   - 再次點按 KEY，對應 LED 關閉。

---

## 3. 執行與測試指令 (Commands)

### (1) 啟動常駐驅動（即按即亮 - 預設）
```bash
python3 key_led_driver.py --mode follow
```

### (2) 啟動開關切換模式 (Toggle Mode)
```bash
python3 key_led_driver.py --mode toggle
```

### (3) 硬體線路檢測工具 (LED 自測 + 即時鍵盤監視儀)
```bash
python3 test_key_led.py
```

---

## 4. 設定開機自啟動服務 (Systemd Service)

若希望樹莓派開機後自動在背景執行驅動：

```bash
# 1. 複製 service 檔案至 systemd 目錄
sudo cp pi5-gpio-keyled.service /etc/systemd/system/

# 2. 重新載入並啟用開機自啟
sudo systemctl daemon-reload
sudo systemctl enable pi5-gpio-keyled.service
sudo systemctl start pi5-gpio-keyled.service

# 3. 檢查運作狀態
sudo systemctl status pi5-gpio-keyled.service
```
