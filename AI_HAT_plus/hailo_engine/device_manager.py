# -*- coding: utf-8 -*-
"""
AI_HAT_plus/hailo_engine/device_manager.py
Raspberry Pi 5 PCIe Hailo NPU 硬體探測與資源管理器
"""

import os
import subprocess
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("hailo-device-manager")

class HailoDeviceManager:
    """Hailo 硬件生命週期與 PCIe 狀態探測器"""
    _instance: Optional["HailoDeviceManager"] = None
    _is_available: Optional[bool] = None
    _device_info: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._probe_hardware()
        return cls._instance

    def _probe_hardware(self):
        """主動探測 PCIe 匯流排與 HailoRT 驅動狀態"""
        self._is_available = False
        self._device_info = {
            "device_type": "None",
            "pcie_detected": False,
            "driver_loaded": False,
            "device_id": None
        }

        # 1. 檢查 PCIe 設備節點
        try:
            res = subprocess.run(["lspci"], capture_output=True, text=True, timeout=3)
            for line in res.stdout.splitlines():
                if "1e60:" in line or "Hailo" in line.capitalize():
                    self._device_info["pcie_detected"] = True
                    if "Hailo-8" in line:
                        self._device_info["device_type"] = "Hailo-8"
                    elif "Hailo-10" in line:
                        self._device_info["device_info"] = "Hailo-10H"
                    else:
                        self._device_info["device_type"] = "Hailo-PCIe"
                    break
        except Exception as e:
            logger.debug(f"lspci probe failed: {e}")

        # 2. 檢查 hailortcli 狀態
        try:
            res = subprocess.run(["hailortcli", "scan"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0 and "Hailo" in res.stdout:
                self._device_info["driver_loaded"] = True
                self._is_available = True
                logger.info(f"✅ Hailo NPU 成功就緒: {self._device_info['device_type']}")
        except Exception:
            pass

        # 3. 檢查 Python hailort 庫是否可用
        try:
            import hailo_platform
            self._device_info["python_binding"] = True
        except ImportError:
            self._device_info["python_binding"] = False

    def is_available(self) -> bool:
        """回傳當前是否有可用的 Hailo NPU 硬體加速"""
        return bool(self._is_available)

    def get_info(self) -> Dict[str, Any]:
        """獲取硬體詳細中繼資料"""
        return self._device_info


def is_hailo_available() -> bool:
    """便利全域探測函數"""
    return HailoDeviceManager().is_available()
