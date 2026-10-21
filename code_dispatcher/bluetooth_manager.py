# f:\12_prj_raspi5\telegram_agent\bluetooth_manager.py
"""
Headless Bluetooth Audio Headset Manager for Raspberry Pi 5.
Features:
1. One-click Discovery & Scanning (Filtered for Audio/Headsets)
2. Automated Headless Pairing (NoInputNoOutput Agent) & Trusting
3. PipeWire / WirePlumber Audio Sink Auto-routing
4. Persistent Target Device Registry (JSON)
5. Background Auto-Reconnect Worker with Mutual Exclusion Lock
"""

import os
import re
import json
import time
import logging
import subprocess
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger("Pi5Bluetooth")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(DATA_DIR, "bluetooth_config.json")
SCAN_CACHE_PATH = os.path.join(DATA_DIR, "last_scan_cache.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Mutual exclusion lock to prevent auto_reconnect_worker from colliding with pair_and_trust
_is_pairing_lock: bool = False


def load_bt_config() -> Dict:
    """Load persistent Bluetooth configuration."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read bluetooth config: {e}")
    return {"default_mac": None, "default_name": None, "auto_reconnect": True}


def save_bt_config(config: Dict) -> bool:
    """Save Bluetooth configuration to JSON file."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save bluetooth config: {e}")
        return False


def load_scan_cache() -> List[Dict[str, str]]:
    """Load scanned devices cache from file."""
    if os.path.exists(SCAN_CACHE_PATH):
        try:
            with open(SCAN_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read scan cache: {e}")
    return []


def save_scan_cache(devices: List[Dict[str, str]]) -> None:
    """Save scanned devices cache to file."""
    try:
        with open(SCAN_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(devices, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save scan cache: {e}")


def get_controller_status() -> Tuple[bool, str]:
    """Check if Bluetooth controller is powered on."""
    try:
        res = subprocess.run(["bluetoothctl", "show"], capture_output=True, text=True, timeout=5)
        if "Powered: yes" in res.stdout:
            return True, "藍牙控制器正常運作中 (Powered On)"
        else:
            subprocess.run(["bluetoothctl", "power", "on"], capture_output=True, timeout=5)
            return True, "已重新啟用藍牙控制器 (Power On)"
    except Exception as e:
        return False, f"藍牙控制器異常: {e}"


def scan_devices(timeout_sec: int = 8) -> List[Dict[str, Any]]:
    """
    Scan for nearby Bluetooth devices for a given duration.
    Filters and formats devices, caching them for convenient selection.
    Uses 'scan on' to capture both Dual-Mode BR/EDR and BLE Audio devices in real time.
    """
    get_controller_status()
    logger.info(f"Starting Bluetooth scan for {timeout_sec} seconds...")
    discovered_map: Dict[str, str] = {}

    try:
        # 1. 執行全域掃描 (scan on)，捕獲即時發現的設備 (包含 BLE 與傳統藍牙雙模設備)
        proc = subprocess.Popen(
            ["bluetoothctl", "--timeout", str(timeout_sec), "scan", "on"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            stdout_data, _ = proc.communicate(timeout=timeout_sec + 3)
            EXCLUDE_PROPS = ("RSSI:", "ManufacturerData", "TxPower:", "UUID", "ServicesResolved:", "Connected:", "Paired:", "Trusted:", "Modalias:", "Class:", "Icon:", "Blocked:", "LegacyPairing:", "Adapter:")
            for line in stdout_data.splitlines():
                if "[NEW] Device" in line:
                    m_new = re.search(r"Device\s+([0-9A-Fa-f:]{17})\s+(.+)$", line)
                    if m_new:
                        mac = m_new.group(1).upper()
                        name = m_new.group(2).strip()
                        if name and not any(name.startswith(p) for p in EXCLUDE_PROPS) and not re.match(r"^[0-9A-Fa-f:-]{17}$", name):
                            discovered_map[mac] = name
                elif "Name:" in line or "Alias:" in line:
                    m_name = re.search(r"Device\s+([0-9A-Fa-f:]{17})\s+(?:Name|Alias):\s*(.+)$", line)
                    if m_name:
                        mac = m_name.group(1).upper()
                        name = m_name.group(2).strip()
                        if name and not any(name.startswith(p) for p in EXCLUDE_PROPS):
                            discovered_map[mac] = name
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
    except Exception as e:
        logger.warning(f"Bluetooth scan timeout/interrupted: {e}")

    # 2. 結合 bluetoothctl devices 中的所有設備 (含已配對/記憶之裝置)
    try:
        res = subprocess.run(["bluetoothctl", "devices"], capture_output=True, text=True, timeout=5)
        for line in res.stdout.strip().split("\n"):
            match = re.match(r"^Device\s+([0-9A-Fa-f:]{17})\s+(.*)$", line.strip())
            if match:
                mac = match.group(1).upper()
                name = match.group(2).strip()
                if mac not in discovered_map or not discovered_map[mac]:
                    discovered_map[mac] = name
    except Exception as e:
        logger.error(f"Error querying bluetoothctl devices: {e}")

    # 3. 逐一查詢設備詳情並過濾有效名稱
    devices: List[Dict[str, Any]] = []
    seen_macs = set()
    for mac, name in discovered_map.items():
        if mac in seen_macs:
            continue
        seen_macs.add(mac)

        info_res = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True, timeout=3)
        info_out = info_res.stdout
        is_connected = "Connected: yes" in info_out
        is_paired = "Paired: yes" in info_out
        is_trusted = "Trusted: yes" in info_out

        # 設備真實名稱修正
        m_real_name = re.search(r"Name:\s*(.+)", info_out) or re.search(r"Alias:\s*(.+)", info_out)
        if m_real_name:
            resolved_name = m_real_name.group(1).strip()
            if resolved_name and not re.match(r"^[0-9A-Fa-f:-]{17}$", resolved_name):
                name = resolved_name

        # 提取 RSSI 與電量
        rssi_m = re.search(r"RSSI:\s*(-?\d+)", info_out)
        rssi = int(rssi_m.group(1)) if rssi_m else None
        bat_m = re.search(r"Battery Percentage:\s*.*\((\d+)\)", info_out) or re.search(r"Battery Percentage:\s*(\d+)", info_out)
        battery = int(bat_m.group(1)) if bat_m else None

        # 僅保留具有可讀名稱之音訊相關設備或已配對/連線設備
        # 排除純隨機 MAC 廣播（無名稱且未配對未連線者）
        has_readable_name = bool(name and not re.match(r"^[0-9A-Fa-f:-]{17}$", name))
        if is_paired or is_connected or has_readable_name:
            devices.append({
                "mac": mac,
                "name": name if name else "未知音訊設備",
                "connected": is_connected,
                "paired": is_paired,
                "trusted": is_trusted,
                "rssi": rssi,
                "battery_level": battery
            })

    # 排序：連線中 > 已配對 > 訊號強度由強到弱
    devices.sort(key=lambda d: (
        1 if d["connected"] else (2 if d["paired"] else 3),
        -(d["rssi"] if d["rssi"] is not None else -999)
    ))

    save_scan_cache(devices)
    return devices


def pair_and_trust(target: str) -> Tuple[bool, str]:
    """
    Pair, trust, and connect to a target device (by MAC or 1-based index).
    Uses interactive BlueZ session with NoInputNoOutput agent for headless environments.
    """
    global _is_pairing_lock

    if _is_pairing_lock:
        return False, "⚠️ 系統目前正有另一項藍牙配對任務進行中，請稍候 10 秒後再試。"

    _is_pairing_lock = True
    try:
        target = target.strip()
        target_mac = None
        target_name = "藍牙耳麥"

        cached_devices = load_scan_cache()

        # 1. Resolve target
        if target.isdigit():
            idx = int(target) - 1
            if 0 <= idx < len(cached_devices):
                target_mac = cached_devices[idx]["mac"]
                target_name = cached_devices[idx]["name"]
            else:
                return False, f"❌ 無效的清單編號 [{target}]，請先輸入 `/bt_scan` 獲取最新清單。"
        elif re.match(r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$", target):
            target_mac = target.upper()
            for dev in cached_devices:
                if dev["mac"] == target_mac:
                    target_name = dev["name"]
                    break
        else:
            for dev in cached_devices:
                if target.lower() in dev["name"].lower():
                    target_mac = dev["mac"]
                    target_name = dev["name"]
                    break
            if not target_mac:
                return False, f"❌ 找不到符合「{target}」的設備，請先執行 `/bt_scan` 查看編號。"

        logger.info(f"Starting Headless Pairing for: {target_name} ({target_mac})...")

        # Step 1: Ensure controller powered on & agent registered
        subprocess.run(["bluetoothctl", "power", "on"], capture_output=True, timeout=3)
        subprocess.run(["bluetoothctl", "agent", "NoInputNoOutput"], capture_output=True, timeout=3)
        subprocess.run(["bluetoothctl", "default-agent"], capture_output=True, timeout=3)

        # Step 2: Ensure device is discovered in BlueZ BR/EDR
        dev_check = subprocess.run(["bluetoothctl", "info", target_mac], capture_output=True, text=True, timeout=3)
        if "Device " not in dev_check.stdout:
            logger.info(f"Target {target_mac} not in BlueZ cache, performing 4s scan...")
            subprocess.run(["bluetoothctl", "--timeout", "4", "scan", "on"], capture_output=True, text=True, timeout=6)

        # Step 3: Trust, Pair and Connect in sequence
        subprocess.run(["bluetoothctl", "trust", target_mac], capture_output=True, text=True, timeout=5)
        pair_res = subprocess.run(["bluetoothctl", "pair", target_mac], capture_output=True, text=True, timeout=12)
        conn_res = subprocess.run(["bluetoothctl", "connect", target_mac], capture_output=True, text=True, timeout=10)

        combined_log = (pair_res.stdout + "\n" + pair_res.stderr + "\n" + conn_res.stdout + "\n" + conn_res.stderr).strip()

        # Step 3: Check device state
        time.sleep(1)
        info_res = subprocess.run(["bluetoothctl", "info", target_mac], capture_output=True, text=True, timeout=5)
        info_out = info_res.stdout

        is_connected = "Connected: yes" in info_out
        is_paired = "Paired: yes" in info_out
        is_trusted = "Trusted: yes" in info_out

        name_match = re.search(r"Name:\s*(.+)", info_out)
        if name_match:
            target_name = name_match.group(1).strip()

        # Step 4: If paired but not connected, try one direct connect
        if (is_paired or is_trusted) and not is_connected:
            conn_res = subprocess.run(["bluetoothctl", "connect", target_mac], capture_output=True, text=True, timeout=8)
            time.sleep(1)
            info_res = subprocess.run(["bluetoothctl", "info", target_mac], capture_output=True, text=True, timeout=3)
            info_out = info_res.stdout
            is_connected = "Connected: yes" in info_out

        if is_connected:
            config = load_bt_config()
            config["default_mac"] = target_mac
            config["default_name"] = target_name
            config["auto_reconnect"] = True
            save_bt_config(config)

            # Route audio default sink to bluetooth in PipeWire
            audio_switch_msg = _switch_pipewire_default_sink(target_name)

            return True, (
                f"🎉 **配對與連線成功！**\n"
                f"• 耳麥設備：*{target_name}*\n"
                f"• MAC 位址：`{target_mac}`\n"
                f"• 音訊狀態：{audio_switch_msg}\n"
                f"• 自動秒連：已寫入設定，耳機開機靠近時樹莓派將自動連線！"
            )

        elif is_paired or is_trusted:
            config = load_bt_config()
            config["default_mac"] = target_mac
            config["default_name"] = target_name
            config["auto_reconnect"] = True
            save_bt_config(config)
            return True, (
                f"✅ **已完成藍牙信任配對：** *{target_name}* (`{target_mac}`)\n"
                f"⚠️ 目前音訊通道尚未建立（可能是耳麥剛重啟中）。\n"
                f"背景守護進程已啟動，將在數秒內自動為您連上！您亦可隨時輸入 `/bt_connect`。"
            )

        else:
            # Detailed diagnosis
            diag = []
            if "103" in combined_log or "abort" in combined_log.lower() or "br-connection-busy" in combined_log:
                diag.append("• **耳麥正連接著其他設備：** 請先關閉手機/電腦上的藍牙，切斷它與耳機的連線。")
                diag.append("• **未處於配對模式：** 請長按耳機配對鍵直到指示燈「紅藍交替快速閃爍」。")
            elif "not available" in combined_log.lower():
                diag.append("• **找不到設備信號：** 請確保耳機已開機，且距離樹莓派 3 公尺以內。")
            else:
                snippet = combined_log[-300:] if len(combined_log) > 300 else combined_log
                diag.append(f"• 系統日誌摘要：`{snippet.strip()}`")
                diag.append("• 請確認耳機處於配對模式（紅藍閃爍）後再次嘗試。")

            diag_text = "\n".join(diag)
            return False, f"❌ **配對失敗：{target_name}** (`{target_mac}`)\n\n**可能原因與排查方式：**\n{diag_text}"

    finally:
        _is_pairing_lock = False


def _switch_pipewire_default_sink(device_name_hint: str) -> str:
    """Attempt to route PipeWire default sink to the newly connected Bluetooth device."""
    try:
        wp_status = subprocess.run(["wpctl", "status"], capture_output=True, text=True, timeout=5)
        lines = wp_status.stdout.split("\n")
        in_sinks = False
        target_sink_id = None

        for line in lines:
            if "Sinks:" in line:
                in_sinks = True
                continue
            if in_sinks:
                if line.strip() == "" or "Sources:" in line or "Filters:" in line:
                    break
                # Format: │  *   53. Built-in Audio Stereo          [vol: 1.00]
                # Format: │      72. BlueZ 5.82 ... [vol: 0.80]
                m = re.search(r"(\d+)\.\s+(.*)", line)
                if m:
                    sink_id = m.group(1)
                    sink_name = m.group(2)
                    if "bluez" in sink_name.lower() or device_name_hint.lower() in sink_name.lower():
                        target_sink_id = sink_id
                        break

        if target_sink_id:
            subprocess.run(["wpctl", "set-default", target_sink_id], capture_output=True, timeout=3)

        # 確保藍牙音訊輸入處於 16kHz mSBC 模式 (headset-head-unit)
        try:
            import json
            dump_proc = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=3)
            if dump_proc.returncode == 0:
                nodes = json.loads(dump_proc.stdout)
                for n in nodes:
                    props = n.get("info", {}).get("props", {})
                    if props.get("device.api") == "bluez5" and n.get("type") == "PipeWire:Interface:Device":
                        dev_id = n.get("id")
                        params = n.get("info", {}).get("params", {})
                        active_profiles = params.get("Profile", [])
                        if not any(p.get("name") == "headset-head-unit" for p in active_profiles):
                            msbc_idx = 196865
                            for ep in params.get("EnumProfile", []):
                                if ep.get("name") == "headset-head-unit":
                                    msbc_idx = ep.get("index", 196865)
                                    break
                            subprocess.run(["wpctl", "set-profile", str(dev_id), str(msbc_idx)], capture_output=True, timeout=3)
        except Exception:
            pass

        if target_sink_id:
            return "已自動將 PipeWire 音訊輸出切換為該耳麥並鎖定 16kHz mSBC"
        return "PipeWire 已識別該藍牙音訊端點"
    except Exception as e:
        return f"音訊輸出切換略過: {e}"


def connect_default_device() -> Tuple[bool, str]:
    """Connect to the configured default device."""
    config = load_bt_config()
    target_mac = config.get("default_mac")
    target_name = config.get("default_name") or "預設耳麥"

    if not target_mac:
        return False, "尚未設定預設耳麥，請先執行 `/bt_scan` 與 `/bt_pair`。"

    try:
        res = subprocess.run(["bluetoothctl", "connect", target_mac], capture_output=True, text=True, timeout=10)
        if "Connection successful" in res.stdout or is_device_connected(target_mac):
            _switch_pipewire_default_sink(target_name)
            return True, f"🎧 已成功連線至預設耳麥：*{target_name}* (`{target_mac}`)"
        else:
            return False, f"無法連線至 {target_name}，請確認耳麥已開機且處於可連線狀態。"
    except Exception as e:
        return False, f"連線錯誤: {e}"


def disconnect_device(mac: Optional[str] = None) -> Tuple[bool, str]:
    """Disconnect the current or specified Bluetooth device."""
    if not mac:
        config = load_bt_config()
        mac = config.get("default_mac")

    if not mac:
        return False, "未指定要中斷的設備 MAC。"

    try:
        subprocess.run(["bluetoothctl", "disconnect", mac], capture_output=True, text=True, timeout=5)
        return True, f"🔌 已中斷藍牙連線：`{mac}`"
    except Exception as e:
        return False, f"中斷連線失敗: {e}"


def is_device_connected(mac: str) -> bool:
    """Check if specific MAC address is actively connected."""
    try:
        res = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True, timeout=3)
        return "Connected: yes" in res.stdout
    except Exception:
        return False


def get_full_bt_status() -> Dict:
    """
    Get comprehensive status of Bluetooth controller, paired devices, and PipeWire audio sinks.
    """
    config = load_bt_config()
    default_mac = config.get("default_mac")
    default_name = config.get("default_name")

    controller_ok, ctrl_msg = get_controller_status()

    # Get paired devices using modern BlueZ syntax
    paired_devices = []
    default_connected = False
    try:
        res = subprocess.run(["bluetoothctl", "devices", "Paired"], capture_output=True, text=True, timeout=5)
        for line in res.stdout.strip().split("\n"):
            m = re.match(r"^Device\s+([0-9A-Fa-f:]{17})\s+(.*)$", line.strip())
            if m:
                mac, name = m.group(1).upper(), m.group(2).strip()
                connected = is_device_connected(mac)
                if mac == default_mac and connected:
                    default_connected = True
                paired_devices.append({
                    "mac": mac,
                    "name": name,
                    "connected": connected,
                    "is_default": (mac == default_mac)
                })
    except Exception as e:
        logger.warning(f"Error fetching paired devices: {e}")

    # Get PipeWire Sinks
    pipewire_sinks = []
    try:
        res_wp = subprocess.run(["wpctl", "status"], capture_output=True, text=True, timeout=5)
        sinks_section = False
        for line in res_wp.stdout.split("\n"):
            if "Sinks:" in line:
                sinks_section = True
                continue
            if sinks_section:
                if line.strip() == "" or "Sources:" in line or "Filters:" in line:
                    break
                pipewire_sinks.append(line.strip())
    except Exception:
        pass

    return {
        "controller_ok": controller_ok,
        "controller_message": ctrl_msg,
        "default_mac": default_mac,
        "default_name": default_name,
        "default_connected": default_connected,
        "paired_devices": paired_devices,
        "pipewire_sinks": pipewire_sinks,
        "auto_reconnect": config.get("auto_reconnect", True)
    }


_last_connection_state: Optional[bool] = None


def get_device_battery(mac: str) -> Optional[int]:
    """Retrieve device battery percentage from BlueZ info."""
    try:
        res = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True, timeout=3)
        m = re.search(r"Battery Percentage:\s*.*\((\d+)\)", res.stdout) or re.search(r"Battery Percentage:\s*(\d+)", res.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def auto_reconnect_tick() -> Optional[str]:
    """
    Invoked periodically by background worker (every 10~15s).
    Monitors Bluetooth connection transitions (Connect / Disconnect).
    Returns a formatted notification string whenever state changes.
    """
    global _is_pairing_lock, _last_connection_state

    if _is_pairing_lock:
        return None

    config = load_bt_config()
    default_mac = config.get("default_mac")
    auto_reconn = config.get("auto_reconnect", True)
    auto_map = config.get("auto_connect_devices", {})

    # Check if ANY device is currently connected
    res_conn = subprocess.run(["bluetoothctl", "devices", "Connected"], capture_output=True, text=True, timeout=3)
    currently_connected_mac = None
    for line in res_conn.stdout.strip().split("\n"):
        m_c = re.match(r"^Device\s+([0-9A-Fa-f:]{17})", line.strip())
        if m_c:
            currently_connected_mac = m_c.group(1).upper()
            break

    # If no device connected and auto reconnect is enabled:
    if not currently_connected_mac and auto_reconn:
        candidates = []
        if default_mac and auto_map.get(default_mac.upper(), True):
            candidates.append(default_mac.upper())
        for mac, enabled in auto_map.items():
            mac_u = mac.upper()
            if enabled and mac_u not in candidates:
                candidates.append(mac_u)

        for cand_mac in candidates:
            try:
                res = subprocess.run(["bluetoothctl", "connect", cand_mac], capture_output=True, text=True, timeout=6)
                if "Connection successful" in res.stdout or is_device_connected(cand_mac):
                    currently_connected_mac = cand_mac
                    break
            except Exception:
                pass

    is_connected = bool(currently_connected_mac)

    # Initialize baseline state on first run
    if _last_connection_state is None:
        _last_connection_state = is_connected
        return None

    # Transition: Disconnected -> Connected
    if not _last_connection_state and is_connected:
        _last_connection_state = True
        name = config.get("default_name") or "藍牙耳麥"
        bat = get_device_battery(default_mac)
        bat_str = f"{bat}%" if bat is not None else "未知"
        _switch_pipewire_default_sink(name)
        logger.info(f"Bluetooth device auto-connected: {name} ({default_mac})")
        return (
            f"🎧 **藍牙耳麥已自動連線！(Auto-Connected)**\n"
            f"• 設備名稱：*{name}*\n"
            f"• MAC 位址：`{default_mac}`\n"
            f"• 🔋 耳機電量：*{bat_str}*\n"
            f"• 🎙️ 語音麥克風已就緒，可隨時發送 `/voice 5` 進行錄音測試！"
        )

    # Transition: Connected -> Disconnected
    if _last_connection_state and not is_connected:
        _last_connection_state = False
        name = config.get("default_name") or "藍牙耳麥"
        logger.info(f"Bluetooth device disconnected: {name}")
        return (
            f"🔌 **藍牙耳麥已中斷連線 (Disconnected)**\n"
            f"• 設備：*{name}*\n"
            f"• 系統音訊已自動切換回板載預設端點。\n"
            f"• 耳機下次開機靠近時，樹莓派將自動秒連！"
        )

    return None

