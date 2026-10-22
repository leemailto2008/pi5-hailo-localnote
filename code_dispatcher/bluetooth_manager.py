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
import pty
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger("Pi5Bluetooth")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(DATA_DIR, "bluetooth_config.json")
SCAN_CACHE_PATH = os.path.join(DATA_DIR, "last_scan_cache.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Mutual exclusion lock to prevent auto_reconnect_worker from colliding with pair_and_trust
_is_pairing_lock: bool = False

# How long pair_and_trust keeps BR/EDR inquiry running while waiting for the headset to answer.
PAIR_DISCOVERY_WAIT_SEC: int = 12

# Held during scan_devices() and pair_and_trust(). auto_reconnect_tick() must not issue
# `connect` while a discovery/pairing session is running: a premature connect to a headset
# that is in pairing mode kicks it OUT of pairing mode (observed with RichChip headsets).
_bt_exclusive = threading.Lock()


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


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class BluetoothctlSession:
    """
    Long-lived interactive bluetoothctl session driven through a pseudo-terminal (pty).

    Why: BlueZ discards temporary (unpaired) devices as soon as the discovery client
    disconnects from D-Bus, so "scan" and "pair" issued as separate bluetoothctl
    processes race against that purge and fail with "Device not available".
    Keeping scan -> pair -> trust -> connect inside ONE session removes the race.
    A pty is used because bluetoothctl is readline-based and block-buffers stdout
    when attached to a plain pipe.
    """

    def __init__(self) -> None:
        self._master_fd: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self._lines: List[str] = []
        self._cond = threading.Condition()
        self._reader: Optional[threading.Thread] = None
        self._closed = False

    def start(self) -> None:
        master_fd, slave_fd = pty.openpty()
        self._master_fd = master_fd
        self._proc = subprocess.Popen(
            ["bluetoothctl"],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            start_new_session=True, close_fds=True,
        )
        os.close(slave_fd)
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="btctl-reader")
        self._reader.start()
        # Wait for the prompt so the first command is not swallowed.
        self.wait_for([r"\[bluetooth\]#|Agent registered|Waiting to connect"], timeout=3)

    def _read_loop(self) -> None:
        buf = ""
        while True:
            try:
                chunk = os.read(self._master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            parts = buf.split("\n")
            buf = parts.pop()
            with self._cond:
                self._lines.extend(_strip_ansi(l) for l in parts)
                self._cond.notify_all()
        if buf:
            with self._cond:
                self._lines.append(_strip_ansi(buf))
                self._cond.notify_all()

    def send(self, cmd: str) -> None:
        if self._master_fd is None or self._closed:
            raise RuntimeError("bluetoothctl session not started")
        os.write(self._master_fd, (cmd + "\n").encode("utf-8"))

    def wait_for(self, patterns: List[str], timeout: float, start_index: int = 0) -> Optional[str]:
        """Block until a line (from start_index on) matches any pattern, or timeout. Returns the line."""
        compiled = [re.compile(pt) for pt in patterns]
        deadline = time.monotonic() + timeout
        idx = start_index
        with self._cond:
            while True:
                while idx < len(self._lines):
                    line = self._lines[idx]
                    idx += 1
                    if any(c.search(line) for c in compiled):
                        return line
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    def lines_since(self, start_index: int) -> List[str]:
        with self._cond:
            return list(self._lines[start_index:])

    def cursor(self) -> int:
        with self._cond:
            return len(self._lines)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._master_fd is not None:
                os.write(self._master_fd, b"quit\n")
        except OSError:
            pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=3)
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

    def __enter__(self) -> "BluetoothctlSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_DEVICE_EVENT_RE = re.compile(r"\[(?:NEW|CHG)\]\s+Device\s+([0-9A-Fa-f:]{17})\s+(.+)$")
_MAC_AS_NAME_RE = re.compile(r"^[0-9A-Fa-f:-]{17}$")


def _parse_discovery_lines(lines: List[str], discovered_map: Dict[str, str]) -> None:
    """Fill discovered_map[mac] = name from bluetoothctl [NEW]/[CHG] events (ANSI already stripped)."""
    for line in lines:
        m = _DEVICE_EVENT_RE.search(line)
        if not m:
            continue
        mac = m.group(1).upper()
        rest = m.group(2).strip()
        if "[NEW]" in line:
            name = rest
        else:
            m_name = re.match(r"(?:Name|Alias):\s*(.+)$", rest)
            if not m_name:
                # Device already in BlueZ cache only emits [CHG] (RSSI/Connected/...): remember the
                # MAC, the name is resolved later via `bluetoothctl info`.
                discovered_map.setdefault(mac, "")
                continue
            name = m_name.group(1).strip()
        if name and not _MAC_AS_NAME_RE.match(name) and ":" not in name.split(" ")[0]:
            discovered_map[mac] = name


def _list_devices(filter_name: str) -> Dict[str, str]:
    """bluetoothctl devices <Paired|Trusted|Connected> -> {mac: name}"""
    out: Dict[str, str] = {}
    try:
        res = subprocess.run(["bluetoothctl", "devices", filter_name], capture_output=True, text=True, timeout=3)
        for line in res.stdout.splitlines():
            m = re.match(r"^Device\s+([0-9A-Fa-f:]{17})\s+(.*)$", line.strip())
            if m:
                out[m.group(1).upper()] = m.group(2).strip()
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning(f"bluetoothctl devices {filter_name} failed: {e}")
    return out


def scan_devices(timeout_sec: int = 10) -> List[Dict[str, Any]]:
    """
    Scan for nearby Bluetooth HEADSETS (BR/EDR only) for a given duration.

    Design notes (see CLAUDE.md gotcha on BT scanning):
    - HFP/A2DP headsets live on classic BR/EDR. `scan on` (transport auto) interleaves
      LE and BR/EDR inquiry and in practice misses headsets in an 8s window, while the
      same headset's BLE companion advertisement (random address, e.g. "Air Shutter")
      is always seen and misleads the user. `scan bredr` fixes both.
    - Unpaired BLE random-address entries are never listed.
    """
    if not _bt_exclusive.acquire(blocking=False):
        logger.info("scan_devices skipped: another scan/pair session is running, returning cache")
        return load_scan_cache()
    try:
        return _scan_devices_locked(timeout_sec)
    finally:
        _bt_exclusive.release()


def _scan_devices_locked(timeout_sec: int) -> List[Dict[str, Any]]:
    get_controller_status()
    logger.info(f"Starting Bluetooth BR/EDR scan for {timeout_sec} seconds...")
    discovered_map: Dict[str, str] = {}

    try:
        with BluetoothctlSession() as sess:
            start_idx = sess.cursor()
            sess.send("scan bredr")
            started = sess.wait_for([r"Discovery started", r"Failed to start discovery", r"Invalid argument"], timeout=4)
            if started and "Discovery started" not in started:
                logger.warning(f"scan bredr rejected ({started}), falling back to 'scan on'")
                sess.send("scan on")
                sess.wait_for([r"Discovery started"], timeout=4)
            time.sleep(timeout_sec)
            sess.send("scan off")
            sess.wait_for([r"Discovery stopped", r"Discovering: no"], timeout=3)
            _parse_discovery_lines(sess.lines_since(start_idx), discovered_map)
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
        logger.warning(f"Bluetooth scan session failed: {e}")

    paired_map = _list_devices("Paired")
    connected_map = _list_devices("Connected")
    trusted_map = _list_devices("Trusted")

    # Paired devices are always listed (even when out of range)
    for mac, name in paired_map.items():
        if mac not in discovered_map:
            discovered_map[mac] = name if (name and not _MAC_AS_NAME_RE.match(name)) else "已配對設備"

    def _inspect_device(item: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        mac, name = item
        is_paired = mac in paired_map
        is_connected = mac in connected_map
        is_trusted = mac in trusted_map
        rssi: Optional[int] = None
        battery: Optional[int] = None
        try:
            # Best effort only: BlueZ may already have purged a temporary device.
            info_res = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True, timeout=2)
            info_out = info_res.stdout
            first_line = info_out.splitlines()[0] if info_out else ""
            if "(random)" in first_line and not is_paired:
                return None  # BLE companion advertisement (e.g. "Air Shutter"), not a headset endpoint
            m_real_name = re.search(r"Name:\s*(.+)", info_out) or re.search(r"Alias:\s*(.+)", info_out)
            if m_real_name:
                resolved = m_real_name.group(1).strip()
                if resolved and not _MAC_AS_NAME_RE.match(resolved):
                    name = resolved
            rssi_m = re.search(r"RSSI:\s*(-?\d+)", info_out)
            rssi = int(rssi_m.group(1)) if rssi_m else None
            bat_m = re.search(r"Battery Percentage:\s*.*\((\d+)\)", info_out) or re.search(r"Battery Percentage:\s*(\d+)", info_out)
            battery = int(bat_m.group(1)) if bat_m else None
        except (subprocess.SubprocessError, OSError) as e:
            logger.debug(f"info {mac} unavailable: {e}")

        has_readable_name = bool(name and not _MAC_AS_NAME_RE.match(name))
        if not (is_paired or is_connected or has_readable_name):
            return None
        return {
            "mac": mac,
            "name": name if name else "未知音訊設備",
            "connected": is_connected,
            "paired": is_paired,
            "trusted": is_trusted,
            "rssi": rssi,
            "battery_level": battery,
        }

    devices: List[Dict[str, Any]] = []
    if discovered_map:
        with ThreadPoolExecutor(max_workers=8) as pool:
            devices = list(filter(None, pool.map(_inspect_device, discovered_map.items())))

    devices.sort(key=lambda d: (
        1 if d["connected"] else (2 if d["paired"] else 3),
        -(d["rssi"] if d["rssi"] is not None else -999)
    ))

    save_scan_cache(devices)
    logger.info(f"Scan finished, found {len(devices)} headset candidates")
    return devices


def pair_and_trust(target: str) -> Tuple[bool, str]:
    """
    Pair, trust, and connect to a target device (by MAC or 1-based index).
    Uses interactive BlueZ session with NoInputNoOutput agent for headless environments.
    """
    global _is_pairing_lock

    if _is_pairing_lock:
        return False, "⚠️ 系統目前正有另一項藍牙配對任務進行中，請稍候 10 秒後再試。"

    if not _bt_exclusive.acquire(blocking=False):
        return False, "⚠️ 藍牙掃描進行中，請等掃描結束後再配對。"
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

        # Steps 1-3 run inside ONE bluetoothctl session: scan (BR/EDR) -> pair -> trust -> connect.
        # Separate processes race against BlueZ purging temporary devices when discovery ends.
        combined_log = ""
        pair_ok = False
        try:
            with BluetoothctlSession() as sess:
                sess.send("power on")
                sess.send("agent NoInputNoOutput")
                sess.send("default-agent")
                sess.wait_for([r"Default agent request successful", r"Agent registered"], timeout=3)

                mac_re = re.escape(target_mac)
                already_known = subprocess.run(["bluetoothctl", "info", target_mac], capture_output=True, text=True, timeout=3).stdout
                if "Paired: yes" not in already_known:
                    sess.send("scan bredr")
                    sess.wait_for([r"Discovery started"], timeout=4)
                    seen = sess.wait_for([rf"Device\s+{mac_re}"], timeout=PAIR_DISCOVERY_WAIT_SEC)
                    if not seen:
                        sess.send("scan off")
                        return False, (
                            f"❌ **找不到 {target_name}** (`{target_mac}`)\n"
                            f"{PAIR_DISCOVERY_WAIT_SEC} 秒內未收到耳麥的傳統藍牙 (BR/EDR) 回應。\n"
                            f"• 耳麥一旦連上任何裝置就會離開配對模式，請**重新長按配對鍵**至指示燈閃爍後再試。\n"
                            f"• 若耳麥仍連著手機/電腦，請先在該裝置上斷開。"
                        )

                cur = sess.cursor()
                sess.send(f"pair {target_mac}")
                pair_line = sess.wait_for(
                    [r"Pairing successful", r"Already Paired", r"Failed to pair", r"not available",
                     r"AuthenticationFailed", r"AuthenticationCanceled", r"ConnectionAttemptFailed", r"AlreadyExists"],
                    timeout=20, start_index=cur)
                sess.send("scan off")
                pair_ok = bool(pair_line) and ("Pairing successful" in pair_line or "Already Paired" in pair_line or "AlreadyExists" in pair_line)

                if pair_ok:
                    # Trust ONLY after a successful pairing (never leave trusted-but-unpaired ghosts)
                    cur = sess.cursor()
                    sess.send(f"trust {target_mac}")
                    sess.wait_for([r"trust succeeded", r"Failed to set trust"], timeout=5, start_index=cur)
                    cur = sess.cursor()
                    sess.send(f"connect {target_mac}")
                    sess.wait_for([r"Connection successful", r"Failed to connect"], timeout=15, start_index=cur)

                combined_log = "\n".join(sess.lines_since(0)).strip()
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            logger.error(f"bluetoothctl session error during pairing: {e}")
            combined_log = str(e)

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
        if is_paired and not is_connected:
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

        elif is_paired:
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
        _bt_exclusive.release()


def _switch_pipewire_default_sink(device_name_hint: str) -> str:
    """Attempt to route PipeWire default sink and source to the newly connected Bluetooth device."""
    try:
        wp_status = subprocess.run(["wpctl", "status"], capture_output=True, text=True, timeout=5)
        lines = wp_status.stdout.split("\n")
        section = None
        target_sink_id = None
        target_source_id = None

        for line in lines:
            if "Sinks:" in line:
                section = "sinks"
                continue
            elif "Sources:" in line:
                section = "sources"
                continue
            elif "Filters:" in line or "Streams:" in line:
                section = None
                continue

            if section in ("sinks", "sources"):
                m = re.search(r"(\d+)\.\s+(.*)", line)
                if m:
                    node_id = m.group(1)
                    node_desc = m.group(2)
                    if "bluez" in node_desc.lower() or (device_name_hint and device_name_hint.lower() in node_desc.lower()):
                        if section == "sinks" and not target_sink_id:
                            target_sink_id = node_id
                        elif section == "sources" and not target_source_id:
                            target_source_id = node_id

        if target_sink_id:
            subprocess.run(["wpctl", "set-default", target_sink_id], capture_output=True, timeout=3)
        if target_source_id:
            subprocess.run(["wpctl", "set-default", target_source_id], capture_output=True, timeout=3)

        # 檢驗藍牙設定檔：若未具備語音能力且明確支援 headset-head-unit，才進行切換
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
                        is_voice_profile = any(
                            p.get("name") in ("headset-head-unit", "bap-duplex", "bap-source", "hsp-hs")
                            for p in active_profiles
                        )
                        if not is_voice_profile:
                            for ep in params.get("EnumProfile", []):
                                if ep.get("name") == "headset-head-unit":
                                    msbc_idx = ep.get("index")
                                    if msbc_idx is not None:
                                        subprocess.run(["wpctl", "set-profile", str(dev_id), str(msbc_idx)], capture_output=True, timeout=3)
                                    break
        except Exception:
            pass

        msg_parts = []
        if target_sink_id:
            msg_parts.append(f"耳麥揚聲器 (ID {target_sink_id})")
        if target_source_id:
            msg_parts.append(f"耳麥麥克風 (ID {target_source_id})")

        if msg_parts:
            return f"已自動將 PipeWire 預設導向 {' 與 '.join(msg_parts)}"
        return "PipeWire 已識別該藍牙音訊端點"
    except Exception as e:
        return f"音訊路由略過: {e}"


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

    if _is_pairing_lock or _bt_exclusive.locked():
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

        # Never `connect` an unpaired device: BlueZ would open an SDP-only link that is dropped
        # immediately, and the headset leaves pairing mode, so the user can never pair it via the UI.
        paired_macs = _list_devices("Paired")
        candidates = [m for m in candidates if m in paired_macs]

        for cand_mac in candidates:
            if _bt_exclusive.locked():
                break
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

