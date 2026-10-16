# f:\12_prj_raspi5\code_dispatcher\hotspot_controller.py
"""
High-Reliability Atomic Hotspot Controller with Auto-Fallback Watchdog.
Prevents Raspberry Pi 5 from being bricked, trapped in offline state, or permanently disconnected.
"""

import os
import sys
import time
import subprocess
import threading

STA_PROFILE = "netplan-wlan0-My_5G_Guest1"
AP_PROFILE = "Pi5-Hotspot"
WATCHDOG_SCRIPT = "/home/pi/code_dispatcher/hotspot_watchdog.py"

NFT_GUARD_FILE = "/etc/nftables.hotspot_guard.nft"

def find_sta_profile() -> str:
    """Find the best matching STA Wi-Fi connection profile."""
    try:
        res = subprocess.run(["nmcli", "-t", "-f", "NAME,TYPE", "connection", "show"], capture_output=True, text=True, timeout=5)
        for line in res.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split(":")
            if len(parts) >= 2:
                name, ctype = parts[0], parts[1]
                if "wireless" in ctype and "hotspot" not in name.lower() and "ap" not in name.lower():
                    return name
    except Exception:
        pass
    return "netplan-wlan0-My_5G_Guest1"

def enable_hotspot_guard():
    """Apply kernel-level nftables guard if file exists."""
    if os.path.exists(NFT_GUARD_FILE):
        try:
            subprocess.run(["sudo", "nft", "-f", NFT_GUARD_FILE], capture_output=True, timeout=5)
            print("[HotspotController] Applied kernel HotspotGuard.")
        except Exception as e:
            print(f"[HotspotController] Failed to apply HotspotGuard: {e}")

def disable_hotspot_guard():
    """Remove hotspot nftables guard when switching back to home Wi-Fi."""
    try:
        subprocess.run(["sudo", "nft", "delete", "table", "inet", "hotspot_guard"], capture_output=True, timeout=5)
        print("[HotspotController] Removed HotspotGuard.")
    except Exception as e:
        pass

def switch_to_ap(timeout_sec: int = 180):
    """
    Safely switch to AP mode with Fail-Safe Watchdog protection and HotspotGuard.
    """
    def _worker():
        time.sleep(1.0)
        target_sta = find_sta_profile()
        print("[HotspotController] Preparing AP profile: locking to 2.4GHz Channel 6...")
        subprocess.run([
            "sudo", "nmcli", "connection", "modify", AP_PROFILE,
            "802-11-wireless.band", "bg",
            "802-11-wireless.channel", "6"
        ], capture_output=True)

        print(f"[HotspotController] Spawning fail-safe watchdog (timeout={timeout_sec}s)...")
        subprocess.Popen(["sudo", "python3", WATCHDOG_SCRIPT, "--timeout", str(timeout_sec)])
        time.sleep(0.5)

        print("[HotspotController] Atomically bringing up AP connection...")
        res = subprocess.run(["sudo", "nmcli", "connection", "up", AP_PROFILE], capture_output=True, text=True, timeout=25)
        if res.returncode != 0:
            print(f"[HotspotController] AP startup FAILED ({res.stderr.strip()}). Auto-reverting back to {target_sta}...")
            subprocess.run(["sudo", "nmcli", "connection", "up", target_sta], capture_output=True, timeout=25)
        else:
            enable_hotspot_guard()
            print("[HotspotController] AP successfully online and broadcasting Hotspot!")

    threading.Thread(target=_worker, daemon=True).start()

def switch_to_sta():
    """Immediately and cleanly revert back to home Wi-Fi."""
    def _worker():
        disable_hotspot_guard()
        print(f"[HotspotController] Reverting back to {STA_PROFILE}...")
        subprocess.run(["sudo", "python3", WATCHDOG_SCRIPT, "--revert"], capture_output=True, timeout=30)
    threading.Thread(target=_worker, daemon=True).start()

def touch_keepalive():
    """Signal watchdog that active web UI user is connected."""
    subprocess.run(["python3", WATCHDOG_SCRIPT, "--touch"], capture_output=True, timeout=5)
