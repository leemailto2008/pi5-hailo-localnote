# f:\12_prj_raspi5\code_dispatcher\hotspot_watchdog.py
"""
Fail-Safe Hotspot Watchdog for Raspberry Pi 5.
Ensures that if Pi 5 is switched to standalone Hotspot mode (AP),
it will NEVER get bricked or permanently stranded.
If no client connects within TIMEOUT seconds, it automatically reverts to home Wi-Fi.
"""

import os
import sys
import time
import subprocess
import argparse

WATCHDOG_FLAG_FILE = "/tmp/hotspot_keepalive.flag"
STA_PROFILE = "My_5G_Guest1"
AP_PROFILE = "Pi5-Hotspot"

def has_connected_stations() -> bool:
    """Check if any mobile phone or device is connected to the AP."""
    try:
        res = subprocess.run(["iw", "dev", "wlan0", "station", "dump"], capture_output=True, text=True, timeout=3)
        if "Station " in res.stdout:
            return True
    except Exception:
        pass

    # Check DHCP leases
    lease_paths = [
        "/run/NetworkManager/dnsmasq-wlan0.leases",
        "/var/lib/misc/dnsmasq.leases",
        "/var/lib/NetworkManager/dnsmasq-wlan0.leases"
    ]
    for lp in lease_paths:
        if os.path.exists(lp) and os.path.getsize(lp) > 0:
            return True
    return False

def check_keepalive_flag() -> bool:
    """Check if Web UI or user sent a keepalive signal within the last 60 seconds."""
    if os.path.exists(WATCHDOG_FLAG_FILE):
        mtime = os.path.getmtime(WATCHDOG_FLAG_FILE)
        if time.time() - mtime < 60:
            return True
    return False

def touch_keepalive():
    """Touch keepalive file."""
    with open(WATCHDOG_FLAG_FILE, "w") as f:
        f.write(str(time.time()))

def revert_to_sta():
    """Safely revert Pi 5 back to home Wi-Fi station mode."""
    candidates = ["netplan-wlan0-My_2G_Guest1", "My_5G_Guest1", STA_PROFILE]
    seen = set()
    unique_candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    for prof in unique_candidates:
        print(f"[Watchdog] Reverting Pi 5 back to home Wi-Fi: {prof}...")
        try:
            res = subprocess.run(["sudo", "nmcli", "connection", "up", prof], capture_output=True, text=True, timeout=25)
            if res.returncode == 0:
                print(f"[Watchdog] Successfully connected to {prof}: {res.stdout.strip()}")
                subprocess.run(["sudo", "nft", "delete", "table", "inet", "hotspot_guard"], capture_output=True)
                if os.path.exists(WATCHDOG_FLAG_FILE):
                    os.remove(WATCHDOG_FLAG_FILE)
                return True
        except Exception as e:
            print(f"[Watchdog] Revert error on {prof}: {e}")
    return False

def run_watchdog(timeout_sec: int = 180):
    print(f"[Watchdog] Starting Fail-Safe Hotspot Watchdog with {timeout_sec}s timeout.")
    start_time = time.time()

    # Wait 10 seconds for initial AP stabilization
    time.sleep(10)

    while True:
        elapsed = time.time() - start_time
        remaining = timeout_sec - elapsed

        # Check if connected or keepalive active
        has_client = has_connected_stations()
        has_alive = check_keepalive_flag()

        if has_client or has_alive:
            # Active user connected! Reset countdown or extend
            print(f"[Watchdog] Client active (station={has_client}, keepalive={has_alive}). Watchdog paused.")
            start_time = time.time()  # Reset countdown as long as user is connected
        else:
            print(f"[Watchdog] No client connected. Revert countdown: {remaining:.0f}s remaining...")

        if remaining <= 0:
            print("[Watchdog] Timeout expired with no connected clients! Auto-reverting...")
            revert_to_sta()
            break

        time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in seconds before auto-reverting")
    parser.add_argument("--revert", action="store_true", help="Immediately revert to STA")
    parser.add_argument("--touch", action="store_true", help="Touch keepalive flag")
    args = parser.parse_args()

    if args.revert:
        revert_to_sta()
    elif args.touch:
        touch_keepalive()
    else:
        run_watchdog(args.timeout)
