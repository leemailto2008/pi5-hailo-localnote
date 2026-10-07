# f:\12_prj_raspi5\gpio_driver\key_led_driver.py
"""
Raspberry Pi 5 (RP1) 5-Key & 5-LED Hardware Driver & Daemon.

Hardware Pinout Mapping (BCM GPIO & 40-Pin Header):
  KEY1: GPIO6  (Pin 31) [Active Low, Pull-Up] <---> LED1: GPIO26 (Pin 37) [Active High, 330Ω]
  KEY2: GPIO5  (Pin 29) [Active Low, Pull-Up] <---> LED2: GPIO16 (Pin 36) [Active High, 330Ω]
  KEY3: GPIO22 (Pin 15) [Active Low, Pull-Up] <---> LED3: GPIO25 (Pin 22) [Active High, 330Ω]
  KEY4: GPIO27 (Pin 13) [Active Low, Pull-Up] <---> LED4: GPIO24 (Pin 18) [Active High, 330Ω]
  KEY5: GPIO17 (Pin 11) [Active Low, Pull-Up] <---> LED5: GPIO23 (Pin 16) [Active High, 330Ω]

Special Function:
  - KEY5 & LED5 are dedicated Hotspot Controllers:
    * Click KEY5: Toggle Wi-Fi between Home Wi-Fi (STA) and Standalone Hotspot (AP).
    * LED5 Solid ON:  Hotspot is ACTIVE.
    * LED5 Solid OFF: Hotspot is INACTIVE (STA / Wi-Fi connected).
    * LED5 Blinking (0.5s interval): Transition / Switching in progress.
"""

import sys
import os
import time
import signal
import argparse
import logging
import threading
import subprocess
from typing import List, Dict, Any, Optional
import requests
from gpiozero import Button, LED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("KeyLedDriver")

# PIN Mapping Configuration Table
PIN_MAPPING: List[Dict[str, Any]] = [
    {
        "id": 1,
        "key_pin": 6,       # 40-Pin Header: Pin 31
        "led_pin": 26,      # 40-Pin Header: Pin 37
        "key_header": 31,
        "led_header": 37,
        "desc": "KEY1 <-> LED1"
    },
    {
        "id": 2,
        "key_pin": 5,       # 40-Pin Header: Pin 29
        "led_pin": 16,      # 40-Pin Header: Pin 36
        "key_header": 29,
        "led_header": 36,
        "desc": "KEY2 <-> LED2"
    },
    {
        "id": 3,
        "key_pin": 22,      # 40-Pin Header: Pin 15
        "led_pin": 25,      # 40-Pin Header: Pin 22
        "key_header": 15,
        "led_header": 22,
        "desc": "KEY3 <-> LED3"
    },
    {
        "id": 4,
        "key_pin": 27,      # 40-Pin Header: Pin 13
        "led_pin": 24,      # 40-Pin Header: Pin 18
        "key_header": 13,
        "led_header": 18,
        "desc": "KEY4 <-> LED4"
    },
    {
        "id": 5,
        "key_pin": 17,      # 40-Pin Header: Pin 11
        "led_pin": 23,      # 40-Pin Header: Pin 16
        "key_header": 11,
        "led_header": 16,
        "desc": "KEY5 <-> LED5 (Hotspot Controller)"
    }
]


def check_is_hotspot_active() -> bool:
    """Check if the active connection on wlan0 is an Access Point / Hotspot."""
    try:
        res = subprocess.run(
            ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device"],
            capture_output=True,
            text=True,
            timeout=3
        )
        for line in res.stdout.splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 4 and parts[0] == "wlan0":
                conn = parts[3]
                if "Hotspot" in conn or "ap" in conn.lower() or "raspi543" in conn.lower():
                    return True
                return False
    except Exception as e:
        logger.warning(f"Failed to query network status: {e}")
    return False


class KeyLedPair:
    """Encapsulates a standard paired Key (Button) and LED (KEY1 ~ KEY4)."""

    def __init__(self, cfg: Dict[str, Any], mode: str = "follow", bounce_time: float = 0.05):
        self.cfg = cfg
        self.id = cfg["id"]
        self.mode = mode
        self.is_toggled_on = False

        self.button = Button(
            cfg["key_pin"],
            pull_up=True,
            bounce_time=bounce_time
        )
        self.led = LED(cfg["led_pin"], initial_value=False)

        if self.mode == "toggle":
            self.button.when_pressed = self._on_toggle_press
        else:
            self.button.when_pressed = self._on_follow_press
            self.button.when_released = self._on_follow_release

    def _on_follow_press(self):
        self.led.on()
        logger.info(f"🟢 [PRESSED]  {self.cfg['desc']} (GPIO{self.cfg['key_pin']} -> GPIO{self.cfg['led_pin']}) => LED ON")

    def _on_follow_release(self):
        self.led.off()
        logger.info(f"⚪ [RELEASED] {self.cfg['desc']} (GPIO{self.cfg['key_pin']} -> GPIO{self.cfg['led_pin']}) => LED OFF")

    def _on_toggle_press(self):
        self.is_toggled_on = not self.is_toggled_on
        if self.is_toggled_on:
            self.led.on()
            logger.info(f"💡 [TOGGLED ON]  {self.cfg['desc']} => LED ON")
        else:
            self.led.off()
            logger.info(f"🌑 [TOGGLED OFF] {self.cfg['desc']} => LED OFF")

    def close(self):
        try:
            self.led.off()
            self.led.close()
            self.button.close()
        except Exception as e:
            logger.warning(f"Error closing {self.cfg['desc']}: {e}")


class HotspotKeyLedPair:
    """
    Dedicated controller for KEY5 & LED5:
    - Button: Click to toggle Hotspot (AP <-> STA)
    - LED:
      * Solid ON: Hotspot is active
      * Solid OFF: Hotspot is inactive (STA/WiFi)
      * Blinking (0.5s interval): Transitioning/Switching in progress
    """

    def __init__(self, cfg: Dict[str, Any], bounce_time: float = 0.05):
        self.cfg = cfg
        self.id = cfg["id"]
        self.is_transitioning = False
        self._lock = threading.Lock()
        self._running = True

        self._blink_thread: Optional[threading.Thread] = None
        self._blink_stop_event = threading.Event()

        # Hardware GPIO binding
        self.button = Button(
            cfg["key_pin"],
            pull_up=True,
            bounce_time=bounce_time
        )
        self.led = LED(cfg["led_pin"], initial_value=False)

        # Connect button press event
        self.button.when_pressed = self._on_button_pressed

        # Initial LED state based on current network status
        self._update_led_state_sync()

        # Start background state sync thread
        self._sync_thread = threading.Thread(target=self._background_sync_loop, daemon=True)
        self._sync_thread.start()

    def _update_led_state_sync(self):
        """Update LED solid state immediately based on live hotspot status."""
        active = check_is_hotspot_active()
        if active:
            self.led.on()
            logger.info("📡 [HOTSPOT STATUS] Active -> LED5 SOLID ON")
        else:
            self.led.off()
            logger.info("🌐 [HOTSPOT STATUS] Inactive (STA) -> LED5 SOLID OFF")

    def _start_blinking(self):
        """Start 0.5s blinking thread for transition indication."""
        self._blink_stop_event.clear()

        def _blink_worker():
            state = True
            while not self._blink_stop_event.is_set():
                if state:
                    self.led.on()
                else:
                    self.led.off()
                state = not state
                # Check stop event every 0.05s to ensure quick termination
                for _ in range(10):
                    if self._blink_stop_event.is_set():
                        break
                    time.sleep(0.05)

        self._blink_thread = threading.Thread(target=_blink_worker, daemon=True)
        self._blink_thread.start()
        logger.info("⏳ [HOTSPOT TRANSITION] Started 0.5s LED5 blinking...")

    def _stop_blinking(self):
        """Stop blinking thread safely."""
        self._blink_stop_event.set()
        if self._blink_thread and self._blink_thread.is_alive():
            self._blink_thread.join(timeout=1.0)
        self._blink_thread = None

    def _on_button_pressed(self):
        """Hardware interrupt callback when KEY5 is pressed."""
        with self._lock:
            if self.is_transitioning:
                logger.warning("⚠️ [KEY5 PRESSED] Hotspot transition currently in progress! Ignoring duplicate press.")
                return
            self.is_transitioning = True

        threading.Thread(target=self._execute_toggle_workflow, daemon=True).start()

    def _execute_toggle_workflow(self):
        """Asynchronous workflow to switch network mode while blinking LED5 at 0.5s."""
        logger.info("🔘 [KEY5 PRESSED] Initiating Hotspot Toggle workflow...")
        self._start_blinking()

        try:
            current_is_ap = check_is_hotspot_active()
            target_mode = "sta" if current_is_ap else "ap"
            logger.info(f"🔄 Switching Wi-Fi: Currently {'AP (Hotspot)' if current_is_ap else 'STA (WiFi)'} => Target: {target_mode.upper()}")

            if target_mode == "ap":
                logger.info("📡 Starting AP mode (Pi5-Hotspot, raspi543_AI)...")
                # 1. Ensure 2.4GHz Channel 6 for radio compatibility
                subprocess.run([
                    "sudo", "nmcli", "connection", "modify", "Pi5-Hotspot",
                    "802-11-wireless.band", "bg",
                    "802-11-wireless.channel", "6"
                ], capture_output=True, timeout=5)
                # 2. Spawn fail-safe watchdog (180s auto-revert if no client)
                subprocess.Popen([
                    "sudo", "python3", "/home/pi/code_dispatcher/hotspot_watchdog.py", "--timeout", "180"
                ])
                time.sleep(0.5)
                # 3. Bring up AP connection
                res = subprocess.run(["sudo", "nmcli", "connection", "up", "Pi5-Hotspot"], capture_output=True, text=True, timeout=25)
                logger.info(f"[nmcli up Pi5-Hotspot] code={res.returncode}, out={res.stdout.strip()}, err={res.stderr.strip()}")
                if res.returncode == 0:
                    subprocess.run(["sudo", "nft", "-f", "/etc/nftables.hotspot_guard.nft"], capture_output=True)
                    logger.info("🛡️ [HotspotGuard] Kernel firewall applied: Only 10.20.0.1 services allowed, background junk dropped.")
            else:
                logger.info("🌐 Reverting back to home Wi-Fi...")
                subprocess.run(["sudo", "nft", "delete", "table", "inet", "hotspot_guard"], capture_output=True)
                res = subprocess.run(["sudo", "python3", "/home/pi/code_dispatcher/hotspot_watchdog.py", "--revert"], capture_output=True, text=True, timeout=25)
                logger.info(f"[revert to STA] code={res.returncode}, out={res.stdout.strip()}, err={res.stderr.strip()}")

            # 4. Poll and wait for network state transition
            target_achieved = False
            for step in range(20):
                time.sleep(0.8)
                now_is_ap = check_is_hotspot_active()
                if (target_mode == "ap" and now_is_ap) or (target_mode == "sta" and not now_is_ap):
                    logger.info(f"✅ [HOTSPOT SWITCH] Successfully transitioned to {target_mode.upper()} in ~{(step + 1) * 0.8:.1f}s!")
                    target_achieved = True
                    break

            if not target_achieved:
                logger.warning(f"⚠️ [HOTSPOT SWITCH] Timed out waiting for mode {target_mode.upper()} to become active.")

        except Exception as e:
            logger.error(f"❌ [HOTSPOT SWITCH ERROR] Exception during toggle: {e}", exc_info=True)

        finally:
            # 4. Stop blinking and settle LED state
            self._stop_blinking()
            final_is_ap = check_is_hotspot_active()
            if final_is_ap:
                self.led.on()
                logger.info("💡 [HOTSPOT READY] Final State: AP -> LED5 SOLID ON")
            else:
                self.led.off()
                logger.info("🌑 [HOTSPOT READY] Final State: STA -> LED5 SOLID OFF")

            with self._lock:
                self.is_transitioning = False

    def _background_sync_loop(self):
        """Keep LED5 synchronized with real-time network state even when switched via Web or Watchdog."""
        while self._running:
            try:
                time.sleep(2.0)
                with self._lock:
                    if self.is_transitioning:
                        continue

                is_ap = check_is_hotspot_active()
                if is_ap and not self.led.is_lit:
                    self.led.on()
                    logger.info("🔄 [SYNC] External switch to AP detected -> LED5 ON")
                elif not is_ap and self.led.is_lit:
                    self.led.off()
                    logger.info("🔄 [SYNC] External switch to STA detected -> LED5 OFF")
            except Exception as e:
                logger.debug(f"Sync loop error: {e}")

    def close(self):
        """Release GPIO lines and stop background threads."""
        self._running = False
        self._stop_blinking()
        try:
            self.led.off()
            self.led.close()
            self.button.close()
        except Exception as e:
            logger.warning(f"Error closing HotspotKeyLedPair: {e}")


class KeyLedDriver:
    """Manages all Key-LED peripherals with self-test and hardware daemon loop."""

    def __init__(self, mode: str = "follow", bounce_time: float = 0.05):
        self.mode = mode
        self.bounce_time = bounce_time
        self.standard_pairs: List[KeyLedPair] = []
        self.hotspot_pair: Optional[HotspotKeyLedPair] = None
        self._running = True

    def initialize(self):
        logger.info("=========================================================")
        logger.info(f"Initializing Raspberry Pi 5 Key-LED Driver (Mode: {self.mode})")
        logger.info("KEY1~KEY4: Standard GPIO Keys | KEY5: Hotspot Toggle (LED5 0.5s Blink/Solid)")
        logger.info("=========================================================")

        # Register KEY1 ~ KEY4
        for cfg in PIN_MAPPING[:4]:
            pair = KeyLedPair(cfg, mode=self.mode, bounce_time=self.bounce_time)
            self.standard_pairs.append(pair)
            logger.info(f"  Registered {cfg['desc']}: Key=Pin{cfg['key_header']}(GPIO{cfg['key_pin']}), LED=Pin{cfg['led_header']}(GPIO{cfg['led_pin']})")

        # Register KEY5 / LED5 as dedicated Hotspot Controller
        hotspot_cfg = PIN_MAPPING[4]
        self.hotspot_pair = HotspotKeyLedPair(hotspot_cfg, bounce_time=self.bounce_time)
        logger.info(f"  Registered {hotspot_cfg['desc']}: Key=Pin{hotspot_cfg['key_header']}(GPIO{hotspot_cfg['key_pin']}), LED=Pin{hotspot_cfg['led_header']}(GPIO{hotspot_cfg['led_pin']})")

        logger.info("All 5 Key-LED peripherals registered successfully!")

    def self_test(self, cycles: int = 1):
        """Power-On Self-Test (POST): Chase pattern across all 5 LEDs to verify wiring."""
        logger.info(f"Running Power-On Self-Test (Chase Pattern x {cycles})...")
        all_leds = [p.led for p in self.standard_pairs]
        if self.hotspot_pair:
            all_leds.append(self.hotspot_pair.led)

        for _ in range(cycles):
            for led in all_leds:
                led.on()
                time.sleep(0.10)
                led.off()
            for led in reversed(all_leds):
                led.on()
                time.sleep(0.08)
                led.off()

        # All blink together once
        for led in all_leds:
            led.on()
        time.sleep(0.2)
        for led in all_leds:
            led.off()

        # Reset LED5 to live network status
        if self.hotspot_pair:
            self.hotspot_pair._update_led_state_sync()

        logger.info("Self-test complete. Driver is ready for input events.")

    def run(self):
        """Daemon event loop waiting for hardware interrupts."""
        logger.info("Ready! KEY1~KEY4 control LED1~LED4. KEY5 toggles Hotspot (LED5 blink 0.5s -> solid).")
        logger.info("Press Ctrl+C to terminate.")
        while self._running:
            time.sleep(1.0)

    def shutdown(self):
        """Gracefully shutdown and release all pins."""
        logger.info("Shutting down Key-LED driver, releasing GPIO...")
        self._running = False
        for pair in self.standard_pairs:
            pair.close()
        if self.hotspot_pair:
            self.hotspot_pair.close()
        logger.info("Driver cleanup finished cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 5-Key & 5-LED Driver with Hotspot Toggle")
    parser.add_argument(
        "--mode",
        choices=["follow", "toggle"],
        default="follow",
        help="Control mode for KEY1~KEY4: 'follow' (default) or 'toggle'"
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=0.05,
        help="Debounce filter time in seconds (default: 0.05s = 50ms)"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip power-on LED chase self-test"
    )
    args = parser.parse_args()

    driver = KeyLedDriver(mode=args.mode, bounce_time=args.debounce)

    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, terminating gracefully...")
        driver.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        driver.initialize()
        if not args.no_test:
            driver.self_test(cycles=1)
        driver.run()
    except Exception as e:
        logger.error(f"Fatal error in driver: {e}", exc_info=True)
        driver.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
