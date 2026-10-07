# f:\12_prj_raspi5\gpio_driver\test_key_led.py
"""
Diagnostic & Hardware Verification Utility for 5-Key and 5-LED on Raspberry Pi 5.
"""

import time
import argparse
from gpiozero import Button, LED

MAPPING = [
    {"name": "KEY1/LED1", "key": 6,  "led": 26, "k_pin": 31, "l_pin": 37},
    {"name": "KEY2/LED2", "key": 5,  "led": 16, "k_pin": 29, "l_pin": 36},
    {"name": "KEY3/LED3", "key": 22, "led": 25, "k_pin": 15, "l_pin": 22},
    {"name": "KEY4/LED4", "key": 27, "led": 24, "k_pin": 13, "l_pin": 18},
    {"name": "KEY5/LED5", "key": 17, "led": 23, "k_pin": 11, "l_pin": 16},
]

def test_leds():
    print("\n--- [1] LED Sequential Test (LED1 -> LED5) ---")
    leds = [LED(item["led"]) for item in MAPPING]
    try:
        for idx, (led, item) in enumerate(zip(leds, MAPPING), 1):
            print(f"Lighting {item['name']} (GPIO{item['led']}, Header Pin {item['l_pin']})...")
            led.on()
            time.sleep(0.5)
            led.off()
            time.sleep(0.2)
        print("All LEDs verified successfully!\n")
    finally:
        for l in leds: l.close()

def monitor_keys(duration_sec=30):
    print(f"\n--- [2] Real-time Key Monitoring ({duration_sec} seconds) ---")
    print("Press any key (KEY1~KEY5) on the board now. Press Ctrl+C to stop.")
    buttons = [Button(item["key"], pull_up=True, bounce_time=0.05) for item in MAPPING]
    leds = [LED(item["led"]) for item in MAPPING]

    try:
        start = time.time()
        while time.time() - start < duration_sec:
            states = []
            for b, l, item in zip(buttons, leds, MAPPING):
                pressed = b.is_pressed
                if pressed:
                    l.on()
                    states.append(f"{item['name']}: [DOWN-ON]")
                else:
                    l.off()
                    states.append(f"{item['name']}: [UP-OFF]")
            print("\r" + " | ".join(states), end="", flush=True)
            time.sleep(0.08)
        print("\nMonitoring duration completed.")
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    finally:
        for b in buttons: b.close()
        for l in leds:
            l.off()
            l.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Key-LED Hardware Diagnostics")
    parser.add_argument("--led-only", action="store_true", help="Only run LED chase test")
    parser.add_argument("--duration", type=int, default=30, help="Monitoring duration in seconds")
    args = parser.parse_args()

    test_leds()
    if not args.led_only:
        monitor_keys(args.duration)
