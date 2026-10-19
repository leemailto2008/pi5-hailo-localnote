# f:\13_prj_raspi5_localnote\gpio_driver\test_key_led_interactive.py
"""
Interactive Diagnostic & Hardware Verification for 5-Key & 5-LED on Raspberry Pi 5.
Runs LED sequence test, then continuously logs all button press and release events.
"""

import sys
import time
import argparse
from datetime import datetime
from gpiozero import Button, LED

MAPPING = [
    {"id": 1, "name": "KEY1 <-> LED1", "key": 6,  "led": 26, "k_pin": 31, "l_pin": 37},
    {"id": 2, "name": "KEY2 <-> LED2", "key": 5,  "led": 16, "k_pin": 29, "l_pin": 36},
    {"id": 3, "name": "KEY3 <-> LED3", "key": 22, "led": 25, "k_pin": 15, "l_pin": 22},
    {"id": 4, "name": "KEY4 <-> LED4", "key": 27, "led": 24, "k_pin": 13, "l_pin": 18},
    {"id": 5, "name": "KEY5 <-> LED5", "key": 17, "led": 23, "k_pin": 11, "l_pin": 16},
]

def test_leds():
    print("\n=========================================================")
    print(" 💡 [階段一] LED 依序點亮測試 (LED1 -> LED5)")
    print("=========================================================")
    leds = [LED(item["led"]) for item in MAPPING]
    try:
        for item, led in zip(MAPPING, leds):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 點亮 LED{item['id']} (Header Pin {item['l_pin']} / BCM GPIO{item['led']})...", flush=True)
            led.on()
            time.sleep(0.4)
            led.off()
            time.sleep(0.15)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 全體 5 顆 LED 同時閃爍 2 次...", flush=True)
        for _ in range(2):
            for l in leds: l.on()
            time.sleep(0.25)
            for l in leds: l.off()
            time.sleep(0.2)
        print("✅ LED 測試完畢！全體 LED 正常。\n", flush=True)
    finally:
        for l in leds: l.close()

def monitor_keys(duration_sec=300):
    print("=========================================================")
    print(f" 🔘 [階段二] 實時按鍵事件監聽中 (持續 {duration_sec} 秒)")
    print(" 請在板子上按下任意按鍵 (KEY1 ~ KEY5)，按下時對應 LED 會同步亮起！")
    print(" 按下 Ctrl+C 可隨時提前結束。")
    print("=========================================================\n")

    buttons = []
    leds = []
    press_counts = {item["id"]: 0 for item in MAPPING}

    for item in MAPPING:
        b = Button(item["key"], pull_up=True, bounce_time=0.05)
        l = LED(item["led"])

        def make_handlers(target_item, target_led):
            def on_pressed():
                target_led.on()
                press_counts[target_item["id"]] += 1
                now_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                print(f"🟢 [{now_str}] >>> 按下【KEY {target_item['id']}】(Pin {target_item['k_pin']} / GPIO{target_item['key']}) -> 點亮 LED {target_item['id']} (累計: {press_counts[target_item['id']]} 次)", flush=True)

            def on_released():
                target_led.off()
                now_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                print(f"⚪ [{now_str}] <<< 放開【KEY {target_item['id']}】 -> 熄滅 LED {target_item['id']}", flush=True)

            return on_pressed, on_released

        h_press, h_release = make_handlers(item, l)
        b.when_pressed = h_press
        b.when_released = h_release

        buttons.append(b)
        leds.append(l)

    start_time = time.time()
    try:
        while time.time() - start_time < duration_sec:
            time.sleep(0.2)
        print("\n時間到，監聽結束。")
    except KeyboardInterrupt:
        print("\n使用者手動中斷。")
    finally:
        for b in buttons: b.close()
        for l in leds:
            l.off()
            l.close()

    print("\n=========================================================")
    print(" 📊 各按鍵按下統計結果：")
    for item in MAPPING:
        print(f"  KEY {item['id']} (Pin {item['k_pin']}): 按下 {press_counts[item['id']]} 次")
    print("=========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 5-Key & 5-LED Interactive Test")
    parser.add_argument("--duration", type=int, default=300, help="監聽時長 (秒，預設 300 秒)")
    parser.add_argument("--skip-led", action="store_true", help="跳過開頭的 LED 測試")
    args = parser.parse_args()

    if not args.skip_led:
        test_leds()
    monitor_keys(args.duration)
