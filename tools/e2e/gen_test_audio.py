#!/usr/bin/env python3
"""產生 e2e 測試音檔：以 edge-tts 合成 script_meeting.json 的雙人對話，16 kHz mono，疊加 -55 dBFS 底噪模擬麥克風
(避免整段數位零被 ZeroSplicer 視為 PipeWire 插入的靜音而移除)。
需求：pip install edge-tts；ffmpeg。usage: gen_test_audio.py script_meeting.json test_meeting.wav"""
import sys, json, asyncio, subprocess, wave, tempfile, os
import numpy as np
import edge_tts

VOICES = {"F": "zh-TW-HsiaoChenNeural", "M": "zh-TW-YunJheNeural"}
SR = 16000


async def synth(lines, tmp):
    await asyncio.gather(*[edge_tts.Communicate(t, VOICES[w]).save(os.path.join(tmp, f"l{i:02d}.mp3"))
                           for i, (w, t) in enumerate(lines)])


def main(script_path, out_wav):
    lines = json.load(open(script_path, encoding="utf-8"))["lines"]
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(synth(lines, tmp))
        parts = [np.zeros(int(SR * 1.0), np.float32)]
        for i in range(len(lines)):
            raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", os.path.join(tmp, f"l{i:02d}.mp3"),
                                  "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"], capture_output=True, check=True).stdout
            parts += [np.frombuffer(raw, "<i2").astype(np.float32), np.zeros(int(SR * 0.6), np.float32)]
        parts.append(np.zeros(int(SR * 1.5), np.float32))
    x = np.concatenate(parts) * 0.7
    x += np.random.default_rng(1).standard_normal(len(x)).astype(np.float32) * (32768 * 10 ** (-55 / 20))
    y = np.clip(np.round(x), -32768, 32767).astype("<i2")
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(y.tobytes())
    print(f"{out_wav}: {len(y) / SR:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
