#!/usr/bin/env python3
"""合成 LocalNote e2e 測試會議音檔 (16 kHz mono WAV)，並寫回對應 fixture (含實際長度、每句時間軸、sha256)。

流程：meeting_gen.build_meeting() 產生腳本 → edge-tts 逐句合成 (依 voice+text 快取) → 依 fixture 的 gaps 串接
→ 疊加 -55 dBFS 底噪 (避免整段數位零被 ZeroSplicer 視為 PipeWire 插入的靜音) → 量測實際長度；
若超出 [0.88, 1.12]×目標 或超過 MAX_AUDIO_S，依比例調整目標重建腳本 (最多 4 次)。

需求：pip install edge-tts numpy；ffmpeg；可連線 speech.platform.bing.com。
usage:
  gen_test_audio.py --minutes 1 3 5 10 15 30 --fixtures tools/e2e/fixtures --out ~/ln_e2e_data
"""
import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meeting_gen  # noqa: E402

SR = 16000
NOISE_DBFS = -55.0
GAIN = 0.7


VOICE_FALLBACK = {   # edge-tts 偶發單一聲線限流 ("No audio was received")：同聲線重試 3 次後改用備援聲線
    "zh-TW-YunJheNeural": ["zh-CN-YunxiNeural"],
    "zh-TW-HsiaoChenNeural": ["zh-TW-HsiaoYuNeural"],
    "zh-TW-HsiaoYuNeural": ["zh-TW-HsiaoChenNeural"],
}


def _key(voice, text):
    return hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()[:16]


async def _synth_one(sem, voice, text, path):
    """合成一句並存成 .npy；回傳實際使用的聲線 (同時寫入 path + '.voice')。"""
    import edge_tts
    async with sem:
        last = None
        for v in [voice] + VOICE_FALLBACK.get(voice, []):
            for attempt in range(3):
                try:
                    with tempfile.TemporaryDirectory() as td:
                        tmp = os.path.join(td, "line.mp3")
                        await edge_tts.Communicate(text, v).save(tmp)
                        raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", tmp, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                                             capture_output=True, check=True).stdout
                    np.save(path, np.frombuffer(raw, "<i2"))
                    with open(path + ".voice", "w") as f:
                        f.write(v)
                    return v
                except Exception as e:
                    last = e
                    await asyncio.sleep(3 * (attempt + 1))
        raise RuntimeError(f"TTS failed for {text[:20]}…: {last}")


def synth_lines(meeting, cache_dir, concurrency=3):
    os.makedirs(cache_dir, exist_ok=True)
    todo = []
    for spk, text in meeting["lines"]:
        voice = meeting["speakers"][spk]
        p = os.path.join(cache_dir, _key(voice, text) + ".npy")
        if not os.path.exists(p):
            todo.append((voice, text, p))
    if todo:
        async def run():
            sem = asyncio.Semaphore(concurrency)
            await asyncio.gather(*[_synth_one(sem, v, t, p) for v, t, p in todo])
        asyncio.run(run())
    clips, voices = [], []
    for s, t in meeting["lines"]:
        p = os.path.join(cache_dir, _key(meeting["speakers"][s], t) + ".npy")
        clips.append(np.load(p))
        vf = p + ".voice"
        voices.append(open(vf).read().strip() if os.path.exists(vf) else meeting["speakers"][s])
    return clips, voices, len(todo)


def assemble(meeting, clips, voices, seed=1):
    g = meeting["gaps"]
    parts, timeline, pos, i = [np.zeros(int(SR * g["lead_s"]), np.float32)], [], int(SR * g["lead_s"]), 0
    for si, seg in enumerate(meeting["segments"]):
        if si:
            parts.append(np.zeros(int(SR * g["segment_s"]), np.float32)); pos += int(SR * g["segment_s"])
        for li, line in enumerate(seg["lines"]):
            if li:
                parts.append(np.zeros(int(SR * g["line_s"]), np.float32)); pos += int(SR * g["line_s"])
            c = clips[i].astype(np.float32); i += 1
            timeline.append({"segment": seg["name"], "spk": line["spk"], "voice": voices[i - 1],
                             "start_s": round(pos / SR, 3), "end_s": round((pos + len(c)) / SR, 3)})
            parts.append(c); pos += len(c)
    parts.append(np.zeros(int(SR * g["tail_s"]), np.float32))
    x = np.concatenate(parts) * GAIN
    x += np.random.default_rng(seed).standard_normal(len(x)).astype(np.float32) * (32768 * 10 ** (NOISE_DBFS / 20))
    return np.clip(np.round(x), -32768, 32767).astype("<i2"), timeline


def build_one(minutes, fixtures_dir, out_dir, cache_dir):
    target = min(minutes * 60.0, meeting_gen.MAX_AUDIO_S)
    t, best = target, None
    for it in range(4):
        m = meeting_gen.build_meeting(minutes, target_s=t)
        clips, voices, new = synth_lines(m, cache_dir)
        y, timeline = assemble(m, clips, voices)
        dur = len(y) / SR
        fits_cap = dur <= meeting_gen.MAX_AUDIO_S
        ok = fits_cap and 0.88 * target <= dur <= 1.12 * target
        print(f"  {minutes:g}min iter{it}: topics={m['n_topics']} lines={len(m['lines'])} new_tts={new} audio={dur:.1f}s target={target:.0f}s {'OK' if ok else 'adjust'}", flush=True)
        if fits_cap and (best is None or abs(dur - target) < abs(best[2] - target)):
            best = (m, (y, timeline), dur)
        if ok:
            break
        t = t * target / dur
    # 短會議受議題顆粒度限制 (1 個議題約 30–40 s)：取最接近目標者；≤1 分鐘容許 ±30%，其餘 ±12%
    tol = 0.30 if minutes <= 1 else 0.12
    if best is None or abs(best[2] - target) > tol * target:
        raise SystemExit(f"could not fit {minutes} min within tolerance")
    m, (y, timeline), dur = best
    wav_name = f"meeting_{minutes:g}min.wav"
    wav_path = os.path.join(out_dir, wav_name)
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(y.tobytes())
    sha = hashlib.sha256(open(wav_path, "rb").read()).hexdigest()
    try:
        import edge_tts
        ver = getattr(edge_tts, "__version__", "unknown")
    except Exception:
        ver = "unknown"
    m["audio"] = {"file": wav_name, "duration_s": round(dur, 2), "sr": SR, "noise_dbfs": NOISE_DBFS, "gain": GAIN,
                  "sha256": sha, "tts": f"edge-tts {ver}"}
    m["timeline"] = timeline
    fx_path = os.path.join(fixtures_dir, f"meeting_{minutes:g}min.json")
    with open(fx_path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    return {"minutes": minutes, "wav": wav_path, "fixture": fx_path, "duration_s": round(dur, 1), "topics": m["n_topics"], "sha256": sha[:12]}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", nargs="+", type=float, default=[1, 3, 5, 10, 15, 30])
    ap.add_argument("--fixtures", default=os.path.join(here, "fixtures"))
    ap.add_argument("--out", required=True, help="WAV 輸出目錄 (不進 git)")
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "ln_tts"), help="逐句 TTS 快取目錄")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.fixtures, exist_ok=True)
    cache = a.cache
    res = [build_one(m, a.fixtures, a.out, cache) for m in a.minutes]
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for r in res:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
