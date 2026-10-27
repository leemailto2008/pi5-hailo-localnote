#!/usr/bin/env python3
"""合成 LocalNote e2e 測試會議音檔 (16 kHz mono WAV)，並寫回對應 fixture (含實際長度、每句時間軸、sha256)。

流程：meeting_gen.build_meeting() 產生腳本 → edge-tts 逐句合成 (依 voice+text 快取) → 依 fixture 的 gaps 串接
→ 疊加 -55 dBFS 底噪 (避免整段數位零被 ZeroSplicer 視為 PipeWire 插入的靜音) → 量測實際長度；
若超出 [0.88, 1.12]×目標 或超過 MAX_AUDIO_S，依比例調整目標重建腳本 (最多 4 次)。
音檔以串流方式寫出 (只量長度、逐句載入)，4 小時版也只需數十 MB 記憶體。

需求：pip install edge-tts numpy；ffmpeg；可連線 speech.platform.bing.com。
usage:
  gen_test_audio.py --minutes 1 3 5 10 15 30 --fixtures tools/e2e/fixtures --out ~/ln_e2e_data
  gen_test_audio.py --minutes 45 60 90 120 180 240 --out ~/ln_e2e_data      # 長會議 (最長 4 小時)
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
    paths, voices = [], []
    for s, t in meeting["lines"]:
        p = os.path.join(cache_dir, _key(meeting["speakers"][s], t) + ".npy")
        paths.append(p)
        vf = p + ".voice"
        voices.append(open(vf).read().strip() if os.path.exists(vf) else meeting["speakers"][s])
    return paths, voices, len(todo)


def layout(meeting, paths, voices):
    """依 fixture 的 gaps 排出時間軸，不載入音訊 (只讀各句長度)。回傳 (events, timeline, total_samples)；
    events 依序為 ("gap", 樣本數) 或 ("clip", 快取路徑)。"""
    g = meeting["gaps"]
    n = int(SR * g["lead_s"])
    events, timeline, pos, i = [("gap", n)], [], n, 0
    for si, seg in enumerate(meeting["segments"]):
        if si:
            n = int(SR * g["segment_s"]); events.append(("gap", n)); pos += n
        for li, line in enumerate(seg["lines"]):
            if li:
                n = int(SR * g["line_s"]); events.append(("gap", n)); pos += n
            length = np.load(paths[i], mmap_mode="r").shape[0]
            timeline.append({"segment": seg["name"], "spk": line["spk"], "voice": voices[i],
                             "start_s": round(pos / SR, 3), "end_s": round((pos + length) / SR, 3)})
            events.append(("clip", paths[i])); pos += length; i += 1
    n = int(SR * g["tail_s"]); events.append(("gap", n)); pos += n
    return events, timeline, pos


def write_wav(events, wav_path, seed=1):
    """逐段串接 + 增益 + -55 dBFS 底噪並直接寫入 WAV，記憶體只放一句 (4 小時音訊舊做法需約 3 GB)。
    底噪以同一個 Generator 依序產生，結果與舊版整段一次產生逐位元相同 (既有 fixture 的 sha256 不變)。回傳 sha256。"""
    rng = np.random.default_rng(seed)
    scale = 32768 * 10 ** (NOISE_DBFS / 20)
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        for kind, v in events:
            x = np.zeros(v, np.float32) if kind == "gap" else np.load(v).astype(np.float32)
            x = x * GAIN
            x += rng.standard_normal(len(x)).astype(np.float32) * scale
            w.writeframes(np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes())
    h = hashlib.sha256()
    with open(wav_path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_one(minutes, fixtures_dir, out_dir, cache_dir):
    target = min(minutes * 60.0, meeting_gen.MAX_AUDIO_S)
    t, best = target, None
    for it in range(4):
        m = meeting_gen.build_meeting(minutes, target_s=t)
        paths, voices, new = synth_lines(m, cache_dir)
        events, timeline, n_samples = layout(m, paths, voices)
        dur = n_samples / SR
        fits_cap = dur <= meeting_gen.MAX_AUDIO_S
        ok = fits_cap and 0.88 * target <= dur <= 1.12 * target
        print(f"  {minutes:g}min iter{it}: topics={m['n_topics']} lines={len(m['lines'])} new_tts={new} audio={dur:.1f}s target={target:.0f}s {'OK' if ok else 'adjust'}", flush=True)
        if fits_cap and (best is None or abs(dur - target) < abs(best[2] - target)):
            best = (m, (events, timeline), dur)
        if ok:
            break
        t = t * target / dur
    # 短會議受議題顆粒度限制 (1 個議題約 30–40 s)：取最接近目標者；≤1 分鐘容許 ±30%，其餘 ±12%
    tol = 0.30 if minutes <= 1 else 0.12
    if best is None or abs(best[2] - target) > tol * target:
        raise SystemExit(f"could not fit {minutes} min within tolerance")
    m, (events, timeline), dur = best
    wav_name = f"meeting_{minutes:g}min.wav"
    wav_path = os.path.join(out_dir, wav_name)
    sha = write_wav(events, wav_path)
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
    ap.add_argument("--minutes", nargs="+", type=float, default=list(meeting_gen.SHORT_MINUTES),
                    help=f"長會議：{' '.join(map(str, meeting_gen.LONG_MINUTES))} (共用 seed，TTS 快取可共用)")
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
