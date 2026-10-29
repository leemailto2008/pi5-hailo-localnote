#!/usr/bin/env python3
"""喇叭 → 藍牙耳麥的聲學自測 (2026-10-29)。

在接了喇叭的電腦上播放 e2e 測試會議，同時呼叫 Pi 的 /api/record 錄音，錄完用 fixture 評分。
e2e.py 是數位注入 (null sink)；這支測的是真實路徑：空氣傳播 → 耳麥麥克風 → 藍牙 mSBC → ZeroSplicer → STT → 紀要。
只用標準函式庫：Windows 用 winsound 播放，其他平台依序找 ffplay / afplay / aplay。

usage:
  python tools/e2e/speaker_selftest.py --pi http://192.168.50.228 --wav D:/ln_e2e_data/meeting_1min.wav
  python tools/e2e/speaker_selftest.py --pi http://192.168.50.228 --wav meeting_3min.wav --out result_3min.json

事前準備：
  - 耳麥先連上 Pi，放在喇叭前；喇叭音量調到正常說話大小 (錄音 RMS 約 -25 dBFS，不要削波)。
  - WAV 用 gen_test_audio.py 產生，sha256 要與 fixture 的 audio.sha256 相同 (不同時只警告)。
  - 若改用瀏覽器自動化播放，必須是有畫面的模式：Chromium 無頭模式 (headless) 會自動加 --mute-audio，喇叭沒有聲音。
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scoring  # noqa: E402


def api(base: str, path: str, body=None, timeout: float = 10.0):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wav_seconds(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def default_fixture(wav_path: str) -> str:
    m = re.search(r"meeting_(\d+)min", os.path.basename(wav_path))
    if not m:
        raise SystemExit("無法從檔名判斷 fixture，請用 --fixture 指定")
    return os.path.join(HERE, "fixtures", f"meeting_{m.group(1)}min.json")


def player_command(wav_path: str):
    """回傳阻塞式播放的函式；--player none 時不播放 (只驗證流程)。"""
    if sys.platform.startswith("win"):
        import winsound
        return lambda: winsound.PlaySound(wav_path, winsound.SND_FILENAME)
    for exe, args in (("ffplay", ["-nodisp", "-autoexit", "-loglevel", "error"]), ("afplay", []), ("aplay", ["-q"])):
        if shutil.which(exe):
            return lambda: subprocess.run([exe, *args, wav_path], check=False)
    raise SystemExit("找不到播放程式 (Windows 用 winsound；其他平台需要 ffplay / afplay / aplay)")


def wait_state(base: str, ok, timeout: float, poll: float = 0.25):
    t_end = time.time() + timeout
    while time.time() < t_end:
        st = api(base, "/api/status")
        if ok(st):
            return st
        time.sleep(poll)
    raise SystemExit(f"等待逾時 ({timeout:.0f} s)，目前狀態：{st.get('state')} / {st.get('msg')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pi", default="http://192.168.50.228", help="Pi 的 app 網址")
    ap.add_argument("--wav", required=True, help="測試會議 WAV (meeting_{N}min.wav)")
    ap.add_argument("--fixture", default=None, help="預設依檔名找 fixtures/meeting_{N}min.json")
    ap.add_argument("--lead", type=float, default=1.5, help="錄音開始後再等幾秒才播放 (等 SCO 連線穩定)")
    ap.add_argument("--player", default="auto", choices=("auto", "none"), help="none = 不播放，只驗證 API 流程")
    ap.add_argument("--bypass-summary", action="store_true", help="只錄音 + STT，不產生紀要")
    ap.add_argument("--out", default=None, help="把結果與評分存成 JSON")
    a = ap.parse_args()

    fx_path = a.fixture or default_fixture(a.wav)
    fx = json.load(open(fx_path, encoding="utf-8"))
    want = (fx.get("audio") or {}).get("sha256", "")
    got = hashlib.sha256(open(a.wav, "rb").read()).hexdigest()
    if want and not got.startswith(want):
        print(f"[warn] WAV sha256 {got[:12]} 與 fixture {want[:12]} 不同，評分可能不準", file=sys.stderr)
    dur = wav_seconds(a.wav)
    play = (lambda: time.sleep(dur)) if a.player == "none" else player_command(a.wav)

    st = api(a.pi, "/api/status")
    if st.get("state") in ("recording", "processing"):
        print(f"app 忙碌中 ({st.get('state')})，請稍後再試", file=sys.stderr)
        return 75
    seconds = int(math.ceil(dur + a.lead + 6))
    job = api(a.pi, "/api/record", {"seconds": seconds, "bypass_summary": bool(a.bypass_summary)}).get("job")
    print(f"[selftest] job {job}：錄 {seconds} s，播放 {os.path.basename(a.wav)} ({dur:.1f} s)", flush=True)
    wait_state(a.pi, lambda s: s.get("state") == "recording" and (s.get("elapsed") or 0) > 0, 60)
    time.sleep(a.lead)
    t0 = time.time()
    play()
    print(f"[selftest] 播放結束 ({time.time() - t0:.1f} s)，等待轉錄與紀要…", flush=True)
    wait_state(a.pi, lambda s: s.get("state") in ("done", "error", "idle") and s.get("job") in (job, None), seconds + 1800, poll=2)
    res = api(a.pi, "/api/result")
    tr = res.get("transcript", "")
    summ = scoring.summary_body(res.get("markdown") or "")     # 去掉匯出 markdown 附帶的「## 逐字稿」，與 e2e.py 相同
    report = {"job": res.get("job"), "wav": os.path.basename(a.wav), "fixture": os.path.basename(fx_path),
              "transcript": scoring.transcript_checks(tr, fx),
              "summary": None if a.bypass_summary else scoring.summary_checks(summ, fx, tr)}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(dict(report, transcript_text=tr, summary_markdown=summ), f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
