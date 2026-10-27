#!/usr/bin/env python3
"""LocalNote 全流程 e2e (單一案例)：錄音 → STT → Summary，量測品質、延遲與資源。在 Pi 上、app 閒置時執行。

流程：建立 PipeWire null sink → 寫入 /dev/shm/ln_force_source 讓 app 改錄 sink monitor
→ POST /api/record (預設 bypass_summary=false，只作用於本次，不改全域設定) → ffmpeg -re 即時播放測試音檔
→ 輪詢狀態至 done/error → 以 scoring.py 評分；mon.py 1 Hz 取樣資源。結束一定清除注入檔與 null sink。

usage:
  e2e.py <wav> <fixture.json> [label] [--allow-bypass] [--out /tmp]
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scoring  # noqa: E402

API = "http://127.0.0.1"
SINK = "ln_test_sink"
FORCE = "/dev/shm/ln_force_source"
NOISE_RE = re.compile(r"HailoRT|Hailo-10H runtime initialization failed|Filtered hallucination")
ERR_RE = re.compile(r"Traceback|Exception|Error|error|failed")


def sh(c, t=60):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout


def api(path, body=None, t=15):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    return json.loads(urllib.request.urlopen(req, timeout=t).read())


def wait_idle(timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if api("/api/status").get("state") not in ("recording", "processing"):
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def _resources(rows, t0, rec_end, total, head, tail):
    def sel(a, b):
        return [x for x in rows if t0 + a <= x["t"] < t0 + b]

    def stat(xs, k, f):
        v = [x[k] for x in xs if isinstance(x.get(k), (int, float))]
        return round(f(v), 1) if v else None

    rec = sel(0, rec_end) if rec_end else []
    allr = sel(0, total + 5)
    avg = lambda v: sum(v) / len(v)
    return {
        "rec_cpu_avg": stat(rec, "cpu", avg), "rec_cpu_max": stat(rec, "cpu", max),
        "rec_core_max": max((max(x.get(f"cpu{i}", 0) for i in range(4)) for x in rec), default=None),
        "cpu_avg": stat(allr, "cpu", avg), "cpu_temp_max": stat(allr, "cpu_temp", max),
        "ram_peak_mb": stat(allr, "ram_used_mb", max), "swap_max_mb": stat(allr, "swap_used_mb", max),
        "swap_start_mb": allr[0].get("swap_used_mb") if allr else None,
        "swap_growth_mb": (stat(allr, "swap_used_mb", max) or 0) - (allr[0].get("swap_used_mb") or 0) if allr else None,
        "npu_util_max": stat(allr, "npu_util", max), "npu_temp_max": stat(allr, "npu_temp", max),
        "npu_ram_max_mb": stat(allr, "npu_ram_mb", max), "uvicorn_rss_max_mb": stat(allr, "uvicorn_mb", max),
        "throttled_start": head.get("throttled_start"), "throttled_end": tail.get("throttled_end"),
    }


def run_case(wav, fixture_path, label, bypass_summary=False, out_dir="/tmp"):
    fixture = json.load(open(fixture_path, encoding="utf-8"))
    dur = float(sh(f"ffprobe -v error -show_entries format=duration -of csv=p=0 '{wav}'").strip())
    st = api("/api/status")
    if st.get("state") in ("recording", "processing"):
        raise RuntimeError(f"app busy: {st.get('state')}")
    sh(f"pw-cli destroy {SINK} 2>/dev/null")
    sh("pw-cli create-node adapter '{ factory.name=support.null-audio-sink node.name=%s media.class=Audio/Sink "
       "audio.position=[MONO] audio.rate=16000 object.linger=true }'" % SINK)
    time.sleep(1)
    mon_path = os.path.join(out_dir, f"mon_{label}.jsonl")
    sh("rm -f /tmp/mon.stop")
    mon = subprocess.Popen([sys.executable, os.path.join(HERE, "mon.py"), mon_path])
    time.sleep(3)
    t0 = time.time()
    jt0 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job, phases, play, rec_end = None, {}, None, None
    try:
        with open(FORCE, "w") as f:
            f.write(SINK + ".monitor")
        job = api("/api/record", {"seconds": int(dur + 4), "bypass_summary": bool(bypass_summary)}).get("job")
        time.sleep(1.5)
        play = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-re", "-i", wav, "-f", "pulse", "-device", SINK, "ln_e2e"])
        last = None
        while time.time() - t0 < dur + 1500:
            s = api("/api/status")
            ph = (s.get("state"), s.get("phase"))
            if ph != last:
                phases[f"{ph[0]}/{ph[1]}"] = round(time.time() - t0, 1)
                if last and last[0] == "recording" and ph[0] != "recording":
                    rec_end = round(time.time() - t0, 1)
                last = ph
            if s.get("state") in ("done", "error", "idle") and time.time() - t0 > 5:
                break
            time.sleep(1)
        if play:
            play.wait(timeout=60)
    finally:
        try:
            os.remove(FORCE)
        except FileNotFoundError:
            pass
        sh(f"pw-cli destroy {SINK} 2>/dev/null")
        time.sleep(3)
        open("/tmp/mon.stop", "w").close()
        mon.wait(timeout=15)
        if play and play.poll() is None:
            play.kill()
    total = round(time.time() - t0, 1)
    jt1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s, r = api("/api/status"), api("/api/result")
    transcript = s.get("transcript") or r.get("transcript") or ""
    summary = s.get("summary_stream") or scoring.summary_body(r.get("markdown") or "")
    log = sh(f"journalctl -u pi5-dispatcher-api --since '{jt0}' --until '{jt1}' --no-pager", 60)
    errors = [l[l.find("]: ") + 3:][:200] for l in log.splitlines() if ERR_RE.search(l) and not NOISE_RE.search(l)]
    rows = [json.loads(l) for l in open(mon_path)]
    head, tail_ = rows[0], rows[-1]
    rows = [x for x in rows if "t" in x]
    m = s.get("metrics") or {}
    res = {
        "label": label, "job": job, "state": s.get("state"), "msg": s.get("msg"), "bypass_summary": bool(bypass_summary),
        "minutes": fixture.get("minutes"), "audio_s": round(dur, 1), "total_s": total, "rec_end_s": rec_end, "phases_t": phases,
        "summary_latency_s": m.get("total_latency") if m.get("llm_status") == "done" else None,
        "metrics": {k: m.get(k) for k in ("first_chunk_time", "first_stt_latency", "stt_tail_wait", "stt_chars", "llm_status",
                                          "llm_backend", "llm_duration", "llm_tokens", "llm_tps", "total_latency", "total_elapsed")},
        "stt_npu_chunks": log.count("Hailo-10H 轉錄完成"), "stt_cpu_chunks": log.count("Zero-Disk Transcribed text"),
        "hailo_busy_errors": log.count("HailoWhisper] Hailo-10H runtime initialization failed"),
        "llm_fallbacks": log.count("trying next backend"),
        "zero_splice": [l[l.find("removed"):][:120] for l in log.splitlines() if "zero-splice" in l],
        "errors": errors[:20],
        "resources": _resources(rows, t0, rec_end, total, head, tail_),
        "transcript_eval": scoring.transcript_checks(transcript, fixture),
        "summary_eval": None if bypass_summary else scoring.summary_checks(summary, fixture),
        "transcript": transcript, "summary": summary,
    }
    with open(os.path.join(out_dir, f"e2e_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav"); ap.add_argument("fixture"); ap.add_argument("label", nargs="?", default="run")
    ap.add_argument("--allow-bypass", action="store_true", help="以 bypass_summary=true 執行 (只測錄音與 STT)")
    ap.add_argument("--out", default="/tmp")
    a = ap.parse_args()
    res = run_case(a.wav, a.fixture, a.label, bypass_summary=a.allow_bypass, out_dir=a.out)
    print(json.dumps({k: v for k, v in res.items() if k not in ("transcript", "summary")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
