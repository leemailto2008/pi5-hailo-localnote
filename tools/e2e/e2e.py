#!/usr/bin/env python3
"""LocalNote e2e: play a known-script WAV into a PipeWire null sink, force the app to record its monitor,
run the full pipeline via the HTTP API, then score transcript (CER) and summary (fact recall), timings, NPU usage.
usage: e2e.py test.wav script.json [label]"""
import json, subprocess, sys, time, datetime, re, unicodedata, urllib.request
WAV, SCRIPT = sys.argv[1], sys.argv[2]; LABEL = sys.argv[3] if len(sys.argv) > 3 else "run"
API = "http://127.0.0.1"; SINK = "ln_test_sink"
def sh(c, t=60): return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout
def api(path, body=None, t=10):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    return json.loads(urllib.request.urlopen(req, timeout=t).read())
script = json.load(open(SCRIPT)); ref = "".join(t for _, t in script["lines"])
dur = float(sh(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {WAV}").strip())
st = api("/api/status")
if st.get("state") in ("recording", "processing"): print(json.dumps({"error": "app busy", "state": st.get("state")})); sys.exit(1)
cfg_bypass = api("/api/config/bypass_summary").get("bypass_summary")
# virtual source
sh(f"pw-cli destroy {SINK} 2>/dev/null")
sh("pw-cli create-node adapter '{ factory.name=support.null-audio-sink node.name=%s media.class=Audio/Sink "
   "audio.position=[MONO] audio.rate=16000 object.linger=true }'" % SINK); time.sleep(1)
open("/dev/shm/ln_force_source", "w").write(SINK + ".monitor")
sh("rm -f /tmp/mon.stop"); mon = subprocess.Popen(["python3", __import__("os").path.dirname(__file__) + "/mon.py", f"/tmp/mon_{LABEL}.jsonl"])
time.sleep(3)                                          # idle baseline samples
t0 = time.time(); jt0 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
try:
    job = api("/api/record", {"seconds": int(dur + 4)}).get("job")
    time.sleep(1.5)                                   # let ffmpeg attach to the monitor
    play = subprocess.Popen(f"ffmpeg -nostdin -v error -re -i {WAV} -f pulse -device {SINK} ln_e2e", shell=True)
    last = None; phases = {}
    while time.time() - t0 < dur + 600:
        s = api("/api/status"); ph = (s.get("state"), s.get("phase"))
        if ph != last: phases[f"{ph[0]}/{ph[1]}"] = round(time.time() - t0, 1); last = ph
        if s.get("state") in ("done", "error", "idle") and time.time() - t0 > 5: break
        time.sleep(1)
    play.wait(timeout=30)
finally:
    try: __import__("os").remove("/dev/shm/ln_force_source")
    except FileNotFoundError: pass
    sh(f"pw-cli destroy {SINK} 2>/dev/null")
    time.sleep(3); open("/tmp/mon.stop", "w").close(); mon.wait(timeout=10)
total = round(time.time() - t0, 1); jt1 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
s = api("/api/status"); r = api("/api/result")
def norm(t):
    try:
        import opencc; t = opencc.OpenCC("s2twp").convert(t)
    except Exception: pass
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "N"))
def cer(a, b):
    a, b = norm(a), norm(b); prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1): cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb))
        prev = cur
    return round(prev[-1] / max(1, len(a)), 3)
hyp = s.get("transcript") or r.get("transcript") or ""
summary = (r.get("markdown") or "") + "\n" + (s.get("summary_stream") or "")
def hit(text):
    nz = lambda z: z.replace(" ", "").replace("臺", "台").replace("％", "%")
    t = nz(text); return {k: any(nz(v) in t for v in vs) for k, vs in script["facts"].items()}
fs, ft = hit(summary), hit(hyp)
log = sh(f"journalctl -u pi5-dispatcher-api --since '{jt0}' --until '{jt1}' --no-pager", 30)
rows = [json.loads(l) for l in open(f"/tmp/mon_{LABEL}.jsonl")]
head, tail_ = rows[0], rows[-1]; rows = [x for x in rows if "t" in x]
def seg(a, b):
    sel = [x for x in rows if t0 + a <= x["t"] < t0 + b]
    if not sel: return None
    def st(k):
        v = [x[k] for x in sel if isinstance(x.get(k), (int, float))]
        return [round(sum(v) / len(v), 1), round(max(v), 1)] if v else None
    keys = ["cpu", "cpu_temp", "ram_used_mb", "swap_used_mb", "npu_util", "npu_temp", "npu_ram_mb"] + \
           [f"{p}_{m}" for p in ("uvicorn", "hailo-ollama", "ollama", "ffmpeg", "pipewire", "wireplumber") for m in ("cpu", "mb")]
    out = {k: st(k) for k in keys}; out["core_max"] = max(max(x.get(f"cpu{i}", 0) for i in range(4)) for x in sel)
    return {k: v for k, v in out.items() if v is not None}
pts = sorted(phases.values()); bounds = [(-3, 0, "idle_before")]
names = sorted(phases, key=phases.get)
for i, n in enumerate(names): bounds.append((phases[n], phases[names[i+1]] if i + 1 < len(names) else total, n))
resources = {"throttled_start": head.get("throttled_start"), "throttled_end": tail_.get("throttled_end"),
             "by_phase(avg,max)": {n: seg(a, b) for a, b, n in bounds}}
res = {"label": LABEL, "job": job, "state": s.get("state"), "bypass_summary_cfg": cfg_bypass, "audio_s": round(dur, 1), "total_s": total,
       "phases_t": phases, "cer": cer(ref, hyp), "transcript_chars": len(hyp),
       "facts_in_transcript": f"{sum(ft.values())}/{len(ft)}", "facts_in_summary": f"{sum(fs.values())}/{len(fs)}",
       "missing_in_summary": [k for k, v in fs.items() if not v],
       "hailo_stt_ok": log.count("Hailo-10H 轉錄完成"), "hailo_busy_errors": log.count("HailoWhisper] Hailo-10H runtime initialization failed"),
       "llm_backend": (s.get("metrics") or {}).get("llm_backend"), "metrics": {k: (s.get("metrics") or {}).get(k) for k in
       ("first_chunk_time", "first_stt_latency", "stt_tail_wait", "llm_status", "llm_duration", "llm_tokens", "llm_tps", "total_latency", "total_elapsed")},
       "errors": [l[l.find("]: ") + 3:][:160] for l in log.splitlines() if re.search(r"Traceback|Error|error|failed|Exception", l) and "HailoRT" not in l and "Hailo-10H runtime" not in l][:8],
       "resources": resources, "transcript": hyp, "summary": summary.strip()[:3000]}
json.dump(res, open(f"/tmp/e2e_{LABEL}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("transcript", "summary", "resources")}, ensure_ascii=False))
print("RESOURCES", json.dumps(resources, ensure_ascii=False))
