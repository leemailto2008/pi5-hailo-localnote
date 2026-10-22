#!/usr/bin/env python3
"""1 Hz resource sampler for LocalNote e2e: CPU (total/per-core), CPU temp, RAM/swap, key process CPU%/RSS,
Hailo-10H NPU util/temp/RAM (via app /api/system). Writes JSONL until /tmp/mon.stop exists. usage: mon.py out.jsonl"""
import json, os, sys, time, urllib.request, subprocess
OUT = sys.argv[1]; HZ = os.sysconf("SC_CLK_TCK"); PAGE = os.sysconf("SC_PAGE_SIZE")
NAMES = {"uvicorn": "uvicorn", "hailo-ollama": "hailo-ollama", "ollama": "ollama serve", "ffmpeg": "ffmpeg",
         "pipewire": "pipewire", "wireplumber": "wireplumber"}
def cpu_times():
    out = {}
    for l in open("/proc/stat"):
        if l.startswith("cpu"):
            f = l.split(); v = list(map(int, f[1:8])); out[f[0]] = (sum(v), v[3] + v[4])
    return out
def procs():
    res = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            for k, pat in NAMES.items():
                if pat in cmd and (k != "ollama" or "hailo-ollama" not in cmd) and (k != "pipewire" or "pipewire-pulse" not in cmd):
                    st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
                    rss = int(st[21]) * PAGE; t = int(st[11]) + int(st[12])
                    a = res.setdefault(k, [0, 0]); a[0] += t; a[1] += rss
        except Exception: pass
    return res
def mem():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return {"ram_used_mb": (m["MemTotal"] - m["MemAvailable"]) // 1024, "ram_avail_mb": m["MemAvailable"] // 1024,
            "swap_used_mb": (m["SwapTotal"] - m["SwapFree"]) // 1024}
def npu():
    try:
        d = json.loads(urllib.request.urlopen("http://127.0.0.1/api/system", timeout=1.5).read()).get("npu") or {}
        ram = str(d.get("ram_usage_mb") or "").split("/")[0].strip()
        return {"npu_util": d.get("nnc_util_pct"), "npu_temp": d.get("temp_c"), "npu_ram_mb": float(ram) if ram else None}
    except Exception: return {}
thr0 = subprocess.run("vcgencmd get_throttled", shell=True, capture_output=True, text=True).stdout.strip()
pc, pp, pt = cpu_times(), procs(), time.time()
with open(OUT, "w") as f:
    f.write(json.dumps({"start": pt, "throttled_start": thr0}) + "\n")
    while not os.path.exists("/tmp/mon.stop"):
        time.sleep(1)
        c, p, t = cpu_times(), procs(), time.time(); dt = t - pt
        row = {"t": round(t, 1)}
        for k in c:
            dtot = c[k][0] - pc[k][0]; didle = c[k][1] - pc[k][1]
            row["cpu" if k == "cpu" else k] = round(100 * (1 - didle / dtot), 1) if dtot > 0 else 0
        row["cpu_temp"] = round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000, 1)
        row.update(mem()); row.update(npu())
        for k, (tk, rss) in p.items():
            row[f"{k}_cpu"] = round(100 * (tk - pp.get(k, [tk])[0]) / HZ / dt, 1); row[f"{k}_mb"] = rss // 2**20
        f.write(json.dumps(row) + "\n"); f.flush(); pc, pp, pt = c, p, t
    f.write(json.dumps({"end": time.time(), "throttled_end": subprocess.run("vcgencmd get_throttled", shell=True, capture_output=True, text=True).stdout.strip()}) + "\n")
