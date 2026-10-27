#!/usr/bin/env python3
"""LocalNote HIL 自測套件：依序跑多個長度的測試會議，套用門檻 (thresholds.json)，輸出 JSON / Markdown / JUnit XML。

usage (在 Pi 上，app 閒置時):
  run_suite.py --minutes 1 3                      # quick (CI 每次)
  run_suite.py --minutes 1 3 5 10 15 30           # full (nightly)
  run_suite.py --minutes 5 --allow-bypass         # 只測錄音 + STT
exit code：任一門檻失敗 → 1 (--no-fail 時一律 0)
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from xml.sax.saxutils import quoteattr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2e      # noqa: E402
import scoring  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_junit(path, cases):
    total = sum(len(c["checks"]) for c in cases)
    fails = sum(sum(not k["passed"] for k in c["checks"]) for c in cases)
    out = [f'<?xml version="1.0" encoding="UTF-8"?>', f'<testsuites name="localnote-e2e" tests="{total}" failures="{fails}">']
    for c in cases:
        nf = sum(not k["passed"] for k in c["checks"])
        out.append(f'  <testsuite name="e2e.{c["minutes"]:g}min" tests="{len(c["checks"])}" failures="{nf}" time="{c.get("total_s", 0)}">')
        for k in c["checks"]:
            out.append(f'    <testcase classname="e2e.{c["minutes"]:g}min" name={quoteattr(k["check"])}>')
            if not k["passed"]:
                msg = "value=%s limit=%s" % (k["value"], k["limit"])
                out.append(f'      <failure message={quoteattr(msg)}/>')
            out.append("    </testcase>")
        out.append("  </testsuite>")
    out.append("</testsuites>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def write_markdown(path, cases, meta):
    L = [f"# LocalNote e2e 報告 · {meta['started']}", "", f"- 主機：{meta['host']}　commit：{meta.get('commit') or '-'}", ""]
    L += ["| 長度 | 結果 | CER | 逐字稿事實 | 總結事實 | 議題覆蓋 | summary 延遲 | STT NPU | 錄音 CPU | 最高溫 | RAM 峰值 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        r = c.get("result") or {}
        t, s, res = r.get("transcript_eval") or {}, r.get("summary_eval") or {}, r.get("resources") or {}
        ok = "✅" if all(k["passed"] for k in c["checks"]) else "❌"
        npu = f"{r.get('stt_npu_chunks', 0)}/{r.get('stt_npu_chunks', 0) + r.get('stt_cpu_chunks', 0)}"
        L.append(f"| {c['minutes']:g} 分 | {ok} | {t.get('cer', '-')} | {t.get('fact_recall', '-')} | {s.get('recap_recall', '-')} | "
                 f"{s.get('topic_coverage', '-')} | {r.get('summary_latency_s', '-')} s | {npu} | {res.get('rec_cpu_avg', '-')}% | "
                 f"{res.get('cpu_temp_max', '-')} °C | {res.get('ram_peak_mb', '-')} MB |")
    for c in cases:
        bad = [k for k in c["checks"] if not k["passed"]]
        if bad:
            L += ["", f"## ❌ {c['minutes']:g} 分鐘失敗項目", ""] + [f"- `{k['check']}`：{k['value']}（門檻 {k['limit']}）" for k in bad]
            errs = (c.get("result") or {}).get("errors") or []
            if errs:
                L += ["", "app log 錯誤（前 5 筆）：", ""] + [f"    {e}" for e in errs[:5]]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", nargs="+", type=float, default=[1, 3])
    ap.add_argument("--data", default=os.environ.get("LN_E2E_DATA", "/home/pi/ln_e2e/data"), help="測試 WAV 目錄")
    ap.add_argument("--fixtures", default=os.path.join(HERE, "fixtures"))
    ap.add_argument("--thresholds", default=os.path.join(HERE, "thresholds.json"))
    ap.add_argument("--out", default=None, help="報告目錄 (預設 ./e2e_reports/<timestamp>)")
    ap.add_argument("--junit", default=None)
    ap.add_argument("--allow-bypass", action="store_true")
    ap.add_argument("--no-fail", action="store_true")
    ap.add_argument("--gap", type=float, default=15.0, help="案例間等待秒數 (讓 NPU 釋放)")
    a = ap.parse_args()
    started = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = a.out or os.path.join(os.getcwd(), "e2e_reports", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    cfg = json.load(open(a.thresholds, encoding="utf-8"))
    meta = {"started": started, "host": os.uname().nodename, "commit": os.environ.get("CI_COMMIT_SHORT_SHA") or os.environ.get("GITHUB_SHA", "")[:8]}
    cases = []
    for i, m in enumerate(a.minutes):
        fx = os.path.join(a.fixtures, f"meeting_{m:g}min.json")
        wav = os.path.join(a.data, f"meeting_{m:g}min.wav")
        case = {"minutes": m, "fixture": fx, "wav": wav, "checks": []}
        th = scoring.thresholds_for(cfg, m)
        if a.allow_bypass:
            th["summary_required"] = False
        try:
            meta_fx = json.load(open(fx, encoding="utf-8"))
            want = (meta_fx.get("audio") or {}).get("sha256")
            if not os.path.exists(wav):
                raise FileNotFoundError(f"missing {wav} (先執行 gen_test_audio.py)")
            if want and sha256(wav) != want:
                raise RuntimeError(f"WAV sha256 與 fixture 不符：{wav}")
            if not e2e.wait_idle():
                raise RuntimeError("app 一直處於 recording/processing")
            print(f"[suite] {m:g} min ({meta_fx.get('audio', {}).get('duration_s')} s) ...", flush=True)
            res = e2e.run_case(wav, fx, f"{m:g}min", bypass_summary=a.allow_bypass, out_dir=out)
            case["result"], case["total_s"] = res, res.get("total_s", 0)
            case["checks"] = scoring.evaluate(res, th)
        except Exception as e:
            case["checks"] = [{"check": "case_executed", "value": f"{type(e).__name__}: {e}", "limit": "no exception", "passed": False}]
        nf = sum(not k["passed"] for k in case["checks"])
        print(f"[suite] {m:g} min → {'PASS' if nf == 0 else f'FAIL ({nf})'}", flush=True)
        cases.append(case)
        if i + 1 < len(a.minutes):
            time.sleep(a.gap)
    slim = [{k: v for k, v in c.items() if k != "result"} | {"result": {k: v for k, v in (c.get("result") or {}).items()
                                                                          if k not in ("transcript", "summary")}} for c in cases]
    with open(os.path.join(out, "suite.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "cases": slim}, f, ensure_ascii=False, indent=1)
    write_markdown(os.path.join(out, "summary.md"), cases, meta)
    write_junit(a.junit or os.path.join(out, "junit.xml"), cases)
    failed = sum(any(not k["passed"] for k in c["checks"]) for c in cases)
    print(f"[suite] report: {out}  failed cases: {failed}/{len(cases)}", flush=True)
    sys.exit(0 if a.no_fail or failed == 0 else 1)


if __name__ == "__main__":
    main()
