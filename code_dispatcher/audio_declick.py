# code_dispatcher/audio_declick.py
"""
回放用的跳音修平 (declick)，只處理「回放」檔，不影響 STT 與原始錄音。

背景 (2026-11-03 使用者測試，228，上班時段)：藍牙 SCO 封包到達率 94.4%。沒收到的封包不會補零，
音訊直接接上，接縫處就是一個階梯跳變；解碼錯誤的幀則是幾毫秒的爆音。回放時聽起來是一連串「喀」聲與雜訊。
缺掉的音訊補不回來，這裡只把跳變處 (相鄰樣本差 > 附近平均差 12 倍且 > 3000) 前後約 1.5 ms 改成平滑銜接，
爆音型 (該處能量明顯高於前後) 再加長到約 4 ms 並壓低。5 分鐘錄音約改動 0.3% 的樣本。

make_smooth_ogg() 在錄音結束、編碼完成後於背景執行：ffmpeg 解碼 → declick → ffmpeg 以 nice 編碼成 *.smooth.ogg。
"""
import os
import subprocess
from typing import Tuple

import numpy as np


def _events(x: np.ndarray, ratio: float, floor: float, gap: int):
    d = np.abs(np.diff(x))
    local = np.convolve(d, np.ones(400, dtype=np.float32) / 400, mode="same") + 1.0
    idx = np.flatnonzero((d > ratio * local) & (d > floor))
    if len(idx) == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > gap) + 1
    return [(int(g[0]), int(g[-1]) + 1) for g in np.split(idx, splits)]


def declick(pcm: np.ndarray, sr: int = 16000, ratio: float = 12.0, floor: float = 3000.0,
            half_ms: float = 1.5, burst_ms: float = 4.0) -> Tuple[np.ndarray, int]:
    """int16 單聲道 → (修平後 int16, 修了幾處)。"""
    x = pcm.astype(np.float32)
    n = len(x)
    if n < 1000:
        return pcm.copy(), 0
    w, wb = max(4, int(sr * half_ms / 1000)), max(8, int(sr * burst_ms / 1000))
    y = x.copy()
    ev = _events(x, ratio, floor, gap=2 * w)
    for a, b in ev:
        ctx = 4 * wb
        if a - ctx < 0 or b + ctx >= n:
            continue
        mid = x[a - wb:b + wb]
        side = np.concatenate((x[a - ctx:a - wb], x[b + wb:b + ctx]))
        burst = np.sqrt((mid ** 2).mean()) > 2.0 * (np.sqrt((side ** 2).mean()) + 1.0)
        h = wb if burst else w
        s, e = a - h, b + h
        ramp = np.linspace(y[s], y[e], e - s + 1, dtype=np.float32)
        if burst:                      # 爆音那段保留一點原始內容 (壓低 12 dB)，聽起來比完全換成直線自然
            ramp = ramp + 0.25 * (x[s:e + 1] - np.linspace(x[s], x[e], e - s + 1, dtype=np.float32))
        y[s:e + 1] = ramp
    return np.clip(np.round(y), -32768, 32767).astype(np.int16), len(ev)


def smooth_path(ogg_path: str) -> str:
    root, ext = os.path.splitext(ogg_path)
    return f"{root}.smooth{ext or '.ogg'}"


def make_smooth_ogg(src: str, dst: str = "", sr: int = 16000, bitrate: str = "64k") -> Tuple[bool, int]:
    """原始錄音 → 修平順的回放檔。回傳 (成功與否, 修了幾處)；失敗時不留半成品。
    位元率與錄音母帶相同 (voice_pipeline 的 encoder 是 libopus 64k)，比較兩者時只差在修平，不差在編碼品質。"""
    dst = dst or smooth_path(src)
    tmp = dst + ".part"
    try:
        raw = subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-i", src, "-ac", "1", "-ar", str(sr),
                              "-f", "s16le", "-"], capture_output=True, timeout=600, check=True).stdout
        y, n = declick(np.frombuffer(raw, dtype="<i2"), sr=sr)
        subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1",
                        "-i", "-", "-c:a", "libopus", "-b:a", bitrate, "-f", "ogg", tmp],
                       input=y.astype("<i2").tobytes(), capture_output=True, timeout=600, check=True)
        os.replace(tmp, dst)
        return True, n
    except Exception as e:
        print(f"[Declick] smooth playback file failed: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False, 0


# ---- 背景產生：獨立行程 (不佔服務的 GIL)，新錄音開始時會被停掉 (錄音中不做運算) ----
_bg_proc = None


def start_smooth_job(src: str):
    """錄音檔完成後在背景產生 *.smooth.ogg；前一個還在跑就先停掉。"""
    global _bg_proc
    stop_smooth_job()
    try:
        _bg_proc = subprocess.Popen(["nice", "-n", "10", "python3", os.path.abspath(__file__), src],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[Declick] cannot start smooth job: {e}")
        _bg_proc = None


def stop_smooth_job():
    global _bg_proc
    if _bg_proc is not None and _bg_proc.poll() is None:
        try:
            _bg_proc.kill()
        except Exception:
            pass
    _bg_proc = None


if __name__ == "__main__":
    import sys
    ok, n = make_smooth_ogg(sys.argv[1])
    print(f"[Declick] {'ok' if ok else 'failed'}: {n} clicks smoothed -> {smooth_path(sys.argv[1])}")
