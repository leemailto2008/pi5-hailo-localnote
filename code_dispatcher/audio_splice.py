# code_dispatcher/audio_splice.py
"""
串流式數位靜音剪接器 (Streaming Zero-Run Splicer)

背景 (2026-10-22 實測, raspi678, btsnoop + libsbc ground truth 比對):
PipeWire 1.4.x 的 bluez5 sco-source 在 mSBC 幀解碼失敗時直接丟棄該幀 (原始碼 `TODO: PLC`)，
解碼緩衝區 (decode buffer) 隨即 underrun 並進入 re-buffering，期間每個 graph cycle 都輸出整段數位零，
形成約 40 ms (常見 40/84/126… ms) 的「插入式」靜音；之後的音訊完整接續 (相位延後)，實際遺失的只有
壞掉的那一幀 (7.5 ms)。PipeWire 1.6 起才內建 mSBC PLC。

本模組在編碼/STT 之前移除這些插入的數位零段 (|x| <= thr 且持續 >= min_ms)，並以短交叉淡化接縫，
還原連續語音。真實麥克風底噪不會出現連續 5 ms 的 |x|<=2，故不會誤刪真實內容。
"""
import numpy as np


class ZeroSplicer:
    """逐塊餵入 16-bit LE mono PCM bytes，輸出剪除插入式數位零段後的 PCM bytes (固定延遲 xfade_ms)。"""

    def __init__(self, sr: int = 16000, min_ms: float = 5.0, thr: int = 2, xfade_ms: float = 3.0):
        self.min_len = max(1, int(sr * min_ms / 1000))
        self.thr = thr
        self.xf = max(1, int(sr * xfade_ms / 1000))
        self.sr = sr
        self._rem = b""                                  # 奇數 byte 殘留
        self._zbuf = np.zeros(0, dtype=np.int16)         # 尚未達門檻的近零樣本
        self._in_run = False                             # 正在剔除的零段中
        self._tail = np.zeros(0, dtype=np.float32)       # 保留供交叉淡化的尾端樣本
        self._post = np.zeros(0, dtype=np.float32)       # 零段之後、等待交叉淡化的樣本
        self._need_xf = False
        self.removed_samples = 0
        self.runs = 0

    def feed(self, data: bytes) -> bytes:
        data = self._rem + data
        n = len(data) // 2 * 2
        self._rem = data[n:]
        if n == 0:
            return b""
        a = np.frombuffer(data[:n], dtype="<i2")
        small = np.abs(a.astype(np.int32)) <= self.thr
        cuts = np.flatnonzero(np.diff(small.astype(np.int8))) + 1
        bounds = np.concatenate(([0], cuts, [len(a)]))
        for s, e in zip(bounds[:-1], bounds[1:]):
            seg = a[s:e]
            if small[s]:
                if self._in_run:
                    self.removed_samples += len(seg)
                    continue
                self._zbuf = np.concatenate((self._zbuf, seg))
                if len(self._zbuf) >= self.min_len:
                    if self._need_xf:                    # 前一個接縫的淡化區不足 xfade 長度，以現有樣本完成
                        self._finish_xf()
                    self._in_run = True
                    self._need_xf = True
                    self.runs += 1
                    self.removed_samples += len(self._zbuf)
                    self._zbuf = self._zbuf[:0]
            else:
                if len(self._zbuf):                      # 短暫近零：屬於真實音訊，原樣保留
                    self._keep(self._zbuf.astype(np.float32))
                    self._zbuf = self._zbuf[:0]
                self._keep(seg.astype(np.float32))
                self._in_run = False
        return self._emit()

    def flush(self) -> bytes:
        if len(self._zbuf) and not self._in_run:
            self._keep(self._zbuf.astype(np.float32))
        self._zbuf = self._zbuf[:0]
        if self._need_xf:
            self._finish_xf()
        out, self._tail = self._tail, self._tail[:0]
        return self._to_bytes(out)

    @property
    def removed_ms(self) -> float:
        return self.removed_samples * 1000.0 / self.sr

    def _keep(self, seg: np.ndarray):
        if self._need_xf:
            self._post = np.concatenate((self._post, seg))
            if len(self._post) >= self.xf:
                self._finish_xf()
        else:
            self._tail = np.concatenate((self._tail, seg))

    def _finish_xf(self):
        k = min(self.xf, len(self._tail), len(self._post))
        if k > 0:
            w = np.linspace(0.0, 1.0, k, dtype=np.float32)
            mixed = self._tail[-k:] * (1.0 - w) + self._post[:k] * w
            self._tail = np.concatenate((self._tail[:-k], mixed, self._post[k:]))
        else:
            self._tail = np.concatenate((self._tail, self._post))
        self._post = self._post[:0]
        self._need_xf = False

    def _emit(self) -> bytes:
        if len(self._tail) <= self.xf:
            return b""
        out, self._tail = self._tail[:-self.xf], self._tail[-self.xf:]
        return self._to_bytes(out)

    @staticmethod
    def _to_bytes(x: np.ndarray) -> bytes:
        return np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()


def pause_cut_index(samples: np.ndarray, sr: int = 16000, min_s: float = 4.0, max_s: float = 9.5,
                    win_ms: float = 200.0, hop_ms: float = 20.0) -> int:
    """在 [min_s, max_s] 範圍內找能量最低的 win_ms 視窗，回傳其中心的樣本索引，作為 STT 切片切點。

    2026-10-22 e2e 實測 (raspi678, Hailo-10H Whisper-Small)：固定每 8 秒硬切常落在句中，Whisper 從半句開始解碼時
    容易提早輸出 EOT，整段只剩一兩個字 (例：8 秒內容只轉出「第一」)。改為停頓處切點後，轉出字數 243→271/277。
    """
    lo, hi = int(min_s * sr), min(int(max_s * sr), len(samples))
    w, h = max(1, int(win_ms * sr / 1000)), max(1, int(hop_ms * sr / 1000))
    if hi - lo < w:
        return hi
    seg = samples[lo:hi].astype(np.float64)
    energy = np.convolve(seg * seg, np.ones(w) / w, mode="valid")[::h]
    return lo + int(np.argmin(energy)) * h + w // 2
