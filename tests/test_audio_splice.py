"""audio_splice：串流版 ZeroSplicer 必須與批次參考實作逐樣本一致；pause_cut_index 必須落在停頓內。"""
import numpy as np
import pytest

from audio_splice import ZeroSplicer, pause_cut_index


def splice_reference(x, sr=16000, min_ms=5.0, thr=2, xfade_ms=3.0):
    q = (np.abs(x.astype(np.int32)) <= thr).astype(np.int8)
    d = np.diff(np.concatenate([[0], q, [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (en - st) >= int(sr * min_ms / 1000)
    st, en = st[keep], en[keep]
    if len(st) == 0:
        return x.copy(), 0, 0
    xf, segs, prev = int(sr * xfade_ms / 1000), [], 0
    for s, e in zip(st, en):
        segs.append(x[prev:s].astype(np.float32)); prev = e
    segs.append(x[prev:].astype(np.float32))
    y = segs[0]
    for seg in segs[1:]:
        n = min(xf, len(y), len(seg))
        if n > 0:
            w = np.linspace(0, 1, n, dtype=np.float32)
            y = np.concatenate([y[:-n], y[-n:] * (1 - w) + seg[:n] * w, seg[n:]])
        else:
            y = np.concatenate([y, seg])
    return np.clip(np.round(y), -32768, 32767).astype(np.int16), int((en - st).sum()), int(len(st))


@pytest.mark.parametrize("trial", range(80))
def test_stream_matches_reference(trial):
    rng = np.random.default_rng(trial)
    n = int(rng.integers(2000, 40000))
    x = (rng.standard_normal(n) * rng.choice([30, 300, 3000])).astype(np.int16)
    for _ in range(int(rng.integers(0, 10))):           # PipeWire 插入式零段 (含邊界、恰好門檻)
        s = int(rng.integers(0, n)); x[s:s + int(rng.choice([40, 79, 80, 81, 656, 1350]))] = 0
    for _ in range(int(rng.integers(0, 6))):            # 真實音訊中的短暫近零
        s = int(rng.integers(0, n)); x[s:s + int(rng.integers(1, 79))] = rng.integers(-2, 3)
    ref, removed, runs = splice_reference(x)
    zs, raw, out, i = ZeroSplicer(), x.tobytes(), bytearray(), 0
    while i < len(raw):                                 # 隨機 (常為奇數) 塊大小
        k = int(rng.integers(1, 5000)); out += zs.feed(raw[i:i + k]); i += k
    out += zs.flush()
    y = np.frombuffer(bytes(out), "<i2")
    assert len(y) == len(ref)
    if len(y):
        assert int(np.max(np.abs(y.astype(int) - ref.astype(int)))) <= 1
    assert (zs.runs, zs.removed_samples) == (runs, removed)


def test_no_zero_runs_is_passthrough():
    x = (np.random.default_rng(1).standard_normal(16000) * 1000).astype(np.int16)
    zs = ZeroSplicer()
    y = np.frombuffer(zs.feed(x.tobytes()) + zs.flush(), "<i2")
    assert np.array_equal(x, y) and zs.runs == 0


def test_pause_cut_lands_in_pause():
    sr = 16000
    x = (np.random.default_rng(3).standard_normal(sr * 10) * 3000).astype(np.int16)
    x[int(6.3 * sr):int(6.6 * sr)] = (np.random.default_rng(4).standard_normal(int(0.3 * sr)) * 20).astype(np.int16)
    c = pause_cut_index(x, sr, 4.0, 9.5)
    assert 6.3 * sr <= c <= 6.6 * sr


def test_pause_cut_short_buffer_returns_end():
    assert pause_cut_index(np.zeros(16000 * 3, np.int16), 16000, 4.0, 9.5) == 16000 * 3
