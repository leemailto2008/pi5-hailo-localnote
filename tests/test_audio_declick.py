import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code_dispatcher"))
import audio_declick  # noqa: E402


def _tone(sec=1.0, sr=16000, f=300.0, amp=8000.0):
    t = np.arange(int(sec * sr)) / sr
    return amp * np.sin(2 * np.pi * f * t)


def test_clean_audio_is_left_alone():
    x = _tone().astype(np.int16)
    y, n = audio_declick.declick(x)
    assert n == 0 and np.array_equal(x, y)


def test_dropped_packet_step_is_smoothed():
    # 藍牙掉一個封包 (7.5 ms = 120 樣本) 後音訊直接接上：在相位相差半週期處接起來，形成大跳變
    x = _tone(2.0)
    cut = 16013                                                         # 波峰處 (1/4 週期)
    spliced = np.concatenate((x[:cut], x[cut + 27:])).astype(np.int16)   # 27 樣本 ≈ 300 Hz 半週期
    jump = np.abs(np.diff(spliced.astype(np.float32)))[cut - 2:cut + 2].max()
    y, n = audio_declick.declick(spliced)
    assert n == 1
    assert np.abs(np.diff(y.astype(np.float32)))[cut - 40:cut + 40].max() < 0.5 * jump
    changed = np.flatnonzero(y != spliced)
    assert changed.min() > cut - 80 and changed.max() < cut + 80                # 只動接縫附近


def test_smooth_path():
    assert audio_declick.smooth_path("/dev/shm/audio_cache/m_1.ogg") == "/dev/shm/audio_cache/m_1.smooth.ogg"
