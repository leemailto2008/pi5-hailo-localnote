"""bt_link_stats：SCO 到達率的滑動視窗與整段計算；讀不到計數或不是 mSBC 時回傳 None。"""
import bt_link_stats
from bt_link_stats import MSBC_PKTS_PER_SEC, ArrivalMeter


class FakeLink:
    """假時鐘 + 假計數器：rate 為每秒收到的封包數。"""
    def __init__(self):
        self.t, self.n, self.rate = 0.0, 1000.0, MSBC_PKTS_PER_SEC

    def advance(self, secs, step=0.1):
        for _ in range(int(round(secs / step))):
            self.t += step
            self.n += self.rate * step
            yield

    def read(self):
        return int(self.n)


def _meter(link, **kw):
    return ArrivalMeter(reader=link.read, clock=lambda: link.t, **kw)


def test_full_rate_is_100_and_loss_shows_in_live_window():
    link = FakeLink(); m = _meter(link)
    for _ in link.advance(20):
        m.tick()
    assert m.live_pct() == 100.0 and m.total_pct() == 100.0
    link.rate = MSBC_PKTS_PER_SEC * 0.8                     # 掉 20%
    for _ in link.advance(12):
        m.tick()
    assert 79.0 <= m.live_pct() <= 81.0                     # 最近 10 秒只看到掉包後
    assert 88.0 <= m.total_pct() <= 93.0                    # 整段平均：20 s × 100% + 12 s × 80%


def test_reads_counter_only_once_per_interval():
    link = FakeLink(); calls = []
    m = ArrivalMeter(reader=lambda: calls.append(1) or link.read(), clock=lambda: link.t, interval_s=1.0)
    for _ in link.advance(5, step=0.01):                   # 錄音迴圈每 10 ms 呼叫一次
        m.tick()
    assert 5 <= len(calls) <= 6


def test_unknown_when_no_counter_or_not_msbc():
    link = FakeLink(); m = ArrivalMeter(reader=lambda: None, clock=lambda: link.t)
    for _ in link.advance(5):
        m.tick()
    assert m.live_pct() is None and m.total_pct() is None
    link = FakeLink(); link.rate = MSBC_PKTS_PER_SEC * 2      # 例如 CVSD：封包比 mSBC 密
    m = _meter(link)
    for _ in link.advance(5):
        m.tick()
    assert m.live_pct() is None and m.total_pct() is None and m.unknown_codec


def test_no_sco_traffic_is_none_not_100_percent_loss():
    link = FakeLink(); link.rate = 0.0                       # USB 麥克風或 SCO 還沒建立：計數不動
    m = _meter(link)
    for _ in link.advance(12):
        m.tick()
    assert m.live_pct() is None and m.total_pct() is None


def test_too_short_is_none():
    link = FakeLink(); m = _meter(link)
    for _ in link.advance(1.0):
        m.tick()
    assert m.live_pct() is None


def test_sco_rx_packets_handles_missing_bluetooth(monkeypatch):
    def boom(*a, **k):
        raise OSError("Address family not supported")
    monkeypatch.setattr(bt_link_stats.socket, "socket", boom)
    assert bt_link_stats.sco_rx_packets() is None
