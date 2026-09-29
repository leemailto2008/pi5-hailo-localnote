import urllib.request
import time

try:
    t0 = time.time()
    req = urllib.request.Request(
        "https://speed.cloudflare.com/__down?bytes=2000000",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        b = len(resp.read())
    dt = time.time() - t0
    mbps = (b * 8) / (dt * 1024 * 1024)
    print(f"Speed: {mbps:.2f} Mbps")
except Exception as e:
    print(f"Speed error: {e}")
