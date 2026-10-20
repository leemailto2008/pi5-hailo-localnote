# test_api_endpoints.py
"""
End-to-End API endpoint integration test for LocalNote Demo UI.
"""

import os
import unittest
from fastapi.testclient import TestClient

# Ensure environment variables
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
os.environ.setdefault("PULSE_SERVER", "unix:/run/user/1000/pulse/native")

from code_dispatcher.main import app


class TestDemoUIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_web_ui_root(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("LocalNote 聽記盒", resp.text)
        self.assertIn("開始錄音", resp.text)
        self.assertIn("會議紀要", resp.text)

    def test_dbg01_and_arena_routes(self):
        resp_dbg = self.client.get("/dbg01")
        self.assertEqual(resp_dbg.status_code, 200)

        resp_arena = self.client.get("/arena")
        self.assertEqual(resp_arena.status_code, 200)

        resp_dbg02 = self.client.get("/dbg02")
        self.assertEqual(resp_dbg02.status_code, 200)
        self.assertIn("Bluetooth Voice Gateway", resp_dbg02.text)

        resp_dgb02 = self.client.get("/dgb02")
        self.assertEqual(resp_dgb02.status_code, 200)

        resp_bt = self.client.get("/bluetooth")
        self.assertEqual(resp_bt.status_code, 200)
        self.assertIn("藍牙音訊設備中心", resp_bt.text)

        resp_bt_short = self.client.get("/bt")
        self.assertEqual(resp_bt_short.status_code, 200)


    def test_api_system(self):
        resp = self.client.get("/api/system")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("offline", data)
        self.assertIn("cpu", data)
        self.assertIn("temp", data)
        self.assertIn("mode", data)
        self.assertTrue(data["ssid"].endswith("_AI"))
        self.assertTrue(data["host"].endswith(".local"))

    def test_api_bt(self):
        resp = self.client.get("/api/bt")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("connected", data)
        self.assertIn("name", data)

    def test_api_status_and_result(self):
        resp_st = self.client.get("/api/status")
        self.assertEqual(resp_st.status_code, 200)
        st = resp_st.json()
        self.assertIn(st["state"], ["idle", "recording", "processing", "done", "error"])

        resp_res = self.client.get("/api/result")
        self.assertEqual(resp_res.status_code, 200)
        res = resp_res.json()
        self.assertIn("summary", res)
        self.assertIn("todos", res)
        self.assertIn("decisions", res)

    def test_static_qr_code(self):
        resp = self.client.get("/static/wifi.png")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("content-type"), "image/png")


if __name__ == "__main__":
    unittest.main()
