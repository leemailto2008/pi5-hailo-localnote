# LocalNote 全流程 e2e 自測 (錄音 → STT → Summary)

不需真人說話：以已知內容的測試音檔驅動 app，量測 CER、事實命中、各階段耗時與 CPU/NPU/RAM/溫度。

## 檔案
- `script_meeting.json`：11 句雙人「產品週會」腳本與 9 項待核對事實。
- `gen_test_audio.py`：用 edge-tts 合成 `test_meeting.wav`（16 kHz、-55 dBFS 底噪）。
- `e2e.py`：建立 PipeWire null sink，寫入 `/dev/shm/ln_force_source` 讓 app 改錄 sink monitor，呼叫 `/api/record`，即時播放測試音檔並輪詢至完成，輸出結果 JSON 到 `/tmp/e2e_<label>.json`，結束後清除注入檔與 sink。
- `mon.py`：1 Hz 資源取樣（由 e2e.py 自動啟動）。

## 使用（在 Pi 上，app 閒置時）
```bash
python3 gen_test_audio.py script_meeting.json test_meeting.wav   # 可在任何有網路的機器產生
python3 e2e.py test_meeting.wav script_meeting.json r1
```

## 2026-10-22 基準 (raspi678)
修正前 r1：CER 42%、summary 事實 5/9、STT 0/10 使用 NPU、錄音中 CPU 80%。
修正後 r4：CER 29%、summary 事實 8/9、STT 11/11 使用 NPU、錄音中 CPU 3.4%、CPU 最高 49.6 °C、RAM 峰值 1.67 GB。
