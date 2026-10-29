# LocalNote 全流程自測 (e2e / HIL) 與 CI/CD

以「內容已知」的測試會議驅動 app 完整流程（錄音 → ZeroSplicer → 串流 STT → Summary），自動評分品質、延遲與資源，不需要真人說話。

## 組成

| 檔案 | 用途 |
|---|---|
| `meeting_gen.py` | 測試會議腳本產生器：12 種議題模板 × 固定 seed 填值，每個議題內建可核對事實，結尾總結重述決議與待辦 |
| `gen_test_audio.py` | edge-tts 三聲線合成 → 依實際長度自動校正議題數 → 寫回 fixture (`audio.sha256`、每句時間軸) |
| `fixtures/meeting_{1,3,5,10,15,30}min.json` | 已校正的 fixtures（**進 git**）；音檔不進 git |
| `scoring.py` | 純函式評分：CER、事實召回、summary 結構／重複／總結召回／議題覆蓋、門檻判定 |
| `e2e.py` | 單一案例：null sink 注入 (`/dev/shm/ln_force_source`) → `/api/record` → 即時播放 → 評分 + `mon.py` 資源取樣 |
| `run_suite.py` | 多案例：套 `thresholds.json`，輸出 `suite.json`、`summary.md`、`junit.xml`；任一失敗 exit 1 |
| `speaker_selftest.py` | 聲學自測（在接喇叭的電腦上執行）：喇叭播放講稿 → 耳麥 → 藍牙 mSBC → app，錄完用 fixture 評分；只用標準函式庫 |
| `thresholds.json` | 門檻（`default` + `per_minutes` 覆寫） |
| `../ci/deploy_on_pi.sh` | CD：部署程式到 Pi、健康檢查、失敗自動回滾 |
| `../ci/prepare_test_data.sh` | 檢查/解出與 fixture sha256 相符的測試音檔 |
| `../../tests/` | 離線單元測試 (pytest)：`audio_splice`、`summary_utils`、`scoring`、fixtures 一致性 |

## 測試會議規格

| 版本 | 實際長度 | 議題數 | 逐字稿事實 | 總結事實 |
|---|---|---|---|---|
| 1 分 | 75.8 s | 2 | 11 | 4 |
| 3 分 | 193.0 s | 7 | 34 | 6 |
| 5 分 | 312.1 s | 13 | 67 | 6 |
| 10 分 | 625.1 s | 27 | 140 | 6 |
| 15 分 | 941.8 s | 42 | 218 | 6 |
| 30 分 | 1786.6 s | 82 | 423 | 6 |

- 三種聲線：`zh-TW-HsiaoChenNeural`、`zh-TW-YunJheNeural`、`zh-TW-HsiaoYuNeural`；聲線被限流時自動改用備援聲線，實際使用的聲線記錄在 `timeline[].voice`。
- 疊加 -55 dBFS 底噪：避免整段數位零被 ZeroSplicer 當成 PipeWire 插入的靜音而剪掉。
- 長會議版本（45 分／1／1.5／2／3／4 小時）共用同一個 seed（`LONG_SERIES_SEED`），議題序列互為前綴：4 小時版的前段就是 3 小時版的內容。逐句 TTS 快取因此可共用，6 個長度合計 6440 句、不重複的只有約 1360 句。
  - 題庫的值有限，所以長會議裡會有重複的句子。長版本主要用來測長時間穩定度（記憶體、NPU、keep_alive 視窗、網頁停頓、溫度），內容品質仍照同一套門檻評分。
- 4 小時版限制在 14390 s 以內（`/api/record` 上限 14400 s）。
- 音檔以串流方式寫出（只讀各句長度、逐句載入），4 小時版也只需數十 MB 記憶體。新舊寫法的輸出逐位元相同，既有 6 個短版本的 sha256 不變。

## 產生測試音檔（需網路與 edge-tts）

```bash
pip install edge-tts numpy
python3 tools/e2e/gen_test_audio.py --minutes 1 3 5 10 15 30 --out ~/ln_e2e_data
python3 tools/e2e/gen_test_audio.py --minutes 45 60 90 120 180 240 --out ~/ln_e2e_data   # 長會議，WAV 合計約 1.4 GB
# 傳到 Pi：FLAC 無損壓縮後再解回，sha256 不變
ffmpeg -i meeting_30min.wav -c:a flac meeting_30min.flac
scp meeting_*min.flac pi@<pi>:/home/pi/ln_e2e/data/
ssh pi@<pi> 'cd /path/to/repo && bash tools/ci/prepare_test_data.sh 1 3 5 10 15 30'
```

模板或 seed 改變時 fixtures 會跟著改變（`audio.sha256` 不同），必須重新產生音檔並重新分發，`prepare_test_data.sh` 會擋下不一致的情況。

## 執行

```bash
python3 -m pytest -q tests                                          # 離線單元測試，任何機器
python3 tools/e2e/run_suite.py --minutes 1 3                        # Pi 上，quick（約 10 分鐘）
python3 tools/e2e/run_suite.py --minutes 1 3 5 10 15 30             # Pi 上，full（約 2 小時）
python3 tools/e2e/run_suite.py --minutes 45 60 90 120 180 240        # Pi 上，long（約 14 小時，含紀要）
python3 tools/e2e/run_suite.py --minutes 5 --allow-bypass           # 只測錄音 + STT
python tools/e2e/speaker_selftest.py --pi http://192.168.50.228 --wav meeting_1min.wav   # 接喇叭的電腦上，耳麥放喇叭前（聲學路徑）
```

- 每次都用 `bypass_summary=false` 發送請求，只作用於該次錄音，**不改 UI 的全域設定**。
- app 必須閒置；suite 會等候 recording/processing 結束，案例之間間隔 15 秒讓 NPU 釋放。

## CI/CD

- **GitLab** (`.gitlab-ci.yml`)：`unit`（docker）→ `deploy_raspi678` → `hil_quick`（main 每次 push）／`hil_full`（排程或手動）／`hil_long`（排程變數 `LN_E2E_LONG=1`，建議每週；或手動）。
- **GitHub** (`.github/workflows/ci.yml`)：`unit` 在 GitHub 託管 runner；`hil` 需在 Pi 上註冊 self-hosted runner（label `raspi678`），以 workflow_dispatch 手動觸發。
- Pi runner 註冊（shell executor，以 `pi` 使用者執行，需要 `sudo -n systemctl restart pi5-dispatcher-api` 免密碼）：
  - GitLab：安裝 gitlab-runner → `gitlab-runner register --executor shell --tag-list raspi678` → 讓服務以 `pi` 執行。
  - GitHub：Settings → Actions → Runners → New self-hosted runner (Linux ARM64)，labels 加 `raspi678`。
- 部署只動程式碼，不碰 WirePlumber/systemd 設定；`ui.html` 為唯一原始檔，部署時複製成 `ui_client.html`、`static/index.html`、`static/ui_client.html`。
