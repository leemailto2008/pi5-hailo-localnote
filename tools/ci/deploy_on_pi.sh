#!/usr/bin/env bash
# CD：把 repo checkout 部署到本機 Pi 的 /home/pi/code_dispatcher，健康檢查失敗自動回滾。
# 只部署程式碼 (code_dispatcher/*.py、ui.html、static/、hailo_whisper/*.py)；
# 不碰 data/、audio_cache/、WirePlumber 與 systemd 設定 (系統層設定請人工審核後再套用)。
# usage: tools/ci/deploy_on_pi.sh [repo_root]     (需在 Pi 上以 pi 使用者執行，且 sudo -n systemctl 免密碼)
set -euo pipefail
SRC=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
DST=${LN_APP_DIR:-/home/pi/code_dispatcher}
SVC=pi5-dispatcher-api.service

api_state() { curl -s -m 5 http://127.0.0.1/api/status | python3 -c 'import sys,json;print(json.load(sys.stdin).get("state",""))' 2>/dev/null || echo down; }
st=$(api_state)
case "$st" in recording|processing) echo "app 忙碌中 ($st)，停止部署"; exit 75;; esac

echo "[deploy] 1/4 語法檢查"
python3 -m py_compile "$SRC"/code_dispatcher/*.py "$SRC"/hailo_whisper/*.py

echo "[deploy] 2/4 備份"
TS=$(date +%Y%m%d_%H%M%S); BK="$DST/.deploy_backup/$TS"
mkdir -p "$BK/static" "$BK/hailo_whisper"
cp -p "$DST"/*.py "$DST"/*.html "$BK/" 2>/dev/null || true
cp -rp "$DST"/static/. "$BK/static/" 2>/dev/null || true
cp -p "$DST"/hailo_whisper/*.py "$BK/hailo_whisper/" 2>/dev/null || true

echo "[deploy] 3/4 安裝"
cp -f "$SRC"/code_dispatcher/*.py "$SRC"/code_dispatcher/ui.html "$DST"/
cp -rf "$SRC"/code_dispatcher/static/. "$DST"/static/
cp -f "$SRC"/hailo_whisper/*.py "$DST"/hailo_whisper/
for f in ui_client.html static/ui_client.html static/index.html; do cp -f "$DST/ui.html" "$DST/$f"; done   # ui.html 為唯一原始檔

echo "[deploy] 4/4 重啟與健康檢查"
T0=$(date '+%Y-%m-%d %H:%M:%S')
sudo -n systemctl restart "$SVC"
ok=0
for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1/health >/dev/null && { ok=1; break; }; sleep 1; done
sleep 5
if [ "$ok" = 1 ] && ! journalctl -u "$SVC" --since "$T0" --no-pager | grep -q -E "Traceback|ImportError|ModuleNotFoundError|SyntaxError"; then
  echo "[deploy] OK → $DST (backup $BK)"
  ls -1dt "$DST"/.deploy_backup/* 2>/dev/null | tail -n +11 | xargs -r rm -rf    # 保留最近 10 份備份
  exit 0
fi
echo "[deploy] FAILED，回滾到 $BK"
journalctl -u "$SVC" --since "$T0" --no-pager | tail -20 || true
cp -f "$BK"/*.py "$BK"/*.html "$DST"/ 2>/dev/null || true
cp -rf "$BK"/static/. "$DST"/static/ 2>/dev/null || true
cp -f "$BK"/hailo_whisper/*.py "$DST"/hailo_whisper/ 2>/dev/null || true
sudo -n systemctl restart "$SVC"
exit 1
