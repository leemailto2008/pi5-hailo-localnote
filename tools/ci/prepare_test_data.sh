#!/usr/bin/env bash
# 確認 $LN_E2E_DATA 內有與 fixtures (audio.sha256) 相符的測試 WAV；若只有 FLAC 則無損解回 (-bitexact，sha256 不變)。
# usage: tools/ci/prepare_test_data.sh 1 3 5      (預設 1 3)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA=${LN_E2E_DATA:-/home/pi/ln_e2e/data}
MINUTES=${*:-1 3}
mkdir -p "$DATA"
for m in $MINUTES; do
  fx="$ROOT/tools/e2e/fixtures/meeting_${m}min.json"; wav="$DATA/meeting_${m}min.wav"; flac="$DATA/meeting_${m}min.flac"
  want=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["audio"]["sha256"])' "$fx")
  have() { [ -f "$wav" ] && sha256sum "$wav" | cut -d' ' -f1 || echo none; }
  if [ "$(have)" = "$want" ]; then echo "ok    ${m}min"; continue; fi
  [ -f "$flac" ] && ffmpeg -nostdin -v error -y -i "$flac" -f wav -bitexact "$wav"
  if [ "$(have)" != "$want" ]; then
    echo "FAIL  ${m}min：測試音檔缺少或與 fixture 不符 (want ${want:0:12}, have $(have | cut -c1-12))"
    echo "      fixtures 改版後須重新執行 gen_test_audio.py，並把 WAV/FLAC 複製到 $DATA"
    exit 1
  fi
  echo "ok    ${m}min (decoded from flac)"
done
