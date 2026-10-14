#!/usr/bin/env bash
# ==============================================================================
# AI_HAT_plus/download_models.sh
# Download pre-compiled Whisper Base/Tiny HEF models for Hailo-8 (26 TOPS)
# Source: Hailo Official Community CS-Data S3 Bucket
# ==============================================================================
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
mkdir -p "$MODEL_DIR"

BASE_HEF="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/whisper/h8"

echo "=========================================================================="
echo " 📥 下載 Hailo-8 (26 TOPS) 官方 Whisper 加速模型 (HEF)"
echo " 目標目錄: $MODEL_DIR"
echo "=========================================================================="

# 1. Base Encoder (5s chunk)
ENCODER_BASE="base-whisper-encoder-5s.hef"
if [ ! -f "$MODEL_DIR/$ENCODER_BASE" ]; then
    echo "下載 $ENCODER_BASE..."
    curl -L --progress-bar -f -o "$MODEL_DIR/$ENCODER_BASE" "$BASE_HEF/$ENCODER_BASE"
else
    echo "✅ $ENCODER_BASE 已存在。"
fi

# 2. Base Decoder
DECODER_BASE="base-whisper-decoder-fixed-sequence-matmul-split.hef"
if [ ! -f "$MODEL_DIR/$DECODER_BASE" ]; then
    echo "下載 $DECODER_BASE..."
    curl -L --progress-bar -f -o "$MODEL_DIR/$DECODER_BASE" "$BASE_HEF/$DECODER_BASE"
else
    echo "✅ $DECODER_BASE 已存在。"
fi

# 建立符號連結方便統一口徑調用
cd "$MODEL_DIR"
ln -sf "$ENCODER_BASE" whisper_base.hef 2>/dev/null || true

echo "=========================================================================="
echo " 🎉 Hailo-8 Whisper HEF 模型下載就緒！"
ls -lh "$MODEL_DIR"
echo "=========================================================================="
