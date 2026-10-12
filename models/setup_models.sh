#!/usr/bin/env bash
# ==============================================================================
# Pi 5 Local Note - Automated AI Models Provisioning Script
# ==============================================================================
set -e

echo "=== [1/3] 檢查並預熱 faster-whisper 語音轉錄模型 (STT) ==="
python3 -c "
from faster_whisper import WhisperModel
print('Downloading & caching faster-whisper small (int8 NEON)...')
model = WhisperModel('small', device='cpu', compute_type='int8')
print('Whisper model loaded and ready!')
"

echo "=== [2/3] 拉取 Ollama 邊緣大語言模型 ==="
if command -v ollama >/dev/null 2>&1; then
    echo "Pulling qwen2.5:3b (主即時翻譯模型)..."
    ollama pull qwen2.5:3b
    
    echo "建立 qwen2.5:3b-opt 最佳化口譯專用模型..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ollama create qwen2.5:3b-opt -f "${SCRIPT_DIR}/Modelfile.qwen2.5-3b-opt"
    
    echo "Pulling optional coding model qwen2.5-coder:7b..."
    ollama pull qwen2.5-coder:7b || echo "Optional 7B model pull skipped"
else
    echo "⚠️ 系統未偵測到 ollama 命令，請確認已安裝 Ollama 服務 (curl -fsSL https://ollama.com/install.sh | sh)"
fi

echo "=== [3/3] 測試 Edge-TTS 神經語音合成 ==="
python3 -c "
import edge_tts, asyncio
async def test():
    c = edge_tts.Communicate('語音初始化完成', 'zh-TW-HsiaoChenNeural')
    await c.save('/dev/null')
asyncio.run(test())
print('Edge-TTS ready!')
" || echo "Edge-TTS check skipped"

echo "✅ 所有 AI 模型設定與預熱完成！"
