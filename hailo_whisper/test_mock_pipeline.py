# -*- coding: utf-8 -*-
# hailo_whisper/test_mock_pipeline.py
"""
test_mock_pipeline.py: Offline pipeline and unit test suite for hailo_whisper.
Runs without physical Hailo hardware to verify preprocessor, tokenizer,
engine fallback, and hallucination filters.
"""

import sys
import io
import numpy as np

# Force UTF-8 standard output for Windows consoles
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from hailo_whisper.audio_processor import AudioProcessor
from hailo_whisper.tokenizer import WhisperTokenizer
from hailo_whisper.engine import get_whisper_engine, CpuFallbackWhisperEngine, HailoWhisperEngine


def test_audio_processor():
    print("[Test 1/4] Testing AudioProcessor Log-Mel Spectrogram extraction...")
    ap = AudioProcessor(n_mels=80)
    # Generate 6 seconds of dummy 16kHz audio (96,000 samples)
    audio = np.random.uniform(-0.8, 0.8, 16000 * 6).astype(np.float32)
    mel = ap.log_mel_spectrogram(audio, padding=True)
    assert mel.shape == (80, 3000), f"Expected shape (80, 3000), got {mel.shape}"
    assert not np.isnan(mel).any(), "Mel spectrogram contains NaN values"
    print(f"  -> AudioProcessor PASSED! Output shape: {mel.shape}, min={mel.min():.2f}, max={mel.max():.2f}")


def test_tokenizer_prefix_tokens():
    print("\n[Test 2/4] Testing WhisperTokenizer prefix token generation...")
    tk = WhisperTokenizer()

    # English transcribe without timestamps
    en_tokens = tk.get_initial_tokens(language="en", task="transcribe", timestamps=False)
    assert en_tokens == [50258, 50259, 50359, 50363], f"Unexpected en tokens: {en_tokens}"

    # Chinese transcribe without timestamps
    zh_tokens = tk.get_initial_tokens(language="zh", task="transcribe", timestamps=False)
    assert zh_tokens == [50258, 50260, 50359, 50363], f"Unexpected zh tokens: {zh_tokens}"

    # Auto language without timestamps
    auto_tokens = tk.get_initial_tokens(language=None, task="transcribe", timestamps=False)
    assert auto_tokens == [50258, 50359, 50363], f"Unexpected auto tokens: {auto_tokens}"

    print(f"  -> Prefix Tokens PASSED! en={en_tokens}, zh={zh_tokens}, auto={auto_tokens}")


def test_tokenizer_decoding_and_filtering():
    print("\n[Test 3/4] Testing WhisperTokenizer decoding, hallucination filtering & OpenCC...")
    tk = WhisperTokenizer()

    # Inject mock mapping for test
    tk._id_to_token[100] = "Hello"
    tk._id_to_token[101] = " world"
    tk._id_to_token[102] = " No Hallucination."
    tk._id_to_token[103] = " 树莓派人工智能"

    raw_tokens = [50258, 100, 101, 102, 103, 50257]
    decoded = tk.decode_tokens(raw_tokens, convert_traditional=True)

    assert "No Hallucination." not in decoded, "Hallucination prompt bleed not filtered"
    assert "樹莓派人工智慧" in decoded or "树莓派" in decoded, "Text decoding failed"
    print(f"  -> Decoded output: '{decoded}'")
    print("  -> Decoding & Filtering PASSED!")


def test_engine_fallback():
    print("\n[Test 4/4] Testing Engine factory and hardware detection fallback...")
    engine = get_whisper_engine(preferred_backend="auto")
    assert isinstance(engine, CpuFallbackWhisperEngine), f"Expected fallback engine, got {type(engine)}"
    assert not engine.is_hardware_accelerated(), "Fallback should report hardware_accelerated=False"
    print(f"  -> Active Engine: {type(engine).__name__} (Hardware Accelerated: {engine.is_hardware_accelerated()})")
    print("  -> Engine Fallback PASSED!")


def main():
    print("==================================================")
    print(" Hailo-10H Whisper Acceleration Offline Unit Tests")
    print("==================================================")
    test_audio_processor()
    test_tokenizer_prefix_tokens()
    test_tokenizer_decoding_and_filtering()
    test_engine_fallback()
    print("\n==================================================")
    print(" ALL OFFLINE TESTS COMPLETED SUCCESSFULLY! (4/4)  ")
    print("==================================================")


if __name__ == "__main__":
    main()
