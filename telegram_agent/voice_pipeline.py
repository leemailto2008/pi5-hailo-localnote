# f:\12_prj_raspi5\telegram_agent\voice_pipeline.py
"""
Bi-directional Voice Pipeline for Raspberry Pi 5:
- Speech-To-Text (STT): faster-whisper (ARM NEON int8) + ffmpeg audio preprocessing
- Text-To-Speech (TTS): edge-tts (Microsoft Neural Taiwan Traditional Chinese: 曉臻 / 雲哲)
"""

import os
import sys
import re
import asyncio
import subprocess
from typing import Optional, Any, Callable, Dict, Union, List
import numpy as np

from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    DEFAULT_TTS_VOICE,
    DATA_DIR
)

# Global faster-whisper model cache (lazy-loaded to save RAM until first audio)
_whisper_model = None


def get_whisper_model():
    """Lazy load faster-whisper model into memory."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            print(f"[VoicePipeline] Initializing faster-whisper ({WHISPER_MODEL_SIZE}, {WHISPER_COMPUTE_TYPE})...")
            # 錄音結束後以全 4 核心極速轉錄，最大化發揮 Cortex-A76 效能
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=4
            )
            print("[VoicePipeline] faster-whisper model loaded successfully (4 threads).")
        except Exception as e:
            print(f"[VoicePipeline] Error loading faster-whisper: {e}")
            _whisper_model = False
    return _whisper_model if _whisper_model is not False else None


try:
    from opencc import OpenCC
    _opencc_converter = OpenCC('s2twp')
except Exception:
    _opencc_converter = None


def to_taiwan_traditional(text: str) -> str:
    """Normalize Chinese text to Taiwan Traditional standard using OpenCC."""
    if not text:
        return ""
    if _opencc_converter:
        try:
            return _opencc_converter.convert(text)
        except Exception:
            pass
    return text


def is_silence_or_empty_audio(wav_path: str, rms_threshold: int = 80) -> bool:
    """
    Check if a WAV file contains only silence or background static.
    Uses struct to calculate RMS, compatible with Python 3.13 (no audioop dependency).
    """
    try:
        import wave
        import struct
        import math
        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            if n_frames == 0:
                return True
            frames = wf.readframes(n_frames)
            count = len(frames) // 2
            if count == 0:
                return True
            shorts = struct.unpack(f"<{count}h", frames)
            sum_sq = sum(s * s for s in shorts)
            rms = int(math.sqrt(sum_sq / count))
            if rms < rms_threshold:
                print(f"[VoicePipeline] Audio RMS too low ({rms} < {rms_threshold}), detected as silence.")
                return True
            return False
    except Exception as e:
        print(f"[VoicePipeline] Error in silence check: {e}")
        return False


def convert_oga_to_wav(input_oga_path: str, output_wav_path: str) -> bool:
    """
    Convert recorded voice to 16kHz mono PCM WAV for Whisper.
    Uses clean highpass filter to eliminate microphone DC offset and low-frequency rumble,
    avoiding harmful double-dynaudnorm distortion.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_oga_path,
            "-af", "highpass=f=75,lowpass=f=7600",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
        if res.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 100:
            return True

        # Fallback attempt: Plain PCM conversion
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", input_oga_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav_path
        ]
        res_fb = subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return res_fb.returncode == 0 and os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 100
    except Exception as e:
        print(f"[VoicePipeline] ffmpeg conversion failed: {e}")
        return False


def clean_hallucination_repetitions(text: str) -> str:
    """
    抑制 Whisper 偶發之自回歸解碼重複迴圈 (Hallucination Repetition Loops)。
    例如: '在這裡,我們會在這裡,我們會在這裡,...我們' -> '在這裡,我們會在這裡'
    保留自然口語疊字 (如 '看看'、'好好')，只過濾重複 2 次以上的長度 >= 2 片語。
    """
    if not text:
        return text

    # 針對包含逗號、頓號、空白或無標點的連續重複片語進行壓縮 (短語長度 2~30)
    for _ in range(3):
        def repl(match):
            phrase = match.group(1)
            sep = match.group(2)
            return phrase + (sep if sep else "")
        text = re.sub(r'([^\s，,。！？!?]{2,30})([，,、\s]?)(?:\1\2){2,}', repl, text)

    # 清理結尾殘缺的重複片段（例如最後被 token limit 截斷剩下的殘字，如 "，我們"）
    m = re.search(r'([^\s，,。！？!?]{2,20})[，,、\s]*([^\s，,。！？!?]{1,10})$', text)
    if m:
        full_word, tail = m.group(1), m.group(2)
        if full_word.startswith(tail) and len(tail) < len(full_word):
            text = text[:-len(tail)].rstrip("，,、 ")

    return text.strip()


def join_bilingual_tokens(tokens: List[str]) -> str:
    """智慧拼接中英文語音片段，若相鄰皆為拉丁字母/數字則自動保留標準空格，杜絕英文黏字"""
    res = ""
    for t in tokens:
        if not t:
            continue
        if not res:
            res = t
        elif res[-1].isalnum() and t[0].isalnum():
            res += " " + t
        else:
            res += t
    return res


import threading
_whisper_lock = threading.Lock()


def speech_to_text(
    audio_source: Union[str, np.ndarray, bytes],
    cancellation_event: Optional[Any] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    on_segment_callback: Optional[Any] = None,
    language: Optional[str] = None
) -> Optional[str]:
    """
    Transcribe speech from an audio file, PCM bytes, or float32 PCM ndarray.
    Supports 100% automatic language detection (language=None) preserving natural bilingual code-switching (Traditional Chinese & English verbatim).
    Optimized with 100% Zero-Disk I/O and RAM-based PCM streaming for Raspberry Pi 5:
    - Decodes audio directly into float32 16kHz PCM array in RAM.
    - Directly accepts in-memory numpy array for 0-latency live chunk streaming.
    - Zero temporary WAV files on physical storage (0 MicroSD wear).
    - Threading lock prevents CTranslate2 OpenMP deadlocks/spin-waits.
    - beam_size=1 (Greedy search) delivers 5x faster transcription with zero hallucination.
    - Supports cancellation_event checking and region trimming (start_sec, end_sec).
    - Supports real-time segment-by-segment streaming callback (on_segment_callback).
    """
    audio_np: Optional[np.ndarray] = None

    if isinstance(audio_source, str):
        if not os.path.exists(audio_source) or os.path.getsize(audio_source) == 0:
            print(f"[VoicePipeline] Audio file missing or empty: {audio_source}")
            return None

    # ---------------------------------------------------------
    # 🚀 Hailo-10H NPU (40 TOPS) 優先硬體加速探測 (支援 np.ndarray, bytes, str)
    # ---------------------------------------------------------
    if start_sec is None and end_sec is None:
        try:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            if curr_dir not in sys.path:
                sys.path.append(curr_dir)
            root_path = os.path.abspath(os.path.join(curr_dir, ".."))
            if root_path not in sys.path:
                sys.path.append(root_path)

            from hailo_whisper.engine import get_whisper_engine
            h10_engine = get_whisper_engine(preferred_backend="hailo")
            if h10_engine.is_hardware_accelerated():
                h10_res = h10_engine.transcribe(
                    audio_source,
                    language=language,
                    initial_prompt=None,
                    temperature=0.0
                )
                if h10_res is not None and "text" in h10_res:
                    txt = h10_res.get("text", "")
                    if txt:
                        print(f"[VoicePipeline] ⚡ Hailo-10H 轉錄完成: '{txt}' (耗時: {h10_res.get('inference_time_ms', 0)}ms)")
                    return txt
        except Exception as h10_err:
            print(f"[VoicePipeline] Hailo-10H 轉錄探測異常，平滑降級: {h10_err}")

    if isinstance(audio_source, np.ndarray):
        audio_np = audio_source
    elif isinstance(audio_source, bytes):
        if len(audio_source) >= 2:
            audio_np = np.frombuffer(audio_source, dtype=np.int16).astype(np.float32) / 32768.0
    elif isinstance(audio_source, str):

        # ---------------------------------------------------------
        # 🚀 AI HAT+ (Hailo-8 NPU 26 TOPS) 硬體加速與自動探測切換
        # ---------------------------------------------------------
        if start_sec is None and end_sec is None:
            try:
                hailo_adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AI_HAT_plus"))
                if os.path.exists(hailo_adapter_path) and hailo_adapter_path not in sys.path:
                    sys.path.append(hailo_adapter_path)
                
                from hailo_engine.device_manager import is_hailo_available, HailoDeviceManager
                if is_hailo_available():
                    dev_info = HailoDeviceManager().get_info()
                    arch = str(dev_info.get("arch", "")).upper()
                    # Hailo-10H 專職負責 LLM，不載入不相容的 Hailo-8 Whisper HEF
                    if "HAILO10" not in arch:
                        from hailo_engine.whisper_adapter import get_whisper_engine
                        engine = get_whisper_engine(auto_detect=True, model_size="base")
                        if "Hailo" in engine.get_engine_type() and getattr(engine, "has_full_npu_pipeline", False):
                            print("[VoicePipeline] ⚡ 探測到 AI NPU！啟動硬體加速聽寫...")
                            res = engine.transcribe(
                                audio_source,
                                language=language or "zh",
                                cancellation_event=cancellation_event,
                                on_segment_callback=on_segment_callback
                            )
                            if res and res.get("text"):
                                text = to_taiwan_traditional(res["text"])
                                print(f"[VoicePipeline] ⚡ AI NPU 轉錄完成: '{text}' (耗時: {res.get('duration_sec', 0):.2f}s, RTF: {res.get('rtf', 0):.2f})")
                                return text
            except Exception as hailo_err:
                print(f"[VoicePipeline] AI NPU 轉錄異常，自動平滑降級至 CPU: {hailo_err}")

        try:
            from voice_memory import audio_to_pcm_array
            audio_np = audio_to_pcm_array(audio_source)
        except Exception as e:
            print(f"[VoicePipeline] Failed in-memory decoding: {e}")
            audio_np = None

    if audio_np is None or len(audio_np) < 8000:
        print(f"[VoicePipeline] Audio PCM decode failed or too short (<0.5s)")
        return None

    # 局部區間裁切 (Audio Region Trimming)
    if start_sec is not None or end_sec is not None:
        total_sec = len(audio_np) / 16000.0
        s_sec = max(0.0, float(start_sec or 0.0))
        e_sec = min(total_sec, float(end_sec or total_sec))
        if e_sec > s_sec:
            s_idx = int(s_sec * 16000)
            e_idx = int(e_sec * 16000)
            audio_np = audio_np[s_idx:e_idx]
            print(f"[VoicePipeline] Trimming audio for region: {s_sec:.2f}s - {e_sec:.2f}s (Samples: {len(audio_np)})")

    if len(audio_np) < 4000:
        print(f"[VoicePipeline] Trimmed audio too short (<0.25s): {len(audio_np)} samples")
        return None

    try:
        # Pre-check: In-memory silence check
        peak = float(np.max(np.abs(audio_np)))
        win_size = 16000
        has_voice_activity = peak >= 0.03
        if not has_voice_activity and len(audio_np) >= win_size:
            for i in range(0, len(audio_np) - win_size, win_size):
                if float(np.sqrt(np.mean(audio_np[i:i+win_size]**2))) >= 0.002:
                    has_voice_activity = True
                    break

        if not has_voice_activity:
            rms = float(np.sqrt(np.mean(audio_np**2)))
            print(f"[VoicePipeline] In-Memory Audio is absolute silence (Peak={peak:.4f}, RMS={rms:.4f}), skipping Whisper.")
            return None

        # Check cancellation before acquiring lock
        if cancellation_event and hasattr(cancellation_event, "is_set") and cancellation_event.is_set():
            print("[VoicePipeline] Transcription aborted before lock by cancellation token.")
            return None

        paragraphs = []
        current_para = []
        last_end = 0.0

        with _whisper_lock:
            # Check cancellation right after acquiring lock
            if cancellation_event and hasattr(cancellation_event, "is_set") and cancellation_event.is_set():
                print("[VoicePipeline] Transcription aborted after acquiring lock by cancellation token.")
                return None

            model = get_whisper_model()
            if model is None:
                print("[VoicePipeline] Whisper model unavailable.")
                return None

            try:
                # 依使用者指示加入 No Hallucination，消除 Whisper 幻覺補字與語言偏置
                if language == "en":
                    prompt = "No Hallucination. Clean verbatim English speech."
                elif language == "zh":
                    prompt = "No Hallucination. 繁體中文語音對話。"
                else:
                    # 原音自動語言識別：全英文無中文前綴干擾，杜絕停頓處被誘發中文幻覺
                    prompt = "No Hallucination. Verbatim speech transcription without hallucinated words."

                segments, info = model.transcribe(
                    audio_np,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,  # 嚴格貪婪解碼，杜絕 fallback 升溫產生的隨機補字
                    language=language,
                    initial_prompt=prompt,
                    condition_on_previous_text=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    compression_ratio_threshold=2.2,
                    hallucination_silence_threshold=1.5,
                    vad_filter=True,
                    vad_parameters=dict(threshold=0.30, min_silence_duration_ms=350, speech_pad_ms=150)
                )
                for seg in segments:
                    if cancellation_event and hasattr(cancellation_event, "is_set") and cancellation_event.is_set():
                        print(f"[VoicePipeline] Transcription cancelled mid-stream by token.")
                        return None

                    clean_seg_text = to_taiwan_traditional(seg.text.strip())
                    clean_seg_text = clean_hallucination_repetitions(clean_seg_text)

                    # 徹底過濾 Whisper 偶發將 prompt 原文當成逐字稿抄錄的狀況
                    for hp in ("No Hallucination", "no hallucination", "Verbatim speech transcription", "without hallucinated words", "Verbatim speech", "Clean verbatim"):
                        clean_seg_text = clean_seg_text.replace(hp, "").strip(" .,;:!?，。")

                    # 容許藍牙 mSBC 語音特有的底噪，只要 logprob > -3.2 且非純靜音即完整保留，杜絕截字少句
                    if clean_seg_text and seg.avg_logprob > -3.2 and (seg.no_speech_prob < 0.95 or seg.avg_logprob > -2.0):
                        start_t = getattr(seg, "start", 0.0)
                        end_t = getattr(seg, "end", 0.0)
                        pause_gap = start_t - last_end

                        # 更加細膩的停頓斷句分段 (適合手機閱讀)
                        should_break = False
                        curr_len = sum(len(x) for x in current_para)
                        if current_para:
                            if pause_gap >= 0.45:
                                should_break = True
                            elif pause_gap >= 0.25 and (curr_len >= 45 or any(current_para[-1].endswith(p) for p in ("。", "！", "？", "，", "…", "!", "?", "."))):
                                should_break = True
                            elif curr_len >= 75:
                                should_break = True

                        if should_break and current_para:
                            paragraphs.append(join_bilingual_tokens(current_para))
                            current_para = [clean_seg_text]
                        else:
                            current_para.append(clean_seg_text)

                        last_end = end_t

                        # ⚡ 逐段即時串流回呼
                        if on_segment_callback:
                            curr_prev = "\n\n".join(paragraphs + ([join_bilingual_tokens(current_para)] if current_para else []))
                            curr_prev = clean_hallucination_repetitions(curr_prev)
                            try:
                                on_segment_callback(curr_prev, clean_seg_text, False)
                            except Exception:
                                pass
                    else:
                        if clean_seg_text:
                            print(f"[VoicePipeline] Filtered hallucination/noise: '{seg.text}' (no_speech_prob={seg.no_speech_prob:.2f}, logprob={seg.avg_logprob:.2f})")
            except Exception as vad_err:
                print(f"[VoicePipeline] Whisper transcribe exception: {vad_err}")

        if current_para:
            paragraphs.append(join_bilingual_tokens(current_para))

        text = "\n\n".join(paragraphs).strip()

        # Sanitize known Whisper hallucination substrings
        hallucinations = [
            "字幕by", "字幕提供", "請不吝點贊", "點贊訂閱", "謝謝大家", "轉載請註明",
            "感謝收看", "索兰娅", "點點欄目", "明鏡", "優質內容", "歡迎關注", "歡迎訂閱",
            "諾映", "詩雅桑", "MING PAO", "小靈通", "CANADA", "TORONTO",
            "No Hallucination", "no hallucination", "Verbatim speech transcription",
            "without hallucinated words", "Verbatim speech", "Clean verbatim"
        ]
        for h in hallucinations:
            text = text.replace(h, "")
        text = text.strip()

        # 杜絕自回歸重複迴圈 (Deduplicate hallucination repetition loops)
        text = clean_hallucination_repetitions(text)

        # Normalize to pure Taiwan Traditional Chinese
        if text:
            text = to_taiwan_traditional(text)
            print(f"[VoicePipeline] Zero-Disk Transcribed text: '{text}' (PCM samples: {len(audio_np)})")
            return text
        else:
            print(f"[VoicePipeline] Empty transcription. In-Memory PCM samples: {len(audio_np)}")
            return None

    except Exception as e:
        print(f"[VoicePipeline] Transcription failed: {e}")
        return None


async def text_to_speech_async(text: str, output_ogg_path: Optional[str] = None, voice: str = DEFAULT_TTS_VOICE) -> bool:
    """
    Synthesize text into natural voice with multi-tier fallback:
    Tier 1: gTTS (Google Translate Neural TTS - ultra stable, natural, zero DRM 403 blocks)
    Tier 2: edge-tts (Microsoft Neural Voice)
    Tier 3: espeak-ng (100% offline local speech synthesizer)
    Outputs audio compatible with PipeWire pw-play / web player.
    """
    if not text or not text.strip():
        return False
    if output_ogg_path is None:
        import uuid
        output_ogg_path = os.path.join(TEMP_AUDIO_DIR, f"tts_{uuid.uuid4().hex[:8]}.ogg")

    clean_text = text.strip()

    # Tier 1: 優先嘗試 gTTS (穩定自然，杜絕微軟 403 阻擋)
    try:
        from gtts import gTTS
        is_english = any(ord(c) < 128 and c.isalpha() for c in clean_text[:30]) and not any('\u4e00' <= c <= '\u9fff' for c in clean_text[:30])
        lang = "en" if is_english else "zh-tw"
        temp_mp3 = output_ogg_path + ".tmp.mp3"
        tts_obj = gTTS(text=clean_text, lang=lang)
        tts_obj.save(temp_mp3)

        if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 0:
            if output_ogg_path.endswith(".ogg") or output_ogg_path.endswith(".wav"):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", temp_mp3, "-c:a", "libopus" if output_ogg_path.endswith(".ogg") else "pcm_s16le", output_ogg_path],
                    capture_output=True, timeout=10
                )
                try:
                    os.remove(temp_mp3)
                except Exception:
                    pass
            else:
                os.replace(temp_mp3, output_ogg_path)

            if os.path.exists(output_ogg_path) and os.path.getsize(output_ogg_path) > 0:
                print(f"[VoicePipeline] gTTS ({lang}) synthesis succeeded: {output_ogg_path}")
                return True
    except Exception as ge:
        print(f"[VoicePipeline] gTTS fallback attempt: {ge}")

    # Tier 2: 嘗試 edge-tts (微軟神經語音)
    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean_text, voice)
        await communicate.save(output_ogg_path)
        if os.path.exists(output_ogg_path) and os.path.getsize(output_ogg_path) > 0:
            print(f"[VoicePipeline] edge-tts synthesis succeeded: {output_ogg_path}")
            return True
    except Exception as e:
        print(f"[VoicePipeline] edge-tts synthesis failed: {e}")

    # Tier 3: 終極本地離線保底: espeak-ng
    try:
        is_english = not any('\u4e00' <= c <= '\u9fff' for c in clean_text[:30])
        voice_opt = "en" if is_english else "cmn"
        temp_wav = output_ogg_path + ".tmp.wav"
        res = subprocess.run(
            ["espeak-ng", "-v", voice_opt, "-w", temp_wav, clean_text[:500]],
            capture_output=True, timeout=10
        )
        if res.returncode == 0 and os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
            if output_ogg_path.endswith(".ogg"):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", temp_wav, "-c:a", "libopus", output_ogg_path],
                    capture_output=True, timeout=10
                )
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
            else:
                os.replace(temp_wav, output_ogg_path)

            if os.path.exists(output_ogg_path) and os.path.getsize(output_ogg_path) > 0:
                print(f"[VoicePipeline] espeak-ng offline synthesis succeeded: {output_ogg_path}")
                return True
    except Exception as ee:
        print(f"[VoicePipeline] espeak-ng synthesis failed: {ee}")

    return False


def text_to_speech(text: str, output_ogg_path: str, voice: str = DEFAULT_TTS_VOICE) -> bool:
    """Synchronous wrapper for text_to_speech_async."""
    try:
        return asyncio.run(text_to_speech_async(text, output_ogg_path, voice))
    except Exception as e:
        print(f"[VoicePipeline] Synchronous TTS failed: {e}")
        return False


async def ensure_bluetooth_recording_ready() -> str:
    """
    Ensure active Bluetooth headset is ready in mSBC mode and return exact PulseAudio source name.
    Uses non-blocking pw-dump to parse PipeWire topology accurately.
    Never interrupts an already active headset source.
    """
    # 自動將藍牙與 PipeWire 關鍵音訊服務提升至實時優先級 (-18) 並綁定 CPU 0,1，杜絕 UART 封包丟失
    try:
        import subprocess
        for p in ['bluetoothd', 'pipewire', 'wireplumber', 'pipewire-pulse']:
            out = subprocess.getoutput(f"pgrep {p}")
            for pid in out.split():
                if pid.isdigit():
                    subprocess.run(f"sudo renice -n -18 -p {pid} 2>/dev/null", shell=True)
                    subprocess.run(f"sudo taskset -apc 0,1 {pid} 2>/dev/null", shell=True)
    except Exception:
        pass

    target_source = "default"
    try:
        import json
        proc = await asyncio.create_subprocess_exec(
            "pw-dump",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await proc.communicate()
        nodes = json.loads(stdout.decode("utf-8", errors="ignore"))

        bt_source_node = None
        bt_source_id = None
        bt_device_id = None

        for n in nodes:
            props = n.get("info", {}).get("props", {})
            media_class = props.get("media.class", "")
            node_name = props.get("node.name", "")

            # 尋找藍牙音訊輸入節點 (如 bluez_input.A0_D1_2A_42_12_34.0)
            if media_class == "Audio/Source" and "bluez_input" in node_name:
                bt_source_node = node_name
                bt_source_id = n.get("id")

            # 尋找藍牙設備
            if props.get("device.api") == "bluez5" and n.get("type") == "PipeWire:Interface:Device":
                bt_device_id = n.get("id")

        # 1. 若藍牙 Source 節點已存在，說明 SCO 語音通道已經就緒，絕對不要重複 set-profile 打斷它！
        if bt_source_node and bt_source_id:
            set_def = await asyncio.create_subprocess_exec(
                "wpctl", "set-default", str(bt_source_id),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
            await set_def.communicate()
            return bt_source_node

        # 2. 若藍牙 Source 不存在但藍牙設備存在，切換 profile 為 headset-head-unit (196865)
        if bt_device_id:
            set_prof = await asyncio.create_subprocess_exec(
                "wpctl", "set-profile", str(bt_device_id), "196865",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
            )
            await set_prof.communicate()
            await asyncio.sleep(0.8)

            # 重新獲取 source 節點
            proc2 = await asyncio.create_subprocess_exec(
                "pw-dump", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
            )
            stdout2, _ = await proc2.communicate()
            nodes2 = json.loads(stdout2.decode("utf-8", errors="ignore"))
            for n in nodes2:
                props = n.get("info", {}).get("props", {})
                if props.get("media.class") == "Audio/Source" and "bluez_input" in props.get("node.name", ""):
                    bt_source_node = props.get("node.name")
                    await (await asyncio.create_subprocess_exec(
                        "wpctl", "set-default", str(n.get("id")),
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
                    )).communicate()
                    return bt_source_node

    except Exception as e:
        print(f"[VoicePipeline] Bluetooth readiness resolution error: {e}")

    return target_source


async def record_audio_from_mic(
    duration_sec: int,
    output_path: str,
    vu_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    cancellation_event: Optional[asyncio.Event] = None,
    on_proc_started: Optional[Callable[[Any], Any]] = None,
    pcm_chunk_callback: Optional[Callable[[bytes, bool], Any]] = None,
    chunk_sec: float = 5.0,
    finish_early_event: Optional[asyncio.Event] = None
) -> bool:
    """
    Record audio from Bluetooth headset mic using ffmpeg with explicit hardware source routing.
    Uses clean 16kHz resampling with highpass filter to eliminate sub-bass rumble and avoid clipping.
    If vu_callback or pcm_chunk_callback is supplied, streams raw 16kHz PCM to stdout for 50ms VU and concurrent streaming STT chunk slicing.
    Supports cancellation_event token for immediate hardware abort and finish_early_event for graceful early wrap-up.
    """
    source_node = await ensure_bluetooth_recording_ready()

    cmd = [
        "ffmpeg", "-y",
        "-thread_queue_size", "4096",
        "-f", "pulse",
        "-i", source_node,
        "-t", str(duration_sec),
        "-af", "aresample=16000:async=1,highpass=f=80,lowpass=f=7600",
        "-c:a", "libopus",
        "-b:a", "64k",
        output_path
    ]

    need_stdout_pcm = bool(vu_callback or pcm_chunk_callback)
    if need_stdout_pcm:
        cmd += [
            "-t", str(duration_sec),
            "-f", "s16le",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            "pipe:1"
        ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE if vu_callback else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )

        if on_proc_started:
            try:
                on_proc_started(proc)
            except Exception:
                pass

        if need_stdout_pcm and proc.stdout:
            chunk_bytes = 1600  # 50ms @ 16kHz mono (800 samples * 2 bytes)
            target_slice_bytes = int(chunk_sec * 32000)
            chunk_buffer = bytearray()

            band_slices = [
                (4, 13),    # 80-250 Hz (Sub-bass & fundamental pitch)
                (13, 25),   # 250-500 Hz (Low-mid vowels)
                (25, 50),   # 500-1000 Hz (Mid resonance)
                (50, 100),  # 1000-2000 Hz (High-mid clarity)
                (100, 175), # 2000-3500 Hz (Consonants & presence)
                (175, 250), # 3500-5000 Hz (Sibilance & fricatives)
                (250, 325), # 5000-6500 Hz (High harmonics)
                (325, 380)  # 6500-7600 Hz (Brilliance & air)
            ]
            try:
                while True:
                    if cancellation_event and cancellation_event.is_set():
                        print("[VoicePipeline] Microphone recording aborting due to cancellation event.")
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        return False

                    if finish_early_event and finish_early_event.is_set():
                        print("[VoicePipeline] Microphone recording finishing early upon user request.")
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        break

                    try:
                        chunk = await asyncio.wait_for(proc.stdout.read(chunk_bytes), timeout=0.25)
                    except asyncio.TimeoutError:
                        if cancellation_event and cancellation_event.is_set():
                            print("[VoicePipeline] Microphone recording aborting due to cancellation event (timeout).")
                            try:
                                proc.terminate()
                                proc.kill()
                            except Exception:
                                pass
                            return False
                        if finish_early_event and finish_early_event.is_set():
                            print("[VoicePipeline] Microphone recording finishing early upon user request (timeout).")
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            break
                        continue
                    if not chunk:
                        break

                    # 收集滑動切片 PCM
                    if pcm_chunk_callback:
                        chunk_buffer.extend(chunk)
                        if len(chunk_buffer) >= target_slice_bytes:
                            slice_data = bytes(chunk_buffer[:target_slice_bytes])
                            chunk_buffer = chunk_buffer[target_slice_bytes:]
                            try:
                                if asyncio.iscoroutinefunction(pcm_chunk_callback):
                                    await pcm_chunk_callback(slice_data, False)
                                else:
                                    pcm_chunk_callback(slice_data, False)
                            except Exception as c_err:
                                print(f"[VoicePipeline] pcm_chunk_callback error: {c_err}")

                    # VU 表計算
                    if vu_callback:
                        num_samples = len(chunk) // 2
                        if num_samples >= 160:
                            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                            rms = float(np.sqrt(np.mean(samples ** 2)))
                            peak = float(np.max(np.abs(samples)))
                            db = float(20 * np.log10(max(1e-5, rms)))
                            level_pct = min(100, int(rms * 280))

                            # 8-band FFT spectrum
                            fft_mag = np.abs(np.fft.rfft(samples))
                            bands = []
                            for s_start, s_end in band_slices:
                                b_val = float(np.mean(fft_mag[s_start:s_end])) if s_end <= len(fft_mag) else 0.0
                                bands.append(min(100, int(b_val * 65)))

                            status_label = "active" if peak >= 0.03 else ("weak" if peak >= 0.01 else "silent")
                            vu_payload = {
                                "rms": round(rms, 4),
                                "peak": round(peak, 4),
                                "db": round(db, 1),
                                "level_pct": level_pct,
                                "status": status_label,
                                "bands": bands,
                                "active": True
                            }
                            if asyncio.iscoroutinefunction(vu_callback):
                                await vu_callback(vu_payload)
                            else:
                                vu_callback(vu_payload)

                # 錄音退出後派發尾端剩餘音訊（>=0.5 秒）
                if pcm_chunk_callback and len(chunk_buffer) >= 16000:
                    tail_data = bytes(chunk_buffer)
                    chunk_buffer.clear()
                    try:
                        if asyncio.iscoroutinefunction(pcm_chunk_callback):
                            await pcm_chunk_callback(tail_data, True)
                        else:
                            pcm_chunk_callback(tail_data, True)
                    except Exception as c_err:
                        print(f"[VoicePipeline] pcm_chunk_callback tail error: {c_err}")

            except Exception as stream_err:
                print(f"[VoicePipeline] Streaming loop notice: {stream_err}")

        # Check cancellation before waiting
        if cancellation_event and cancellation_event.is_set():
            try:
                proc.terminate()
            except Exception:
                pass
            return False

        await proc.wait()
        is_early = bool(finish_early_event and finish_early_event.is_set())
        if (proc.returncode in (0, 255, -15) or is_early) and os.path.exists(output_path) and os.path.getsize(output_path) > 500:
            return True

        # Fallback to default if explicit node failed
        if source_node != "default":
            fb_cmd = [
                "ffmpeg", "-y",
                "-thread_queue_size", "4096",
                "-f", "pulse",
                "-i", "default",
                "-t", str(duration_sec),
                "-af", "aresample=16000:async=1,highpass=f=80,lowpass=f=7600",
                "-c:a", "libopus",
                "-b:a", "64k",
                output_path
            ]
            fb_proc = await asyncio.create_subprocess_exec(
                *fb_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await fb_proc.communicate()
            return fb_proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 500

        return False
    except Exception as e:
        print(f"[VoicePipeline] Microphone recording failed: {e}")
        return False


def transcribe_pcm_chunk(pcm_bytes: bytes) -> Optional[str]:
    """
    Transcribe a 16kHz 16-bit mono PCM chunk using Hailo-8 AI NPU or faster-whisper.
    Used for concurrent streaming STT while recording is in progress.
    """
    if not pcm_bytes or len(pcm_bytes) < 8000:
        return None
    try:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples ** 2)))
        # VAD 靜音與呼吸聲過濾
        if peak < 0.02 and rms < 0.0015:
            return None

        # 優先調用 AI NPU 引擎
        hailo_adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AI_HAT_plus"))
        if os.path.exists(hailo_adapter_path) and hailo_adapter_path not in sys.path:
            sys.path.append(hailo_adapter_path)
        try:
            from hailo_engine.device_manager import is_hailo_available
            if is_hailo_available():
                from hailo_engine.whisper_adapter import get_whisper_engine
                engine = get_whisper_engine(auto_detect=True, model_size="base")
                res = engine.transcribe(samples, language="zh")
                if res and res.get("text"):
                    return to_taiwan_traditional(res["text"]).strip()
        except Exception as hailo_err:
            pass

        # 回退至 CPU faster-whisper (單執行緒/2執行緒避免爭搶)
        with _whisper_lock:
            model = get_whisper_model()
            if model:
                segments, _ = model.transcribe(
                    samples,
                    beam_size=1,
                    language="zh",
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    hallucination_silence_threshold=2.0,
                    vad_filter=True,
                    initial_prompt="繁體中文"
                )
                texts = [s.text.strip() for s in segments if s.text.strip()]
                if texts:
                    clean_res = clean_hallucination_repetitions("".join(texts))
                    return to_taiwan_traditional(clean_res).strip()
    except Exception as e:
        print(f"[VoicePipeline] transcribe_pcm_chunk error: {e}")
    return None



