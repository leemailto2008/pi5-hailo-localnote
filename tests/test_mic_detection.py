"""voice_pipeline.ensure_bluetooth_recording_ready() 的來源選擇：耳麥沒連線又沒有其他麥克風時要報錯，
不能默默錄 Dummy Output 的 monitor (2026-10-29 228 耳麥自動關機後錄了整段靜音)。用假的 pw-dump，不需要 PipeWire。"""
import asyncio
import json
import sys
import types

import pytest

sys.modules.setdefault("config", types.SimpleNamespace(WHISPER_MODEL_SIZE="small", WHISPER_DEVICE="cpu", WHISPER_COMPUTE_TYPE="int8",
                                                       DEFAULT_TTS_VOICE="zh-TW-HsiaoChenNeural", DATA_DIR="/tmp"))
import voice_pipeline as vp  # noqa: E402


def _node(nid, kind, **props):
    return {"id": nid, "type": kind, "info": {"props": props, "params": {}}}


DUMMY_SINK = _node(80, "PipeWire:Interface:Node", **{"media.class": "Audio/Sink", "node.name": "auto_null"})
USB_MIC = _node(81, "PipeWire:Interface:Node", **{"media.class": "Audio/Source", "node.name": "alsa_input.usb-mic"})
BT_DEV = {"id": 90, "type": "PipeWire:Interface:Device", "info": {"props": {"device.api": "bluez5"},
          "params": {"Profile": [{"name": "headset-head-unit"}], "EnumProfile": [{"name": "headset-head-unit", "index": 1}]}}}
BT_SRC = _node(91, "PipeWire:Interface:Node", **{"media.class": "Audio/Source", "node.name": "bluez_input_internal.A0_D1_2A_42_12_34.0",
                                                  "api.bluez5.codec": "msbc"})


class _Proc:
    def __init__(self, out=b""):
        self.out = out

    async def communicate(self):
        return self.out, b""


@pytest.fixture()
def pw(monkeypatch):
    monkeypatch.setattr(vp, "_ensure_process_affinities", lambda: None)

    async def no_wait(node):
        return None
    monkeypatch.setattr(vp, "_wait_for_live_audio", no_wait)

    def use(nodes):
        async def fake_exec(*args, **kw):
            return _Proc(json.dumps(nodes).encode() if args[0] == "pw-dump" else b"")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return use


def test_no_headset_and_no_microphone_raises(pw):
    pw([DUMMY_SINK])
    with pytest.raises(vp.NoMicrophoneError):
        asyncio.run(vp.ensure_bluetooth_recording_ready())


def test_other_microphone_keeps_default(pw):
    pw([DUMMY_SINK, USB_MIC])
    assert asyncio.run(vp.ensure_bluetooth_recording_ready()) == "default"


def test_headset_connected_but_source_not_ready_keeps_default(pw):
    pw([DUMMY_SINK, BT_DEV])
    assert asyncio.run(vp.ensure_bluetooth_recording_ready()) == "default"


def test_native_bluetooth_source_is_selected(pw):
    pw([DUMMY_SINK, BT_DEV, BT_SRC])
    assert asyncio.run(vp.ensure_bluetooth_recording_ready()) == "bluez_input_internal.A0_D1_2A_42_12_34.0"
