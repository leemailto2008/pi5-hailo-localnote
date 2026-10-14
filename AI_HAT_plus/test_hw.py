# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from hailo_engine import is_hailo_available, get_whisper_engine

print("=========================================")
print("Hailo NPU Hardware Detected:", is_hailo_available())
engine = get_whisper_engine()
print("Whisper Engine in use:", engine.get_engine_type())
print("=========================================")
