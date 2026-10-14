#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

def main():
    model_path = "/home/pi/AI_HAT_plus/models/base-whisper-encoder-5s.hef"
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return

    try:
        from hailo_platform import HEF, VDevice, ConfigureParams
        print(f"Loading HEF from {model_path}...")
        hef = HEF(model_path)
        print("✅ HEF successfully loaded!")
        print("Network groups:", [ng.name for ng in hef.get_network_groups_infos()])
        print("Input streams:", [v.name for v in hef.get_input_vstream_infos()])
        from hailo_platform import HailoStreamInterface, ConfigureParams, VDevice
        print("Testing VDevice configuration on PCIe Hailo-8...")
        target = VDevice()
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        print(f"✅ VDevice configuration successful! Configured {len(network_groups)} network groups.")
        for ng in network_groups:
            print("Configured Network Group:", ng.name)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
