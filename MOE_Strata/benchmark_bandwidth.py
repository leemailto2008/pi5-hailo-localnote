# -*- coding: utf-8 -*-
"""
MOE_Strata/benchmark_bandwidth.py
邊緣端記憶體頻寬 vs PCIe 吞吐瓶頸評估與 MoE / Dense 推論效能估算工具
"""

import sys
import io

# 避免 Windows CP950 編碼問題
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def calculate_token_throughput(model_size_gb: float, bandwidth_gb_s: float) -> dict:
    """
    計算 Memory-Bound (自回歸解碼) 的理論最高 Token 輸出速率 (Tokens/s)
    """
    io_time_per_token = model_size_gb / bandwidth_gb_s
    tokens_per_sec = 1.0 / io_time_per_token if io_time_per_token > 0 else 0
    return {
        "io_time_ms": io_time_per_token * 1000.0,
        "max_tokens_per_sec": tokens_per_sec
    }

def main():
    print("=" * 80)
    print(" 樹莓派 5 邊緣 AI 硬體與推論頻寬理論性能估算矩陣 (Hardware Bandwidth Matrix)")
    print("=" * 80)
    
    # 頻寬定義 (GB/s)
    pi5_ram_bw = 17.0         # LPDDR4X-4267
    pcie_gen2_bw = 0.42       # PCIe 2.0 x1 實測
    pcie_gen3_bw = 0.85       # PCIe 3.0 x1 實測
    hailo10_onboard_bw = 32.0 # Hailo-10H 板載 LPDDR4
    
    scenarios = [
        ("Dense 3B (Q4_K_M: 2.0 GB)", 2.0, "Pi 5 CPU (Host RAM)", pi5_ram_bw),
        ("Dense 7B (Q4_K_M: 4.5 GB)", 4.5, "Pi 5 CPU (Host RAM)", pi5_ram_bw),
        ("Dense 7B (Q4_K_M: 4.5 GB)", 4.5, "Hailo-8 經由 PCIe 3.0 x1 讀取", pcie_gen3_bw),
        ("Dense 7B (INT4: 4.5 GB)",   4.5, "Hailo-10H (板載 8GB LPDDR4)", hailo10_onboard_bw),
        ("MoE 14B (總參 8.0GB, 激活 2.2GB)", 2.2, "Pi 5 16GB + Strata (動態專家快取)", pi5_ram_bw),
    ]
    
    print(f"{'模型架構與負載':<34} | {'執行通道 / 記憶體':<24} | {'頻寬(GB/s)':<8} | {'理論極限 (Tokens/s)':<18}")
    print("-" * 92)
    
    for name, size_gb, channel, bw in scenarios:
        res = calculate_token_throughput(size_gb, bw)
        speed = res['max_tokens_per_sec']
        status = ""
        if speed < 1.0:
            status = " [FAIL - 嚴重卡死]"
        elif speed >= 7.0:
            status = " [PASS - 極速流暢]"
        else:
            status = " [WARN - 勉強可用]"
        print(f"{name:<34} | {channel:<24} | {bw:<8.2f} | {speed:<8.2f} Tok/s {status}")
        
    print("-" * 92)
    print("\n[架構結論分析]")
    print("1. Hailo-8 因無板載記憶體，若嘗試透過 PCIe 3.0 x1 載入 7B 權重，速度僅 0.19 Tokens/s，證實 Hailo-8 絕對不適合大型 Dense LLM。")
    print("2. Hailo-10H 憑藉板載 8GB LPDDR4 (32 GB/s 內部頻寬)，推論 7B 模型理論極限可達 7.11 Tokens/s，配合其 40 TOPS NPU 矩陣核心可實現流暢生成。")
    print("3. Pi 5 (16GB) 配合 Strata MoE 動態專家調度，由於單步活躍專家僅需讀取 2.2GB 權重，在 17 GB/s 系統頻寬下理論極限高達 7.73 Tokens/s，具備極高性價比與智慧上限！\n")

if __name__ == "__main__":
    main()
