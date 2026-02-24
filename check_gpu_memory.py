#!/usr/bin/env python3
"""
檢查 GPU 記憶體使用情況並提供清理建議
"""
import subprocess
import sys

def check_gpu_memory():
    """檢查 GPU 記憶體使用情況"""
    try:
        # 使用 nvidia-smi 檢查 GPU 記憶體
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("=" * 60)
        print("GPU 記憶體使用情況")
        print("=" * 60)
        print(result.stdout)
        
        # 檢查正在使用 GPU 的進程
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip():
            print("\n正在使用 GPU 的進程:")
            print("-" * 60)
            print(result.stdout)
            print("\n建議:")
            print("1. 如果看到其他訓練進程，可以考慮終止它們:")
            print("   kill <PID>")
            print("2. 或者等待它們完成")
            print("3. 如果必須同時運行，請進一步降低 batch_size")
        else:
            print("\n沒有其他進程使用 GPU")
            
    except subprocess.CalledProcessError as e:
        print(f"無法執行 nvidia-smi: {e}")
        print("請確保已安裝 NVIDIA 驅動程式")
    except FileNotFoundError:
        print("找不到 nvidia-smi 命令")
        print("請確保已安裝 NVIDIA 驅動程式")

if __name__ == "__main__":
    check_gpu_memory()
