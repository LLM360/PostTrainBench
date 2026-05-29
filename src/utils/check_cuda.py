#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import torch

def get_gpu_processes(gpu_index):
    """Get processes running on a specific GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=" + str(gpu_index), 
             "--query-compute-apps=pid,used_memory", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                pid, mem = line.split(',')
                processes.append((int(pid.strip()), float(mem.strip())))
        return processes
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def check_h100():
    expected_gpus = int(os.environ.get("NUM_GPUS", "1"))

    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False

    device_count = torch.cuda.device_count()
    print(f"✓ CUDA available with {device_count} device(s) (expected {expected_gpus})")
    if device_count != expected_gpus:
        print(f"❌ Expected {expected_gpus} GPU(s), got {device_count}")
        return False

    # Every visible GPU must match the supported set. A partial match would
    # let a job run on a mixed allocation (e.g. one H100 + one unsupported
    # device), so we fail loudly and name the offending GPU.
    unsupported = []
    all_idle = True
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {name} ({props.total_memory / 1e9:.1f} GB)")

        if "H100" not in name and "H200" not in name:
            unsupported.append((i, name))
            # Skip the idle-process check for unsupported GPUs — the overall
            # function will fail below regardless.
            continue

        # Check for running processes on this GPU (supported GPUs only)
        processes = get_gpu_processes(i)
        if processes is None:
            print(f"  ⚠ Could not check processes on GPU {i} (nvidia-smi failed)")
        elif processes:
            print(f"  ❌ GPU {i} has {len(processes)} process(es) running:")
            for pid, mem in processes:
                print(f"      PID {pid}: {mem:.1f} MiB")
            all_idle = False
        else:
            print(f"  ✓ GPU {i} is idle")

    if unsupported:
        for idx, name in unsupported:
            print(f"❌ GPU {idx} is not an H100/H200: {name}")
        return False

    if not all_idle:
        return False

    print(f"✓ H100/H200 detected ({device_count} GPU(s))")
    return True

if __name__ == "__main__":
    cuda_available = check_h100()
    if not cuda_available:
        Path("cuda_not_available").touch()

    import sys
    sys.exit(0 if cuda_available else 1)
