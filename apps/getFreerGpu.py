import subprocess
import numpy as np
import torch

def getting_freer_gpu():

    if torch.cuda.is_available():
        p = subprocess.Popen(f"nvidia-smi -q -d Memory | findstr Free > ./apps/gpuMemoryTmp", shell=True, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()
        memory_available = [int(x.split()[2]) for x in open('./apps/gpuMemoryTmp', 'r').readlines()]
        if len(memory_available) >= 4 : memory_available = [memory_available[0], memory_available[2]]
        else: memory_available = [memory_available[0], memory_available[1]]
        # print(memory_available) # [10960, 11128]
        FreerGpuId = np.argmax(memory_available)
    else:
        FreerGpuId = -1

    return FreerGpuId # 0 or 1 or -1 (No GPU)
