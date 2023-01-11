import os 
import numpy as np

def getting_freer_gpu():

    os.system('nvidia-smi -q -d Memory | findstr Free > ./apps/gpuMemoryTmp')
    memory_available = [int(x.split()[2]) for x in open('./apps/gpuMemoryTmp', 'r').readlines()]
    if len(memory_available) >= 4 : memory_available = [memory_available[0], memory_available[2]]
    else: memory_available = [memory_available[0], memory_available[1]]
    # print(memory_available) # [10960, 11128]
    FreerGpuId = np.argmax(memory_available)
    return FreerGpuId # 0 or 1
