import torch
import numpy as np
import random

def set_seed(seed, deterministic=False):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 仅在 CUDA 可用时设置 GPU 相关随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_cuda_memory_limit(fraction, device_idx=0):
    """Set CUDA memory limit for the current process.
    在无 CUDA 支持时直接跳过。
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_idx)


from datetime import datetime
import os

def create_timestamped_dir(base_dir):
    """Creates a timestamped directory for storing results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = os.path.join(base_dir, timestamp)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir