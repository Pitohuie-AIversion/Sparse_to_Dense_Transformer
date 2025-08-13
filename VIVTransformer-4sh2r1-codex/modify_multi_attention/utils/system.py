import torch
import numpy as np
import random
import logging
import warnings


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


def apply_performance_optimizations(perf_config, device=None):
    """Apply performance optimizations based on configuration.
    
    Args:
        perf_config: Performance configuration dictionary
        device: Optional device info for hardware-specific optimizations
    """
    logger = logging.getLogger(__name__)
    
    if not perf_config:
        logger.info("No performance configuration provided, using defaults")
        return
    
    # Hardware optimizations
    hardware_config = perf_config.get('hardware', {})
    
    # Enable TF32 for Ampere GPUs
    if hardware_config.get('enable_tf32', False):
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✓ TF32 enabled for matrix multiplications")
        else:
            logger.warning("TF32 not available in this PyTorch version")
    
    # Set float32 matmul precision
    precision = hardware_config.get('set_float32_matmul_precision')
    if precision:
        try:
            torch.set_float32_matmul_precision(precision)
            logger.info(f"✓ Float32 matmul precision set to: {precision}")
        except Exception as e:
            logger.warning(f"Failed to set matmul precision: {e}")
    
    # Enable cuDNN benchmark
    if hardware_config.get('enable_cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        logger.info("✓ cuDNN benchmark enabled")
    
    # Memory optimizations
    memory_config = perf_config.get('memory', {})
    max_split_size = memory_config.get('max_split_size_mb')
    if max_split_size and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)  # Leave some headroom
            # Note: max_split_size_mb configuration would need custom CUDA allocator
            logger.info("✓ CUDA memory optimizations applied")
        except Exception as e:
            logger.warning(f"Failed to apply memory optimizations: {e}")
    
    # Experimental optimizations
    experimental_config = perf_config.get('experimental', {})
    
    if experimental_config.get('use_deterministic_algorithms', False):
        try:
            torch.use_deterministic_algorithms(True, warn_only=experimental_config.get('warn_only_deterministic', True))
            logger.info("✓ Deterministic algorithms enabled")
        except Exception as e:
            logger.warning(f"Failed to enable deterministic algorithms: {e}")
    
    # Enable anomaly detection for debugging
    if experimental_config.get('enable_anomaly_detection', False):
        torch.autograd.set_detect_anomaly(True)
        logger.info("✓ Autograd anomaly detection enabled")
    
    # CPU optimizations
    cpu_config = perf_config.get('hardware_specific', {}).get('cpu', {})
    num_threads = cpu_config.get('num_threads')
    if num_threads:
        torch.set_num_threads(num_threads)
        logger.info(f"✓ PyTorch threads set to: {num_threads}")
    
    # Enable MKL-DNN
    if cpu_config.get('enable_mkldnn', True):
        try:
            torch.backends.mkldnn.enabled = True
            logger.info("✓ MKL-DNN enabled")
        except AttributeError:
            logger.warning("MKL-DNN not available in this PyTorch build")
    
    logger.info("Performance optimizations applied successfully")


from datetime import datetime
import os

def create_timestamped_dir(base_dir):
    """Creates a timestamped directory for storing results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = os.path.join(base_dir, timestamp)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir