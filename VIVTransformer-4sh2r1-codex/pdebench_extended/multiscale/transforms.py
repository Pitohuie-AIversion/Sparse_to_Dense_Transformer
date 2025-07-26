"""数据变换模块

包含用于数据增强的各种变换类
"""

import torch
import torch.nn as nn
from typing import Union, Tuple


class AddGaussianNoise:
    """添加高斯噪声的数据变换
    
    Args:
        std: 噪声标准差
    """
    
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """对输入张量添加高斯噪声
        
        Args:
            tensor: 输入张量
            
        Returns:
            添加噪声后的张量
        """
        return tensor + torch.randn_like(tensor) * self.std
    
    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


class RandomCrop:
    """随机裁剪变换
    
    Args:
        size: 裁剪尺寸 (height, width)
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """随机裁剪张量
        
        Args:
            tensor: 输入张量，形状为 (C, H, W) 或 (H, W)
            
        Returns:
            裁剪后的张量
        """
        if tensor.dim() == 2:
            h, w = tensor.shape
        elif tensor.dim() == 3:
            _, h, w = tensor.shape
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
        
        target_h, target_w = self.size
        
        if h < target_h or w < target_w:
            raise ValueError(f"Input size ({h}, {w}) is smaller than target size {self.size}")
        
        # 随机选择起始位置
        start_h = torch.randint(0, h - target_h + 1, (1,)).item()
        start_w = torch.randint(0, w - target_w + 1, (1,)).item()
        
        if tensor.dim() == 2:
            return tensor[start_h:start_h + target_h, start_w:start_w + target_w]
        else:
            return tensor[:, start_h:start_h + target_h, start_w:start_w + target_w]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"