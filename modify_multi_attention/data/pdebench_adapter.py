"""PDEBench数据集适配器

这个模块提供了将PDEBench数据集集成到现有多注意力机制项目中的适配器。
PDEBench是一个用于科学机器学习的综合基准数据集，包含多种偏微分方程问题。
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path


class PDEBenchDataset(Dataset):
    """PDEBench数据集适配器
    
    支持PDEBench中的多种PDE问题：
    - 1D Advection方程
    - 1D Burgers方程 
    - 1D Reaction-Diffusion方程
    - 2D Darcy Flow
    - 2D Navier-Stokes方程
    - 2D Shallow Water方程
    """
    
    def __init__(
        self,
        data_path: str,
        pde_type: str = "ns_incom",  # 默认使用2D不可压缩Navier-Stokes
        split: str = "train",
        sequence_length: int = 49,
        spatial_resolution: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        transform: Optional[callable] = None
    ):
        """
        初始化PDEBench数据集
        
        Args:
            data_path: PDEBench数据文件路径
            pde_type: PDE类型 ('ns_incom', 'darcy', 'swe', 'advection', 'burgers', 'diff_react')
            split: 数据集分割 ('train', 'val', 'test')
            sequence_length: 序列长度
            spatial_resolution: 空间分辨率 (height, width)
            normalize: 是否标准化数据
            transform: 数据变换函数
        """
        self.data_path = Path(data_path)
        self.pde_type = pde_type
        self.split = split
        self.sequence_length = sequence_length
        self.spatial_resolution = spatial_resolution
        self.normalize = normalize
        self.transform = transform
        
        self.logger = logging.getLogger(__name__)
        
        # 加载数据
        self._load_data()
        
        # 数据预处理
        if self.normalize:
            self._normalize_data()
    
    def _load_data(self):
        """加载PDEBench数据"""
        try:
            if self.data_path.suffix == '.h5':
                self._load_h5_data()
            elif self.data_path.suffix == '.pt':
                self._load_pt_data()
            else:
                raise ValueError(f"不支持的文件格式: {self.data_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def _load_h5_data(self):
        """加载HDF5格式的PDEBench数据"""
        with h5py.File(self.data_path, 'r') as f:
            # 根据PDE类型加载相应的数据字段
            if self.pde_type == "ns_incom":
                # 2D不可压缩Navier-Stokes
                self.input_data = torch.from_numpy(f['u'][:])
                self.target_data = torch.from_numpy(f['p'][:])
                self.time_steps = torch.from_numpy(f['t'][:])
                
            elif self.pde_type == "darcy":
                # 2D Darcy Flow
                self.input_data = torch.from_numpy(f['coeff'][:])
                self.target_data = torch.from_numpy(f['sol'][:])
                
            elif self.pde_type == "swe":
                # 2D Shallow Water
                self.input_data = torch.from_numpy(f['h'][:])
                self.target_data = torch.from_numpy(f['u'][:])
                self.time_steps = torch.from_numpy(f['t'][:])
                
            elif self.pde_type in ["advection", "burgers", "diff_react"]:
                # 1D方程
                self.input_data = torch.from_numpy(f['u'][:])
                self.target_data = torch.from_numpy(f['u'][:])
                self.time_steps = torch.from_numpy(f['t'][:])
                
            else:
                raise ValueError(f"不支持的PDE类型: {self.pde_type}")
    
    def _load_pt_data(self):
        """加载PyTorch格式的数据"""
        data = torch.load(self.data_path)
        
        # 适配不同的数据格式
        if isinstance(data, dict):
            self.input_data = data.get('input', data.get('u', None))
            self.target_data = data.get('target', data.get('p', None))
            self.time_steps = data.get('time_steps', data.get('t', None))
        else:
            # 假设数据是张量格式
            self.input_data = data
            self.target_data = data
            self.time_steps = None
    
    def _normalize_data(self):
        """数据标准化"""
        # 计算统计量
        self.input_mean = self.input_data.mean()
        self.input_std = self.input_data.std()
        self.target_mean = self.target_data.mean()
        self.target_std = self.target_data.std()
        
        # 标准化
        self.input_data = (self.input_data - self.input_mean) / (self.input_std + 1e-8)
        self.target_data = (self.target_data - self.target_mean) / (self.target_std + 1e-8)
    
    def _prepare_sequences(self, data: torch.Tensor) -> torch.Tensor:
        """准备序列数据"""
        if len(data.shape) < 3:
            # 如果数据维度不足，添加时间维度
            data = data.unsqueeze(1)
        
        # 确保序列长度
        if data.shape[1] < self.sequence_length:
            # 如果序列太短，进行填充
            pad_length = self.sequence_length - data.shape[1]
            data = torch.cat([data, data[:, -1:].repeat(1, pad_length, *([1] * (len(data.shape) - 2)))], dim=1)
        elif data.shape[1] > self.sequence_length:
            # 如果序列太长，进行截断
            data = data[:, :self.sequence_length]
        
        return data
    
    def _reshape_spatial_data(self, data: torch.Tensor) -> torch.Tensor:
        """重塑空间数据以适配模型输入"""
        if self.spatial_resolution is not None:
            target_h, target_w = self.spatial_resolution
            
            if len(data.shape) == 4:  # [batch, time, height, width]
                current_h, current_w = data.shape[-2:]
                
                if (current_h, current_w) != (target_h, target_w):
                    # 使用插值调整空间分辨率
                    data = torch.nn.functional.interpolate(
                        data.view(-1, 1, current_h, current_w),
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    ).view(data.shape[0], data.shape[1], target_h, target_w)
        
        return data
    
    def __len__(self) -> int:
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[float]]:
        """
        获取数据样本
        
        Returns:
            Tuple[input_tensor, target_tensor, time_step]
        """
        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]
        
        # 准备序列数据
        input_sample = self._prepare_sequences(input_sample.unsqueeze(0)).squeeze(0)
        target_sample = self._prepare_sequences(target_sample.unsqueeze(0)).squeeze(0)
        
        # 调整空间分辨率
        input_sample = self._reshape_spatial_data(input_sample.unsqueeze(0)).squeeze(0)
        target_sample = self._reshape_spatial_data(target_sample.unsqueeze(0)).squeeze(0)
        
        # 展平空间维度以适配现有模型
        if len(input_sample.shape) > 2:
            input_sample = input_sample.flatten(start_dim=1)
        if len(target_sample.shape) > 2:
            target_sample = target_sample.flatten(start_dim=1)
        
        # 获取时间步
        time_step = None
        if self.time_steps is not None:
            time_step = float(self.time_steps[idx]) if idx < len(self.time_steps) else 0.0
        
        # 应用变换
        if self.transform:
            input_sample, target_sample = self.transform((input_sample, target_sample))
        
        return input_sample, target_sample, time_step
    
    def get_data_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'pde_type': self.pde_type,
            'split': self.split,
            'num_samples': len(self),
            'sequence_length': self.sequence_length,
            'input_shape': self.input_data.shape,
            'target_shape': self.target_data.shape,
            'spatial_resolution': self.spatial_resolution,
            'normalized': self.normalize
        }


class PDEBenchDataLoader:
    """PDEBench数据加载器工厂类"""
    
    SUPPORTED_PDES = {
        'ns_incom': '2D不可压缩Navier-Stokes方程',
        'darcy': '2D Darcy Flow',
        'swe': '2D Shallow Water方程',
        'advection': '1D Advection方程',
        'burgers': '1D Burgers方程',
        'diff_react': '1D Reaction-Diffusion方程'
    }
    
    @classmethod
    def create_dataset(
        cls,
        data_path: str,
        pde_type: str = "ns_incom",
        **kwargs
    ) -> PDEBenchDataset:
        """创建PDEBench数据集"""
        if pde_type not in cls.SUPPORTED_PDES:
            raise ValueError(
                f"不支持的PDE类型: {pde_type}. "
                f"支持的类型: {list(cls.SUPPORTED_PDES.keys())}"
            )
        
        return PDEBenchDataset(data_path=data_path, pde_type=pde_type, **kwargs)
    
    @classmethod
    def get_default_config(cls, pde_type: str) -> Dict:
        """获取不同PDE类型的默认配置"""
        configs = {
            'ns_incom': {
                'sequence_length': 49,
                'spatial_resolution': (64, 64),
                'input_dim': 4096,  # 64*64
                'output_dim': 4096
            },
            'darcy': {
                'sequence_length': 1,
                'spatial_resolution': (85, 85),
                'input_dim': 7225,  # 85*85
                'output_dim': 7225
            },
            'swe': {
                'sequence_length': 49,
                'spatial_resolution': (128, 128),
                'input_dim': 16384,  # 128*128
                'output_dim': 16384
            },
            'advection': {
                'sequence_length': 49,
                'spatial_resolution': None,
                'input_dim': 1024,
                'output_dim': 1024
            },
            'burgers': {
                'sequence_length': 49,
                'spatial_resolution': None,
                'input_dim': 1024,
                'output_dim': 1024
            },
            'diff_react': {
                'sequence_length': 49,
                'spatial_resolution': None,
                'input_dim': 1024,
                'output_dim': 1024
            }
        }
        
        return configs.get(pde_type, configs['ns_incom'])