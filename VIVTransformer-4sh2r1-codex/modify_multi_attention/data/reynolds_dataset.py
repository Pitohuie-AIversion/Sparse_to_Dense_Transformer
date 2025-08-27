"""
Reynolds流场数据集适配器
将4D张量数据 [reynolds_cases, time_steps, height, width] 转换为模型可用格式
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable
import h5py
from pathlib import Path


class ReynoldsFlowDataset(Dataset):
    """Reynolds流场数据集
    
    处理形状为 [reynolds_cases, time_steps, height, width] 的4D张量数据
    """
    
    def __init__(
        self, 
        data_path: str,
        input_size: int = 20,
        output_size: int = 200,
        transform: Optional[Callable] = None,
        use_time_sequence: bool = False,
        sequence_length: int = 5
    ):
        """
        Args:
            data_path: 数据文件路径 (.pt 或 .h5)
            input_size: 输入网格尺寸 (input_size × input_size)
            output_size: 输出网格尺寸 (output_size × output_size)
            transform: 数据变换函数
            use_time_sequence: 是否使用时间序列预测
            sequence_length: 时间序列长度
        """
        self.data_path = Path(data_path)
        self.input_size = input_size
        self.output_size = output_size
        self.transform = transform
        self.use_time_sequence = use_time_sequence
        self.sequence_length = sequence_length
        
        # 加载数据
        self._load_data()
        
        # 计算样本数量
        self._calculate_samples()
    
    def _load_data(self):
        """加载数据文件"""
        if self.data_path.suffix == '.pt':
            # PyTorch tensor文件
            data = torch.load(self.data_path)
            if isinstance(data, dict):
                # 假设数据保存为字典格式
                self.input_data = data.get('input_pressure', data.get('in_pressure'))
                self.output_data = data.get('output_pressure', data.get('pressure'))
                self.reynolds_numbers = data.get('reynolds_numbers', None)
                self.time_steps = data.get('time_steps', None)
            else:
                # 假设数据直接保存为tensor
                self.input_data = data
                self.output_data = data  # 如果没有分离，使用相同数据
                
        elif self.data_path.suffix == '.h5':
            # HDF5文件
            with h5py.File(self.data_path, 'r') as f:
                self.input_data = torch.from_numpy(f['input_pressure'][:])
                self.output_data = torch.from_numpy(f['output_pressure'][:])
                self.reynolds_numbers = f.get('reynolds_numbers', None)
                self.time_steps = f.get('time_steps', None)
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path.suffix}")
        
        # 确保数据是4D张量
        if len(self.input_data.shape) != 4:
            raise ValueError(f"输入数据应为4D张量，得到: {self.input_data.shape}")
        
        self.n_reynolds, self.n_timesteps, self.input_h, self.input_w = self.input_data.shape
        
        # 如果没有时间步信息，创建默认值
        if self.time_steps is None:
            self.time_steps = torch.arange(self.n_timesteps).float()
        
        print(f"数据加载完成:")
        print(f"  Reynolds数量: {self.n_reynolds}")
        print(f"  时间步数: {self.n_timesteps}")
        print(f"  输入尺寸: {self.input_h}×{self.input_w}")
        if hasattr(self, 'output_data'):
            print(f"  输出尺寸: {self.output_data.shape[-2]}×{self.output_data.shape[-1]}")
    
    def _calculate_samples(self):
        """计算总样本数"""
        if self.use_time_sequence:
            # 时间序列模式：每个Reynolds数的每个有效时间窗口是一个样本
            self.samples_per_reynolds = max(0, self.n_timesteps - self.sequence_length + 1)
            self.total_samples = self.n_reynolds * self.samples_per_reynolds
        else:
            # 单时间步模式：每个Reynolds数的每个时间步是一个样本
            self.total_samples = self.n_reynolds * self.n_timesteps
        
        print(f"总样本数: {self.total_samples}")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """获取样本
        
        Returns:
            input_flat: 展平的输入压力场 (input_size²,)
            output_flat: 展平的输出压力场 (output_size²,)
            time_step: 对应的时间步
        """
        if self.use_time_sequence:
            # 时间序列模式
            reynolds_idx = idx // self.samples_per_reynolds
            seq_start_idx = idx % self.samples_per_reynolds
            
            # 获取输入序列
            input_seq = self.input_data[
                reynolds_idx, 
                seq_start_idx:seq_start_idx + self.sequence_length
            ]  # (sequence_length, H, W)
            
            # 获取目标（序列的最后一个时间步）
            target_time_idx = seq_start_idx + self.sequence_length - 1
            time_step = self.time_steps[target_time_idx]
            
            # 使用序列的最后一帧作为输入（或可以修改为使用整个序列）
            input_pressure = input_seq[-1]  # (H, W)
            
        else:
            # 单时间步模式
            reynolds_idx = idx // self.n_timesteps
            time_idx = idx % self.n_timesteps
            
            input_pressure = self.input_data[reynolds_idx, time_idx]  # (H, W)
            time_step = self.time_steps[time_idx]
        
        # 获取对应的输出数据
        if hasattr(self, 'output_data') and self.output_data is not None:
            if self.use_time_sequence:
                output_pressure = self.output_data[reynolds_idx, target_time_idx]
            else:
                output_pressure = self.output_data[reynolds_idx, time_idx]
        else:
            # 如果没有分离的输出数据，使用输入数据
            output_pressure = input_pressure
        
        # 数据变换
        if self.transform:
            input_pressure, output_pressure = self.transform((input_pressure, output_pressure))
        
        # 展平数据
        input_flat = input_pressure.view(-1)  # (input_size²,)
        output_flat = output_pressure.view(-1)  # (output_size²,)
        
        return input_flat, output_flat, time_step.item() if hasattr(time_step, 'item') else float(time_step)
    
    def get_reynolds_info(self, idx: int) -> Tuple[int, int]:
        """获取样本对应的Reynolds数索引和时间步索引"""
        if self.use_time_sequence:
            reynolds_idx = idx // self.samples_per_reynolds
            seq_start_idx = idx % self.samples_per_reynolds
            time_idx = seq_start_idx + self.sequence_length - 1
        else:
            reynolds_idx = idx // self.n_timesteps
            time_idx = idx % self.n_timesteps
        
        return reynolds_idx, time_idx
    
    def get_spatial_dimensions(self) -> Tuple[int, int, int, int]:
        """获取空间维度信息"""
        output_h, output_w = self.output_data.shape[-2:] if hasattr(self, 'output_data') else (self.input_h, self.input_w)
        return self.input_h, self.input_w, output_h, output_w


def create_reynolds_loaders(
    data_path: str,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    use_time_sequence: bool = False,
    **kwargs
) -> Tuple:
    """创建Reynolds数据的训练、验证、测试加载器"""
    from torch.utils.data import DataLoader, random_split
    
    # 创建数据集 (同时先提取 DataLoader 相关参数，避免误传给数据集构造)
    # 先提取 DataLoader 相关参数，避免传入到数据集构造
    num_workers = kwargs.pop('num_workers', 0)
    pin_memory = kwargs.pop('pin_memory', torch.cuda.is_available())
    prefetch_factor = kwargs.pop('prefetch_factor', 2)
    persistent_workers = kwargs.pop('persistent_workers', None)
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    # 创建数据集（仅传入与数据集有关的参数）
    dataset = ReynoldsFlowDataset(
        data_path=data_path,
        use_time_sequence=use_time_sequence,
        **kwargs
    )
    
    # 计算分割大小
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 随机分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器 (Windows兼容性：默认num_workers=0；若需要可根据机器配置调大并设置 persistent_workers=True)
    # 上方已通过 kwargs.pop 提取了 DataLoader 相关参数: num_workers, pin_memory, prefetch_factor, persistent_workers

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None and num_workers > 0:
        common_kwargs['prefetch_factor'] = prefetch_factor

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        **common_kwargs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
        **common_kwargs
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
        **common_kwargs
    )
    
    return train_loader, val_loader, test_loader, dataset