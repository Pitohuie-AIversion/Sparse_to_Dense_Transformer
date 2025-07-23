#!/usr/bin/env python3
"""压力场数据适配器

专门处理20x20→200x200压力场重建任务的数据适配器
将原始压力场数据格式集成到PDEBench统一框架中
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata

# 导入原始数据集类
from .dataset import PressureDataset


class PressureFieldDataset(Dataset):
    """压力场重建数据集
    
    专门处理20x20→200x200压力场数据，支持：
    - 数据加载和预处理
    - 多种归一化方法
    - 数据增强
    - 物理约束验证
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        normalize: bool = True,
        normalize_method: str = 'minmax',
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        augmentation: bool = False,
        physics_validation: bool = True
    ):
        """
        初始化压力场数据集
        
        Args:
            data_path: 数据文件路径（.pt格式）
            split: 数据分割类型 ('train', 'val', 'test')
            normalize: 是否归一化数据
            normalize_method: 归一化方法 ('minmax', 'zscore', 'robust')
            transform: 输入数据变换
            target_transform: 目标数据变换
            augmentation: 是否启用数据增强
            physics_validation: 是否启用物理约束验证
        """
        self.data_path = Path(data_path)
        self.split = split
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation
        self.physics_validation = physics_validation
        
        # 验证文件存在
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        # 加载原始数据集
        self.original_dataset = PressureDataset(str(self.data_path))
        
        # 获取数据信息
        self._analyze_data()
        
        # 创建数据分割
        self._create_splits()
        
        # 初始化归一化器
        if self.normalize:
            self._setup_normalizers()
        
        logging.info(f"压力场数据集初始化完成: {self.split}, 样本数: {len(self)}")
    
    def _analyze_data(self):
        """分析数据特征"""
        # 获取第一个样本来分析数据结构
        sample_input, sample_output, sample_time = self.original_dataset[0]
        
        self.input_dim = sample_input.shape[0]  # 400 (20x20)
        self.output_dim = sample_output.shape[0]  # 40000 (200x200)
        self.input_shape = [20, 20]
        self.output_shape = [200, 200]
        
        # 计算数据统计信息
        self.total_samples = len(self.original_dataset)
        
        logging.info(f"数据分析完成:")
        logging.info(f"  输入维度: {self.input_dim} ({self.input_shape})")
        logging.info(f"  输出维度: {self.output_dim} ({self.output_shape})")
        logging.info(f"  总样本数: {self.total_samples}")
    
    def _create_splits(self):
        """创建数据分割"""
        # 数据分割比例
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        
        # 计算分割大小
        train_size = int(train_ratio * self.total_samples)
        val_size = int(val_ratio * self.total_samples)
        test_size = self.total_samples - train_size - val_size
        
        # 创建随机索引
        indices = torch.randperm(self.total_samples).tolist()
        
        # 分割索引
        if self.split == 'train':
            self.indices = indices[:train_size]
        elif self.split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        elif self.split == 'test':
            self.indices = indices[train_size + val_size:]
        else:
            raise ValueError(f"不支持的数据分割类型: {self.split}")
        
        logging.info(f"数据分割完成: {self.split} - {len(self.indices)} 样本")
    
    def _setup_normalizers(self):
        """设置归一化器"""
        if self.split != 'train':
            return  # 只在训练集上计算归一化参数
        
        # 收集训练数据用于计算归一化参数
        inputs = []
        outputs = []
        
        for idx in self.indices[:min(1000, len(self.indices))]:  # 使用前1000个样本计算统计量
            input_data, output_data, _ = self.original_dataset[idx]
            inputs.append(input_data.numpy())
            outputs.append(output_data.numpy())
        
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        
        # 创建归一化器
        if self.normalize_method == 'minmax':
            self.input_normalizer = MinMaxScaler()
            self.output_normalizer = MinMaxScaler()
        elif self.normalize_method == 'zscore':
            self.input_normalizer = StandardScaler()
            self.output_normalizer = StandardScaler()
        else:
            raise ValueError(f"不支持的归一化方法: {self.normalize_method}")
        
        # 拟合归一化器
        self.input_normalizer.fit(inputs)
        self.output_normalizer.fit(outputs)
        
        logging.info(f"归一化器设置完成: {self.normalize_method}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """获取数据样本"""
        # 获取真实索引
        real_idx = self.indices[idx]
        
        # 从原始数据集获取数据
        input_data, output_data, time_step = self.original_dataset[real_idx]
        
        # 转换为numpy数组
        input_np = input_data.numpy() if isinstance(input_data, torch.Tensor) else input_data
        output_np = output_data.numpy() if isinstance(output_data, torch.Tensor) else output_data
        
        # 重塑为2D形状
        input_2d = input_np.reshape(20, 20)
        output_2d = output_np.reshape(200, 200)
        
        # 数据增强（仅训练时）
        if self.augmentation and self.split == 'train':
            input_2d, output_2d = self._apply_augmentation(input_2d, output_2d)
        
        # 归一化
        if self.normalize and hasattr(self, 'input_normalizer'):
            input_flat = input_2d.flatten().reshape(1, -1)
            output_flat = output_2d.flatten().reshape(1, -1)
            
            input_normalized = self.input_normalizer.transform(input_flat).flatten()
            output_normalized = self.output_normalizer.transform(output_flat).flatten()
            
            input_2d = input_normalized.reshape(20, 20)
            output_2d = output_normalized.reshape(200, 200)
        
        # 物理约束验证
        if self.physics_validation:
            self._validate_physics(input_2d, output_2d)
        
        # 应用变换
        if self.transform:
            input_2d = self.transform(input_2d)
        
        if self.target_transform:
            output_2d = self.target_transform(output_2d)
        
        # 转换回张量并展平
        input_tensor = torch.FloatTensor(input_2d.flatten())
        output_tensor = torch.FloatTensor(output_2d.flatten())
        
        return input_tensor, output_tensor, float(time_step)
    
    def _apply_augmentation(self, input_2d: np.ndarray, output_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用数据增强"""
        # 随机翻转
        if np.random.random() < 0.5:
            input_2d = np.fliplr(input_2d)
            output_2d = np.fliplr(output_2d)
        
        if np.random.random() < 0.5:
            input_2d = np.flipud(input_2d)
            output_2d = np.flipud(output_2d)
        
        # 随机旋转（小角度）
        if np.random.random() < 0.3:
            angle = np.random.uniform(-15, 15)
            input_2d = ndimage.rotate(input_2d, angle, reshape=False, mode='nearest')
            output_2d = ndimage.rotate(output_2d, angle, reshape=False, mode='nearest')
        
        # 添加高斯噪声
        if np.random.random() < 0.3:
            noise_std = 0.01 * np.std(input_2d)
            input_2d += np.random.normal(0, noise_std, input_2d.shape)
        
        return input_2d, output_2d
    
    def _validate_physics(self, input_2d: np.ndarray, output_2d: np.ndarray):
        """验证物理约束"""
        # 检查数值范围
        if np.any(np.isnan(input_2d)) or np.any(np.isnan(output_2d)):
            logging.warning("检测到NaN值")
        
        if np.any(np.isinf(input_2d)) or np.any(np.isinf(output_2d)):
            logging.warning("检测到无穷大值")
        
        # 检查压力场的连续性（梯度不应过大）
        grad_x, grad_y = np.gradient(output_2d)
        max_gradient = np.max(np.sqrt(grad_x**2 + grad_y**2))
        
        if max_gradient > 10.0:  # 阈值可调
            logging.warning(f"检测到异常大的梯度: {max_gradient}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            'data_type': 'pressure_field_reconstruction',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'total_samples': self.total_samples,
            'split_samples': len(self.indices),
            'split': self.split,
            'normalize': self.normalize,
            'normalize_method': self.normalize_method,
            'augmentation': self.augmentation,
            'upsampling_factor': 10,  # 20x20 -> 200x200
            'task_type': 'spatial_super_resolution',
            'physics_domain': 'fluid_dynamics'
        }
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """可视化数据样本"""
        input_tensor, output_tensor, time_step = self[idx]
        
        # 重塑为2D
        input_2d = input_tensor.numpy().reshape(20, 20)
        output_2d = output_tensor.numpy().reshape(200, 200)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 输入压力场
        im1 = axes[0].imshow(input_2d, cmap='coolwarm', interpolation='nearest')
        axes[0].set_title(f'输入压力场 (20x20)\nt={time_step:.3f}')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0])
        
        # 输出压力场
        im2 = axes[1].imshow(output_2d, cmap='coolwarm', interpolation='nearest')
        axes[1].set_title(f'输出压力场 (200x200)\nt={time_step:.3f}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class PressureFieldDataLoader:
    """压力场数据加载器
    
    提供便捷的数据加载接口
    """
    
    def __init__(
        self,
        dataset: PressureFieldDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """自定义批处理函数"""
        inputs, outputs, time_steps = zip(*batch)
        
        inputs = torch.stack(inputs)
        outputs = torch.stack(outputs)
        time_steps = torch.FloatTensor(time_steps)
        
        return inputs, outputs, time_steps
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_pressure_field_datasets(
    data_path: str,
    batch_size: int = 32,
    normalize: bool = True,
    normalize_method: str = 'minmax',
    augmentation: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[PressureFieldDataLoader, PressureFieldDataLoader, PressureFieldDataLoader]:
    """创建压力场数据集的便捷函数
    
    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        normalize: 是否归一化
        normalize_method: 归一化方法
        augmentation: 是否启用数据增强
        num_workers: 工作进程数
        pin_memory: 是否固定内存
    
    Returns:
        训练、验证、测试数据加载器的元组
    """
    # 创建数据集
    train_dataset = PressureFieldDataset(
        data_path=data_path,
        split='train',
        normalize=normalize,
        normalize_method=normalize_method,
        augmentation=augmentation
    )
    
    val_dataset = PressureFieldDataset(
        data_path=data_path,
        split='val',
        normalize=normalize,
        normalize_method=normalize_method,
        augmentation=False  # 验证集不使用数据增强
    )
    
    test_dataset = PressureFieldDataset(
        data_path=data_path,
        split='test',
        normalize=normalize,
        normalize_method=normalize_method,
        augmentation=False  # 测试集不使用数据增强
    )
    
    # 创建数据加载器
    train_loader = PressureFieldDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = PressureFieldDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = PressureFieldDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logging.info("压力场数据集创建完成")
    logging.info(f"训练集: {len(train_dataset)} 样本")
    logging.info(f"验证集: {len(val_dataset)} 样本")
    logging.info(f"测试集: {len(test_dataset)} 样本")
    
    return train_loader, val_loader, test_loader


def analyze_pressure_field_data(data_path: str) -> Dict[str, Any]:
    """分析压力场数据的统计特性
    
    Args:
        data_path: 数据文件路径
    
    Returns:
        数据分析结果字典
    """
    # 创建临时数据集进行分析
    dataset = PressureFieldDataset(
        data_path=data_path,
        split='train',
        normalize=False  # 分析原始数据
    )
    
    # 收集统计信息
    input_values = []
    output_values = []
    time_steps = []
    
    # 采样分析（避免内存溢出）
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    for idx in indices:
        input_data, output_data, time_step = dataset[idx]
        input_values.extend(input_data.numpy().flatten())
        output_values.extend(output_data.numpy().flatten())
        time_steps.append(time_step)
    
    input_values = np.array(input_values)
    output_values = np.array(output_values)
    time_steps = np.array(time_steps)
    
    # 计算统计量
    analysis_result = {
        'dataset_info': {
            'data_type': 'pressure_field_reconstruction',
            'input_dim': dataset.input_dim,
            'output_dim': dataset.output_dim,
            'input_shape': dataset.input_shape,
            'output_shape': dataset.output_shape,
            'total_samples': dataset.total_samples,
            'split_samples': len(dataset.indices),
            'split': dataset.split,
            'normalize': dataset.normalize,
            'normalize_method': dataset.normalize_method,
            'augmentation': dataset.augmentation,
            'upsampling_factor': 10,  # 20x20 -> 200x200
            'task_type': 'spatial_super_resolution',
            'physics_domain': 'fluid_dynamics'
        },
        'input_statistics': {
            'mean': float(np.mean(input_values)),
            'std': float(np.std(input_values)),
            'min': float(np.min(input_values)),
            'max': float(np.max(input_values)),
            'median': float(np.median(input_values)),
            'q25': float(np.percentile(input_values, 25)),
            'q75': float(np.percentile(input_values, 75))
        },
        'output_statistics': {
            'mean': float(np.mean(output_values)),
            'std': float(np.std(output_values)),
            'min': float(np.min(output_values)),
            'max': float(np.max(output_values)),
            'median': float(np.median(output_values)),
            'q25': float(np.percentile(output_values, 25)),
            'q75': float(np.percentile(output_values, 75))
        },
        'time_statistics': {
            'mean': float(np.mean(time_steps)),
            'std': float(np.std(time_steps)),
            'min': float(np.min(time_steps)),
            'max': float(np.max(time_steps)),
            'unique_count': len(np.unique(time_steps))
        },
        'sample_size': sample_size,
        'total_samples': len(dataset)
    }
    
    return analysis_result