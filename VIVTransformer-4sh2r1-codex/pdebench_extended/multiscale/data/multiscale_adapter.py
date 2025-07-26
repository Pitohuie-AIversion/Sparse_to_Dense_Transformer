#!/usr/bin/env python3
"""多尺度数据适配器

实现从局部到整体的升阶重构：
- 支持将原始数据（如128x128）按整数倍缩小作为输入
- 预测回原始分辨率，实现超分辨率重构
- 支持多种缩放比例：2x, 4x, 8x等
"""

import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 导入现有的适配器
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from data.pdebench_adapter import PDEBenchDataset
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pdebench_adapter", 
        project_root / "data" / "pdebench_adapter.py"
    )
    pdebench_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdebench_module)
    PDEBenchDataset = pdebench_module.PDEBenchDataset


class MultiScaleDataset(Dataset):
    """多尺度数据集类
    
    支持将高分辨率数据缩小到低分辨率作为输入，
    然后预测回原始高分辨率，实现超分辨率重构。
    """
    
    def __init__(
        self,
        data_path: str,
        scale_factor: int = 2,
        pde_type: str = "darcy_flow",
        split: str = "train",
        sequence_length: int = 1,
        original_resolution: Optional[List[int]] = None,
        normalize: bool = True,
        downsampling_method: str = "average",  # "average", "bilinear", "nearest", "center_crop"
        enable_center_crop: bool = False,  # 是否启用输入输出的中心截取
        center_crop_input_resolution: Optional[List[int]] = None,  # 输入中心截取分辨率
        center_crop_output_resolution: Optional[List[int]] = None,  # 输出中心截取分辨率
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None
    ):
        """
        初始化多尺度数据集
        
        Args:
            data_path: 数据文件路径
            scale_factor: 缩放因子（2表示缩小2倍，4表示缩小4倍等）
            pde_type: PDE类型
            split: 数据分割类型 ('train', 'val', 'test')
            sequence_length: 序列长度
            original_resolution: 原始分辨率 [H, W]
            normalize: 是否归一化数据
            downsampling_method: 下采样方法
            enable_center_crop: 是否启用输入输出的中心截取
            center_crop_input_resolution: 输入中心截取分辨率 [H, W]
            center_crop_output_resolution: 输出中心截取分辨率 [H, W]
            transform: 输入数据变换
            target_transform: 目标数据变换
        """
        self.data_path = Path(data_path)
        self.scale_factor = scale_factor
        self.pde_type = pde_type
        self.split = split
        self.sequence_length = sequence_length
        self.original_resolution = original_resolution or [128, 128]
        self.normalize = normalize
        self.downsampling_method = downsampling_method
        self.enable_center_crop = enable_center_crop
        self.center_crop_input_resolution = center_crop_input_resolution
        self.center_crop_output_resolution = center_crop_output_resolution
        self.transform = transform
        self.target_transform = target_transform
        
        # 计算低分辨率
        self.low_resolution = [
            self.original_resolution[0] // scale_factor,
            self.original_resolution[1] // scale_factor
        ]
        
        # 验证分辨率
        if self.original_resolution[0] % scale_factor != 0 or self.original_resolution[1] % scale_factor != 0:
            raise ValueError(
                f"原始分辨率 {self.original_resolution} 不能被缩放因子 {scale_factor} 整除"
            )
        
        # 使用PDEBench适配器加载原始数据
        self.base_dataset = PDEBenchDataset(
            data_path=data_path,
            pde_type=pde_type,
            split=split,
            sequence_length=sequence_length,
            spatial_resolution=self.original_resolution,
            normalize=False,  # 我们自己处理归一化
            transform=None,
            target_transform=None
        )
        
        # 获取原始数据信息
        self.data_info = self.base_dataset.get_data_info()
        
        # 计算维度
        self.n_channels = self.data_info['n_channels']
        self.input_dim = self.low_resolution[0] * self.low_resolution[1] * self.n_channels
        self.output_dim = self.original_resolution[0] * self.original_resolution[1] * self.n_channels
        
        # 归一化统计量
        if self.normalize:
            self._compute_normalization_stats()
        
        logging.info(
            f"多尺度数据集初始化完成: {self.pde_type}, 缩放因子: {scale_factor}x, "
            f"输入分辨率: {self.low_resolution}, 输出分辨率: {self.original_resolution}, "
            f"样本数: {len(self)}"
        )
    
    def _compute_normalization_stats(self):
        """计算归一化统计量
        
        重要：为了确保输入和输出数据的数量级一致，我们基于原始数据计算统计量，
        而不是分别为输入和输出计算不同的统计量。这样可以保证预测的准确性。
        """
        if hasattr(self.base_dataset, 'data_mean') and hasattr(self.base_dataset, 'data_std'):
            self.data_mean = self.base_dataset.data_mean
            self.data_std = self.base_dataset.data_std
            print(f"使用基础数据集的归一化统计量: mean={self.data_mean:.6f}, std={self.data_std:.6f}")
        else:
            # 计算全局统计量 - 基于原始高分辨率数据
            all_data = []
            for i in range(min(len(self.base_dataset), 1000)):  # 使用前1000个样本计算统计量
                inputs, targets, _ = self.base_dataset[i]
                all_data.append(targets.numpy())
            
            all_data = np.concatenate(all_data, axis=0)
            self.data_mean = np.mean(all_data)
            self.data_std = np.std(all_data)
            
            if self.data_std == 0:
                self.data_std = 1.0
            
            print(f"计算得到的归一化统计量: mean={self.data_mean:.6f}, std={self.data_std:.6f}")
            print(f"注意：输入和输出数据将使用相同的归一化参数，确保数量级一致")
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """归一化数据"""
        if self.normalize:
            return (data - self.data_mean) / self.data_std
        return data
    
    def _denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """反归一化数据"""
        if self.normalize:
            return data * self.data_std + self.data_mean
        return data
    
    def _downsample_data(self, data: torch.Tensor) -> torch.Tensor:
        """下采样数据
        
        Args:
            data: 输入数据，形状为 [T, H*W*C] 或 [T, H, W, C]
        
        Returns:
            下采样后的数据，形状为 [T, H_low*W_low*C]
        """
        # 重塑为 [T, H, W, C]
        if len(data.shape) == 2:  # [T, H*W*C]
            T, spatial_dim = data.shape
            H, W = self.original_resolution
            C = self.n_channels
            data = data.view(T, H, W, C)
        
        T, H, W, C = data.shape
        
        # 转换为 [T*C, 1, H, W] 格式用于PyTorch的下采样函数
        data = data.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        data = data.view(T * C, 1, H, W)  # [T*C, 1, H, W]
        
        # 下采样
        if self.downsampling_method == "average":
            # 平均池化
            downsampled = F.avg_pool2d(
                data, 
                kernel_size=self.scale_factor, 
                stride=self.scale_factor
            )
        elif self.downsampling_method == "bilinear":
            # 双线性插值
            downsampled = F.interpolate(
                data,
                size=self.low_resolution,
                mode='bilinear',
                align_corners=False
            )
        elif self.downsampling_method == "nearest":
            # 最近邻插值
            downsampled = F.interpolate(
                data,
                size=self.low_resolution,
                mode='nearest'
            )
        elif self.downsampling_method == "center_crop":
            # 中心截取方法：从原数据集中心截取低分辨率区域
            T_C, _, H, W = data.shape
            H_low, W_low = self.low_resolution
            
            # 确保截取区域不超出原始数据边界
            if H_low > H or W_low > W:
                raise ValueError(f"截取分辨率 {self.low_resolution} 不能大于原始分辨率 {[H, W]}")
            
            # 计算中心截取的起始位置
            start_h = (H - H_low) // 2
            start_w = (W - W_low) // 2
            end_h = start_h + H_low
            end_w = start_w + W_low
            
            # 执行中心截取
            downsampled = data[:, :, start_h:end_h, start_w:end_w]
        else:
            raise ValueError(f"不支持的下采样方法: {self.downsampling_method}")
        
        # 转换回 [T, H_low*W_low*C]
        T_C, _, H_low, W_low = downsampled.shape
        downsampled = downsampled.view(T, C, H_low, W_low)  # [T, C, H_low, W_low]
        downsampled = downsampled.permute(0, 2, 3, 1).contiguous()  # [T, H_low, W_low, C]
        downsampled = downsampled.view(T, H_low * W_low * C)  # [T, H_low*W_low*C]
        
        return downsampled
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.base_dataset)
    
    def _center_crop_data(self, data: torch.Tensor, target_resolution: List[int]) -> torch.Tensor:
        """对数据进行中心截取
        
        Args:
            data: 输入数据 [T, H*W*C]
            target_resolution: 目标分辨率 [H, W]
        
        Returns:
            center_cropped_data: 中心截取后的数据 [T, H_target*W_target*C]
        """
        T, HWC = data.shape
        H, W = self.original_resolution
        C = self.n_channels
        H_target, W_target = target_resolution
        
        # 重塑为 [T, H, W, C]
        data = data.view(T, H, W, C)
        
        # 计算中心截取的起始位置
        start_h = (H - H_target) // 2
        start_w = (W - W_target) // 2
        end_h = start_h + H_target
        end_w = start_w + W_target
        
        # 执行中心截取
        cropped_data = data[:, start_h:end_h, start_w:end_w, :]
        
        # 重塑回 [T, H_target*W_target*C]
        cropped_data = cropped_data.reshape(T, H_target * W_target * C)
        
        return cropped_data
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        获取数据项
        
        优化策略：先处理输出数据并记录归一化参数，然后用相同参数处理输入数据
        这样确保输入输出数据的分布一致性，避免训练不稳定
        
        Returns:
            inputs: 输入数据 [T, H_input*W_input*C]
            targets: 目标数据 [T, H_target*W_target*C]
            time_steps: 时间步信息
        """
        # 从基础数据集获取原始数据
        original_inputs, original_targets, time_steps = self.base_dataset[idx]
        
        # 步骤1: 处理目标数据（输出）- 直接使用原始128x128数据，不进行裁剪
        targets = original_targets
        output_resolution = self.original_resolution
        
        # 步骤2: 归一化目标数据并记录当前样本的统计信息
        targets_numpy = targets.numpy()
        
        # 使用全局归一化参数处理目标数据
        targets_normalized = self._normalize_data(targets_numpy)
        targets = torch.from_numpy(targets_normalized).float()
        
        # 记录目标数据的处理参数，用于输入数据的一致性处理
        target_processing_info = {
            'output_resolution': output_resolution,
            'normalization_mean': self.data_mean,
            'normalization_std': self.data_std,
            'original_shape': targets_numpy.shape
        }
        
        # 步骤3: 处理输入数据 - 直接从原始128x128数据中心裁剪9x9
        if hasattr(self, 'enable_center_crop') and self.enable_center_crop and self.center_crop_input_resolution:
            # 直接从原始数据进行中心裁剪生成输入
            inputs = self._center_crop_data(original_targets, self.center_crop_input_resolution)
            
            # 使用相同的归一化参数处理输入数据
            inputs_normalized = self._normalize_data(inputs.numpy())
            inputs = torch.from_numpy(inputs_normalized).float()
        else:
            # 标准下采样模式：确保使用相同的归一化参数
            # 先对原始输入数据进行归一化，保持与目标数据的一致性
            inputs_raw = self._downsample_data(targets if isinstance(targets, torch.Tensor) else torch.from_numpy(targets))
            inputs_normalized = self._normalize_data(inputs_raw.numpy())
            inputs = torch.from_numpy(inputs_normalized).float()
        
        # 应用变换
        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            targets = self.target_transform(targets)
        
        return inputs, targets, time_steps
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        base_info = self.data_info.copy()
        base_info.update({
            'scale_factor': self.scale_factor,
            'original_resolution': self.original_resolution,
            'low_resolution': self.low_resolution,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'downsampling_method': self.downsampling_method,
            'data_mean': getattr(self, 'data_mean', None),
            'data_std': getattr(self, 'data_std', None),
            'task_type': 'super_resolution'
        })
        return base_info
    
    def visualize_sample(self, idx: int = 0) -> Dict[str, np.ndarray]:
        """可视化样本数据
        
        Args:
            idx: 样本索引
        
        Returns:
            包含输入、目标和重塑后数据的字典
        """
        inputs, targets, _ = self[idx]
        
        # 反归一化
        inputs_denorm = self._denormalize_data(inputs.numpy())
        targets_denorm = self._denormalize_data(targets.numpy())
        
        # 重塑为2D格式用于可视化
        def reshape_to_2d(data, resolution):
            if len(data.shape) == 2:  # [T, H*W*C]
                T = data.shape[0]
                H, W = resolution
                C = self.n_channels
                data = data.reshape(T, H, W, C)
                if C == 1:
                    return data[0, :, :, 0]  # 取第一个时间步和通道
                else:
                    return np.mean(data[0], axis=-1)  # 平均所有通道
            return data
        
        input_2d = reshape_to_2d(inputs_denorm, self.low_resolution)
        target_2d = reshape_to_2d(targets_denorm, self.original_resolution)
        
        return {
            'input_low_res': input_2d,
            'target_high_res': target_2d,
            'input_shape': input_2d.shape,
            'target_shape': target_2d.shape,
            'scale_factor': self.scale_factor
        }


class MultiScaleDataLoader(DataLoader):
    """多尺度数据加载器"""
    
    def __init__(
        self,
        dataset: MultiScaleDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        """初始化多尺度数据加载器"""
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
            **kwargs
        )
        
        self.dataset_info = dataset.get_data_info()
    
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """自定义批次整理函数"""
        # 过滤None值
        batch = [item for item in batch if item is not None]
        
        if not batch:
            return None, None, None
        
        # 分离输入、目标和时间步
        inputs, targets, time_steps = zip(*batch)
        
        # 堆叠张量
        inputs = torch.stack(inputs, dim=0)  # [B, T, D_low]
        targets = torch.stack(targets, dim=0)  # [B, T, D_high]
        time_steps = torch.stack(time_steps, dim=0) if time_steps[0] is not None else None  # [B, T]
        
        return inputs, targets, time_steps
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据加载器信息"""
        info = self.dataset_info.copy()
        info.update({
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory
        })
        return info


def create_multiscale_datasets(
    data_path: str,
    scale_factor: int = 2,
    pde_type: str = "darcy_flow",
    sequence_length: int = 1,
    original_resolution: Optional[List[int]] = None,
    normalize: bool = True,
    downsampling_method: str = "average",
    enable_center_crop: bool = False,
    center_crop_input_resolution: Optional[List[int]] = None,
    center_crop_output_resolution: Optional[List[int]] = None,
    train_transform: Optional[Any] = None,
    val_transform: Optional[Any] = None,
    test_transform: Optional[Any] = None
) -> Tuple[MultiScaleDataset, MultiScaleDataset, MultiScaleDataset]:
    """
    创建多尺度训练、验证、测试数据集
    
    Args:
        data_path: 数据文件路径
        scale_factor: 缩放因子
        pde_type: PDE类型
        sequence_length: 序列长度
        original_resolution: 原始分辨率
        normalize: 是否归一化
        downsampling_method: 下采样方法
        enable_center_crop: 是否启用输入输出的中心截取
        center_crop_input_resolution: 输入中心截取分辨率
        center_crop_output_resolution: 输出中心截取分辨率
        train_transform: 训练数据变换
        val_transform: 验证数据变换
        test_transform: 测试数据变换
    
    Returns:
        训练、验证、测试数据集的元组
    """
    train_dataset = MultiScaleDataset(
        data_path=data_path,
        scale_factor=scale_factor,
        pde_type=pde_type,
        split='train',
        sequence_length=sequence_length,
        original_resolution=original_resolution,
        normalize=normalize,
        downsampling_method=downsampling_method,
        enable_center_crop=enable_center_crop,
        center_crop_input_resolution=center_crop_input_resolution,
        center_crop_output_resolution=center_crop_output_resolution,
        transform=train_transform
    )
    
    val_dataset = MultiScaleDataset(
        data_path=data_path,
        scale_factor=scale_factor,
        pde_type=pde_type,
        split='val',
        sequence_length=sequence_length,
        original_resolution=original_resolution,
        normalize=normalize,
        downsampling_method=downsampling_method,
        enable_center_crop=enable_center_crop,
        center_crop_input_resolution=center_crop_input_resolution,
        center_crop_output_resolution=center_crop_output_resolution,
        transform=val_transform
    )
    
    test_dataset = MultiScaleDataset(
        data_path=data_path,
        scale_factor=scale_factor,
        pde_type=pde_type,
        split='test',
        sequence_length=sequence_length,
        original_resolution=original_resolution,
        normalize=normalize,
        downsampling_method=downsampling_method,
        enable_center_crop=enable_center_crop,
        center_crop_input_resolution=center_crop_input_resolution,
        center_crop_output_resolution=center_crop_output_resolution,
        transform=test_transform
    )
    
    return train_dataset, val_dataset, test_dataset


def create_multiscale_loaders(
    data_path: str,
    scale_factor: int = 2,
    pde_type: str = "darcy_flow",
    batch_size: int = 32,
    sequence_length: int = 1,
    original_resolution: Optional[List[int]] = None,
    normalize: bool = True,
    downsampling_method: str = "average",
    num_workers: int = 4,
    pin_memory: bool = True,
    enable_center_crop: bool = False,
    center_crop_input_resolution: Optional[List[int]] = None,
    center_crop_output_resolution: Optional[List[int]] = None
) -> Dict[str, Union[MultiScaleDataLoader, Dict]]:
    """创建多尺度数据加载器
    
    Args:
        data_path: 数据文件路径
        scale_factor: 缩放因子
        pde_type: PDE类型
        batch_size: 批次大小
        sequence_length: 序列长度
        original_resolution: 原始分辨率
        normalize: 是否归一化
        downsampling_method: 下采样方法
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        enable_center_crop: 是否启用输入输出的中心截取
        center_crop_input_resolution: 输入中心截取分辨率
        center_crop_output_resolution: 输出中心截取分辨率
    
    Returns:
        包含训练、验证、测试数据加载器和信息的字典
    """
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_multiscale_datasets(
        data_path=data_path,
        scale_factor=scale_factor,
        pde_type=pde_type,
        sequence_length=sequence_length,
        original_resolution=original_resolution,
        normalize=normalize,
        downsampling_method=downsampling_method,
        enable_center_crop=enable_center_crop,
        center_crop_input_resolution=center_crop_input_resolution,
        center_crop_output_resolution=center_crop_output_resolution
    )
    
    # 创建数据加载器
    train_loader = MultiScaleDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = MultiScaleDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = MultiScaleDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # 获取数据信息
    data_info = train_dataset.get_data_info()
    
    logging.info(f"多尺度数据加载器创建完成: {pde_type}, 缩放因子: {scale_factor}x")
    logging.info(f"输入维度: {data_info['input_dim']}, 输出维度: {data_info['output_dim']}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'info': data_info
    }


def analyze_multiscale_data(
    data_path: str,
    scale_factor: int = 2,
    pde_type: str = "darcy_flow",
    original_resolution: Optional[List[int]] = None
) -> Dict[str, Any]:
    """分析多尺度数据特性
    
    Args:
        data_path: 数据文件路径
        scale_factor: 缩放因子
        pde_type: PDE类型
        original_resolution: 原始分辨率
    
    Returns:
        数据分析结果
    """
    # 创建测试数据集
    dataset = MultiScaleDataset(
        data_path=data_path,
        scale_factor=scale_factor,
        pde_type=pde_type,
        split='train',
        sequence_length=1,
        original_resolution=original_resolution,
        normalize=False
    )
    
    # 获取基本信息
    info = dataset.get_data_info()
    
    # 分析样本数据
    sample_data = dataset.visualize_sample(0)
    
    # 计算数据统计量
    inputs, targets, _ = dataset[0]
    
    analysis = {
        'dataset_info': info,
        'sample_visualization': sample_data,
        'data_statistics': {
            'input_mean': float(torch.mean(inputs)),
            'input_std': float(torch.std(inputs)),
            'target_mean': float(torch.mean(targets)),
            'target_std': float(torch.std(targets)),
            'input_min': float(torch.min(inputs)),
            'input_max': float(torch.max(inputs)),
            'target_min': float(torch.min(targets)),
            'target_max': float(torch.max(targets))
        },
        'compression_ratio': info['output_dim'] / info['input_dim'],
        'memory_reduction': f"{(1 - info['input_dim'] / info['output_dim']) * 100:.1f}%"
    }
    
    return analysis