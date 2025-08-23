"""PDEBench数据集适配器

本模块提供PDEBench数据集的适配功能，支持多种PDE类型的数据加载和预处理。
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class PDEBenchDataset(Dataset):
    """PDEBench数据集类
    
    支持加载和预处理PDEBench格式的数据，包括：
    - Navier-Stokes方程
    - Darcy流方程
    - 浅水方程
    - 对流方程
    - Burgers方程
    - 反应扩散方程
    """
    
    def __init__(
        self,
        data_path: str,
        pde_type: str = "ns_incom",
        split: str = "train",
        sequence_length: int = 49,
        spatial_resolution: Optional[List[int]] = None,
        normalize: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None
    ):
        """
        初始化PDEBench数据集
        
        Args:
            data_path: 数据文件路径（.h5或.pt格式）
            pde_type: PDE类型
            split: 数据分割类型 ('train', 'val', 'test')
            sequence_length: 序列长度
            spatial_resolution: 空间分辨率 [H, W]
            normalize: 是否归一化数据
            transform: 输入数据变换
            target_transform: 目标数据变换
        """
        self.data_path = Path(data_path)
        self.pde_type = pde_type
        self.split = split
        self.sequence_length = sequence_length
        self.spatial_resolution = spatial_resolution or [64, 64]
        self.normalize = normalize
        self.transform = transform
        self.target_transform = target_transform
        
        # 验证文件存在
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        # 加载数据
        self._load_data()
        
        # 数据预处理
        self._preprocess_data()
        
        # 创建数据分割
        self._create_splits()
        
        logging.info(f"PDEBench数据集初始化完成: {self.pde_type}, {self.split}, 样本数: {len(self)}")
    
    def _load_data(self):
        """加载数据文件"""
        if self.data_path.suffix in ['.h5', '.hdf5']:
            self._load_h5_data()
        elif self.data_path.suffix == '.pt':
            self._load_pt_data()
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path.suffix}")
    
    def _load_h5_data(self):
        """加载HDF5格式数据"""
        with h5py.File(self.data_path, 'r') as f:
            # PDEBench数据通常存储在'data'或'tensor'键下
            if 'data' in f:
                self.raw_data = f['data'][:]
            elif 'tensor' in f:
                self.raw_data = f['tensor'][:]
            else:
                # 尝试其他可能的键名
                keys = list(f.keys())
                if len(keys) == 1:
                    self.raw_data = f[keys[0]][:]
                else:
                    # 对于PDEBench格式，优先选择数值数据键
                    data_keys = [k for k in keys if not k.endswith('-coordinate') and k != 'nu']
                    if data_keys:
                        self.raw_data = f[data_keys[0]][:]
                    else:
                        raise KeyError(f"无法确定数据键，可用键: {keys}")
            
            # 加载元数据
            self.metadata = dict(f.attrs) if hasattr(f, 'attrs') else {}
            
            # 加载坐标信息（如果存在）
            if 'x-coordinate' in f:
                self.metadata['x_coords'] = f['x-coordinate'][:]
            if 'y-coordinate' in f:
                self.metadata['y_coords'] = f['y-coordinate'][:]
            if 't-coordinate' in f:
                self.metadata['t_coords'] = f['t-coordinate'][:]
    
    def _load_pt_data(self):
        """加载PyTorch格式数据"""
        data = torch.load(self.data_path, map_location='cpu')
        if isinstance(data, dict):
            self.raw_data = data.get('data', data)
            self.metadata = data.get('metadata', {})
        else:
            self.raw_data = data
            self.metadata = {}
    
    def _preprocess_data(self):
        """数据预处理"""
        # 转换为numpy数组
        if isinstance(self.raw_data, torch.Tensor):
            self.raw_data = self.raw_data.numpy()
        
        # 检查数据形状
        if len(self.raw_data.shape) < 4:
            raise ValueError(f"数据维度不足，期望至少4维，实际: {self.raw_data.shape}")
        
        # 数据形状: [N, T, H, W] 或 [N, T, H, W, C]
        self.n_samples = self.raw_data.shape[0]
        self.n_timesteps = self.raw_data.shape[1]
        
        # 处理空间维度
        if len(self.raw_data.shape) == 4:
            # [N, T, H, W] -> [N, T, H, W, 1]
            self.raw_data = self.raw_data[..., np.newaxis]
        
        self.height, self.width, self.n_channels = self.raw_data.shape[2:5]
        
        # 更新空间分辨率
        if self.spatial_resolution != [self.height, self.width]:
            logging.warning(
                f"指定的空间分辨率 {self.spatial_resolution} 与数据实际分辨率 "
                f"[{self.height}, {self.width}] 不匹配，使用数据实际分辨率"
            )
            self.spatial_resolution = [self.height, self.width]
    
    def _create_splits(self):
        """创建数据分割"""
        # 标准分割比例: 70% train, 15% val, 15% test
        n_train = int(0.7 * self.n_samples)
        n_val = int(0.15 * self.n_samples)
        n_test = self.n_samples - n_train - n_val
        
        if self.split == 'train':
            self.indices = list(range(0, n_train))
        elif self.split == 'val':
            self.indices = list(range(n_train, n_train + n_val))
        elif self.split == 'test':
            self.indices = list(range(n_train + n_val, self.n_samples))
        else:
            raise ValueError(f"不支持的数据分割类型: {self.split}")
        
        # 如果归一化，计算统计量（仅使用训练集）
        if self.normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """计算归一化统计量"""
        if self.split == 'train':
            # 使用训练数据计算统计量
            train_data = self.raw_data[self.indices]
            self.data_mean = np.mean(train_data)
            self.data_std = np.std(train_data)
        else:
            # 对于验证和测试集，使用预设的统计量或计算全局统计量
            self.data_mean = np.mean(self.raw_data)
            self.data_std = np.std(self.raw_data)
        
        # 避免除零
        if self.data_std == 0:
            self.data_std = 1.0
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """归一化数据"""
        if self.normalize:
            return (data - self.data_mean) / self.data_std
        return data
    
    def _prepare_sequence(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """准备序列数据"""
        # 获取完整的时间序列
        full_sequence = self.raw_data[sample_idx]  # [T, H, W, C]
        
        # 根据序列长度截取或填充
        if self.n_timesteps >= self.sequence_length:
            # 随机选择起始点（训练时）或固定起始点（验证/测试时）
            if self.split == 'train' and self.n_timesteps > self.sequence_length:
                start_idx = np.random.randint(0, self.n_timesteps - self.sequence_length + 1)
            else:
                start_idx = 0
            
            sequence = full_sequence[start_idx:start_idx + self.sequence_length]
        else:
            # 如果时间步不足，进行填充
            sequence = np.pad(
                full_sequence,
                ((0, self.sequence_length - self.n_timesteps), (0, 0), (0, 0), (0, 0)),
                mode='edge'
            )
        
        # 分离输入和目标
        # 输入: 前n-1个时间步，目标: 后n-1个时间步
        inputs = sequence[:-1]  # [T-1, H, W, C]
        targets = sequence[1:]  # [T-1, H, W, C]
        
        return inputs, targets
    
    def _reshape_spatial_data(self, data: np.ndarray) -> np.ndarray:
        """重塑空间数据"""
        # 输入形状: [T, H, W, C]
        # 输出形状: [T, H*W*C]
        T = data.shape[0]
        spatial_dim = self.height * self.width * self.n_channels
        return data.reshape(T, spatial_dim)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """获取数据项"""
        # 获取实际的样本索引
        sample_idx = self.indices[idx]
        
        # 准备序列数据
        inputs, targets = self._prepare_sequence(sample_idx)
        
        # 归一化
        inputs = self._normalize_data(inputs)
        targets = self._normalize_data(targets)
        
        # 重塑空间维度
        inputs = self._reshape_spatial_data(inputs)  # [T-1, H*W*C]
        targets = self._reshape_spatial_data(targets)  # [T-1, H*W*C]
        
        # 转换为张量
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()
        
        # 应用变换
        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            targets = self.target_transform(targets)
        
        # 时间步信息（可选）
        time_steps = torch.arange(inputs.shape[0]).float()
        
        return inputs, targets, time_steps
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            'pde_type': self.pde_type,
            'split': self.split,
            'n_samples': len(self),
            'sequence_length': self.sequence_length,
            'spatial_resolution': self.spatial_resolution,
            'n_channels': self.n_channels,
            'input_dim': self.height * self.width * self.n_channels,
            'normalize': self.normalize,
            'data_mean': getattr(self, 'data_mean', None),
            'data_std': getattr(self, 'data_std', None),
            'metadata': self.metadata
        }


class PDEBenchDataLoader(DataLoader):
    """PDEBench专用数据加载器"""
    
    def __init__(
        self,
        dataset: PDEBenchDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ):
        """
        初始化PDEBench数据加载器
        
        Args:
            dataset: PDEBench数据集
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            pin_memory: 是否固定内存
            **kwargs: 其他DataLoader参数
        """
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
        inputs = torch.stack(inputs, dim=0)  # [B, T, D]
        targets = torch.stack(targets, dim=0)  # [B, T, D]
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


def create_pdebench_datasets(
    data_path: str,
    pde_type: str = "ns_incom",
    sequence_length: int = 49,
    spatial_resolution: Optional[List[int]] = None,
    normalize: bool = True,
    train_transform: Optional[Any] = None,
    val_transform: Optional[Any] = None,
    test_transform: Optional[Any] = None
) -> Tuple[PDEBenchDataset, PDEBenchDataset, PDEBenchDataset]:
    """创建PDEBench训练、验证、测试数据集
    
    Args:
        data_path: 数据文件路径
        pde_type: PDE类型
        sequence_length: 序列长度
        spatial_resolution: 空间分辨率
        normalize: 是否归一化
        train_transform: 训练数据变换
        val_transform: 验证数据变换
        test_transform: 测试数据变换
    
    Returns:
        训练、验证、测试数据集的元组
    """
    train_dataset = PDEBenchDataset(
        data_path=data_path,
        pde_type=pde_type,
        split='train',
        sequence_length=sequence_length,
        spatial_resolution=spatial_resolution,
        normalize=normalize,
        transform=train_transform
    )
    
    val_dataset = PDEBenchDataset(
        data_path=data_path,
        pde_type=pde_type,
        split='val',
        sequence_length=sequence_length,
        spatial_resolution=spatial_resolution,
        normalize=normalize,
        transform=val_transform
    )
    
    test_dataset = PDEBenchDataset(
        data_path=data_path,
        pde_type=pde_type,
        split='test',
        sequence_length=sequence_length,
        spatial_resolution=spatial_resolution,
        normalize=normalize,
        transform=test_transform
    )
    
    return train_dataset, val_dataset, test_dataset


def create_pdebench_loaders(
    data_path: str,
    pde_type: str = "ns_incom",
    batch_size: int = 32,
    sequence_length: int = 49,
    spatial_resolution: Optional[List[int]] = None,
    normalize: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[Any] = None,
    val_transform: Optional[Any] = None,
    test_transform: Optional[Any] = None
) -> Tuple[PDEBenchDataLoader, PDEBenchDataLoader, PDEBenchDataLoader]:
    """创建PDEBench数据加载器
    
    Args:
        data_path: 数据文件路径
        pde_type: PDE类型
        batch_size: 批次大小
        sequence_length: 序列长度
        spatial_resolution: 空间分辨率
        normalize: 是否归一化
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        train_transform: 训练数据变换
        val_transform: 验证数据变换
        test_transform: 测试数据变换
    
    Returns:
        训练、验证、测试数据加载器的元组
    """
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_pdebench_datasets(
        data_path=data_path,
        pde_type=pde_type,
        sequence_length=sequence_length,
        spatial_resolution=spatial_resolution,
        normalize=normalize,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = PDEBenchDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=2,
    )
    
    val_loader = PDEBenchDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=2,
    )
    
    test_loader = PDEBenchDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=2,
    )
    
    return train_loader, val_loader, test_loader