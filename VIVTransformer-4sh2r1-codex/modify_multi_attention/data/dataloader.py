from typing import List, Optional, Tuple, Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose
import yaml
from pathlib import Path

from . import transforms
from .dataset import PressureDataset
from .pdebench_adapter import PDEBenchDataset, PDEBenchDataLoader
from .custom_dataset_adapter import CustomDataset, CustomDataLoader


class CustomSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        in_press, pressure, time_step = self.dataset[self.indices[idx]]
        if self.transform:
            in_press, pressure = self.transform((in_press, pressure))
        return in_press, pressure, time_step


def collate_fn(batch: List[Optional[Any]]) -> Optional[Any]:
    """Custom collate function to filter out None samples from a batch.

    Args:
        batch: A list of samples, where some might be None.

    Returns:
        A collated batch tensor or None if the batch is empty after filtering.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_loaders(
    data_path: str, batch_size: int, use_augmentation: bool = False, crop_size: Optional[Tuple[int, int]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates and returns data loaders for training, validation, and testing.

    Args:
        data_path: The path to the dataset file.
        batch_size: The number of samples per batch.
        use_augmentation: Whether to apply data augmentation to the training set.

    Returns:
        A tuple containing the training, validation, and test data loaders.
    """
    train_transform = None
    if use_augmentation:
        train_transform = Compose(
            [
                transforms.RandomFlip(),
                transforms.RandomRotate(),
                transforms.AddGaussianNoise(std=0.01),
                transforms.RandomCrop(crop_size if crop_size else (128, 128)),
            ]
        )

    full_dataset = PressureDataset(data_path)

    train_size = int(0.7 * len(full_dataset))
    valid_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - valid_size

    indices = torch.randperm(len(full_dataset)).tolist()

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    train_dataset = CustomSubset(full_dataset, train_indices, transform=train_transform)
    valid_dataset = CustomSubset(full_dataset, valid_indices, transform=None)
    test_dataset = CustomSubset(full_dataset, test_indices, transform=None)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, valid_loader, test_loader


def get_pdebench_loaders(
    config: Dict,
    pde_type: str = "ns_incom",
    batch_size: Optional[int] = None,
    use_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建PDEBench数据加载器
    
    Args:
        config: 配置字典，包含PDEBench相关设置
        pde_type: PDE类型
        batch_size: 批次大小，如果为None则从config中获取
        use_augmentation: 是否使用数据增强
        
    Returns:
        训练、验证、测试数据加载器的元组
    """
    # 获取PDEBench配置
    pdebench_config = config.get('pdebench', {})
    pde_configs = pdebench_config.get('pde_configs', {})
    
    if pde_type not in pde_configs:
        raise ValueError(f"不支持的PDE类型: {pde_type}")
    
    pde_config = pde_configs[pde_type]
    
    # 构建数据文件路径
    data_root = Path(pdebench_config.get('data_root', './data/pdebench'))
    data_file = data_root / pde_config['data_file']
    
    if not data_file.exists():
        raise FileNotFoundError(f"PDEBench数据文件不存在: {data_file}")
    
    # 获取批次大小
    if batch_size is None:
        batch_size = config.get('data', {}).get('batch_size', 32)
    
    # 数据变换
    train_transform = None
    if use_augmentation:
        train_transform = Compose([
            transforms.AddGaussianNoise(std=0.01),
            # PDEBench数据通常不需要空间变换
        ])
    
    # 创建数据集
    train_dataset = PDEBenchDataset(
        data_path=str(data_file),
        pde_type=pde_type,
        split='train',
        sequence_length=pde_config.get('sequence_length', 49),
        spatial_resolution=pde_config.get('spatial_resolution'),
        input_size=pde_config.get('input_size'),
        normalize=config.get('data', {}).get('normalize', True),
        transform=train_transform,
        use_adaptive_crop=pde_config.get('use_adaptive_crop', False),
        crop_multiplier=pde_config.get('crop_multiplier', 0.5)
    )
    
    valid_dataset = PDEBenchDataset(
        data_path=str(data_file),
        pde_type=pde_type,
        split='val',
        sequence_length=pde_config.get('sequence_length', 49),
        spatial_resolution=pde_config.get('spatial_resolution'),
        input_size=pde_config.get('input_size'),
        normalize=config.get('data', {}).get('normalize', True),
        transform=None,
        use_adaptive_crop=pde_config.get('use_adaptive_crop', False),
        crop_multiplier=pde_config.get('crop_multiplier', 0.5)
    )
    
    test_dataset = PDEBenchDataset(
        data_path=str(data_file),
        pde_type=pde_type,
        split='test',
        sequence_length=pde_config.get('sequence_length', 49),
        spatial_resolution=pde_config.get('spatial_resolution'),
        input_size=pde_config.get('input_size'),
        normalize=config.get('data', {}).get('normalize', True),
        transform=None,
        use_adaptive_crop=pde_config.get('use_adaptive_crop', False),
        crop_multiplier=pde_config.get('crop_multiplier', 0.5)
    )
    
    # 创建数据加载器
    num_workers = config.get('data', {}).get('num_workers', 4)
    pin_memory = config.get('data', {}).get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, test_loader


def get_custom_loaders(
    config: Dict,
    dataset_type: str = None,
    batch_size: Optional[int] = None,
    use_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建自定义数据集加载器
    
    Args:
        config: 配置字典，包含自定义数据集相关设置
        dataset_type: 数据集类型，如果为None则从config中获取
        batch_size: 批次大小，如果为None则从config中获取
        use_augmentation: 是否使用数据增强
        
    Returns:
        训练、验证、测试数据加载器的元组
    """
    # 获取自定义数据集配置
    data_config = config.get('data', {})
    custom_config = data_config.get('custom_dataset', {})
    
    # 获取数据集类型
    if dataset_type is None:
        dataset_type = custom_config.get('dataset_type', 'flow_field')
    
    # 获取批次大小
    if batch_size is None:
        batch_size = data_config.get('batch_size', 32)
    
    # 构建数据文件路径
    if 'file_paths' in custom_config:
        # 支持多个文件路径
        file_paths = custom_config['file_paths']
        if isinstance(file_paths, list):
            data_files = []
            for file_path in file_paths:
                if Path(file_path).is_absolute():
                    data_files.append(file_path)
                else:
                    data_files.append(str(Path(custom_config.get('data_root', '.')) / file_path))
        else:
            if Path(file_paths).is_absolute():
                data_files = [file_paths]
            else:
                data_files = [str(Path(custom_config.get('data_root', '.')) / file_paths)]
    else:
        # 兼容旧配置格式
        data_root = custom_config.get('data_root', '.')
        data_file = data_config.get('data_file', 'data.pt')
        # 如果data_root目录下有HDF5文件，优先使用
        data_root_path = Path(data_root)
        if data_root_path.exists():
            hdf5_files = list(data_root_path.glob('*.hdf5')) + list(data_root_path.glob('*.h5'))
            if hdf5_files:
                data_files = [str(hdf5_files[0])]
            else:
                data_files = [str(data_root_path / data_file)]
        else:
            data_files = [str(data_root_path / data_file)]
    
    # 使用第一个数据文件（后续可扩展支持多文件）
    data_path = data_files[0]
    
    # 准备CustomDataset的通用参数
    common_params = {
        'data_path': data_path,
        'dataset_type': dataset_type,
        'sequence_length': custom_config.get('sequence_length', 48),
        'spatial_resolution': custom_config.get('spatial_resolution', [128, 128]),
        'input_size': custom_config.get('input_size'),
        'normalize': data_config.get('normalize', True),
        'normalization_method': custom_config.get('normalization_method', 'standard'),
        'data_key': custom_config.get('data_keys', {}).get('input'),
        'target_key': custom_config.get('data_keys', {}).get('target'),
        'custom_split_ratios': custom_config.get('data_split_ratios', [0.7, 0.15, 0.15]),
        # Adaptive cropping parameters
        'use_adaptive_crop': custom_config.get('use_adaptive_crop', False),
        'crop_multiplier': custom_config.get('crop_multiplier', 0.1),
        'crop_type': custom_config.get('crop_type', 'default'),
        'crop_config': custom_config.get('crop_config')
    }
    
    # 创建数据集
    train_dataset = CustomDataset(
        split='train',
        **common_params
    )
    
    valid_dataset = CustomDataset(
        split='val',
        **common_params
    )
    
    test_dataset = CustomDataset(
        split='test',
        **common_params
    )
    
    # 创建数据加载器
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    
    train_loader = CustomDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    valid_loader = CustomDataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = CustomDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, valid_loader, test_loader


def get_adaptive_loaders(
    config: Dict,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """自适应数据加载器，根据配置自动选择数据集类型
    
    Args:
        config: 配置字典
        **kwargs: 额外参数
        
    Returns:
        训练、验证、测试数据加载器的元组
    """
    # 检查是否使用自定义数据集
    if config.get('data', {}).get('use_custom_dataset', False):
        dataset_type = config.get('custom_dataset', {}).get('dataset_type', 'flow_field')
        return get_custom_loaders(config, dataset_type=dataset_type, **kwargs)
    # 检查是否使用PDEBench数据集
    elif config.get('data', {}).get('use_pdebench', False):
        pde_type = config.get('pdebench', {}).get('current_pde', 'darcy_flow')
        return get_pdebench_loaders(config, pde_type=pde_type, **kwargs)
    else:
        # 使用原有的数据加载方式
        data_path = config.get('data', {}).get('path')
        batch_size = config.get('data', {}).get('batch_size', 128)
        use_augmentation = config.get('data', {}).get('use_augmentation', False)
        crop_size = tuple(config.get('data', {}).get('crop_size', [128, 128]))
        
        return get_loaders(data_path, batch_size, use_augmentation, crop_size)
