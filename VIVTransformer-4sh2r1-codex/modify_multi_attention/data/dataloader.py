from typing import List, Optional, Tuple, Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import Compose
import yaml
from pathlib import Path

from . import transforms
from .dataset import PressureDataset
from .pdebench_adapter import PDEBenchDataset, PDEBenchDataLoader
from .reynolds_dataset import ReynoldsFlowDataset, create_reynolds_loaders


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

    # Get optimized settings
    num_workers = 6
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    
    # 优化的DataLoader配置
    extra_loader_kwargs = {
        'prefetch_factor': 2,
        'drop_last': True,  # 关键：固定batch_size，避免最后一个小批次
    }

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, collate_fn=collate_fn,
        **extra_loader_kwargs
    )
    
    # 验证和测试集保持drop_last=False以使用全部数据
    eval_loader_kwargs = {
        'prefetch_factor': 2,
        'drop_last': False,
    }
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, collate_fn=collate_fn,
        **eval_loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, collate_fn=collate_fn,
        **eval_loader_kwargs
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
        normalize=config.get('data', {}).get('normalize', True),
        transform=train_transform
    )
    
    valid_dataset = PDEBenchDataset(
        data_path=str(data_file),
        pde_type=pde_type,
        split='val',
        sequence_length=pde_config.get('sequence_length', 49),
        spatial_resolution=pde_config.get('spatial_resolution'),
        normalize=config.get('data', {}).get('normalize', True),
        transform=None
    )
    
    test_dataset = PDEBenchDataset(
        data_path=str(data_file),
        pde_type=pde_type,
        split='test',
        sequence_length=pde_config.get('sequence_length', 49),
        spatial_resolution=pde_config.get('spatial_resolution'),
        normalize=config.get('data', {}).get('normalize', True),
        transform=None
    )
    
    # 创建数据加载器
    num_workers = config.get('data', {}).get('num_workers', 4)
    pin_memory = config.get('data', {}).get('pin_memory', True)
    
    # Performance optimizations from config
    data_loading_cfg = config.get('performance', {}).get('data_loading', {})
    prefetch_factor = data_loading_cfg.get('prefetch_factor', 2)
    drop_last = data_loading_cfg.get('drop_last', False)
    
    persistent_workers = num_workers > 0
    extra_loader_kwargs = {}
    if prefetch_factor and num_workers > 0:
        extra_loader_kwargs['prefetch_factor'] = prefetch_factor
    if 'drop_last' in data_loading_cfg:
        extra_loader_kwargs['drop_last'] = drop_last
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        **extra_loader_kwargs
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        **extra_loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
        **extra_loader_kwargs
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
    # 检查是否使用PDEBench数据集
    if config.get('data', {}).get('use_pdebench', False):
        pde_type = config.get('current_pde', 'ns_incom')
        return get_pdebench_loaders(config, pde_type=pde_type, **kwargs)
    
    # 检查是否使用Reynolds 4D数据集
    reynolds_cfg = config.get('reynolds_data', {})
    if reynolds_cfg.get('enabled', False):
        data_path = reynolds_cfg.get('path') or config.get('data', {}).get('path')
        batch_size = config.get('data', {}).get('batch_size', 128)
        use_time_sequence = reynolds_cfg.get('use_time_sequence', False)
        input_size = reynolds_cfg.get('input_size', 20)
        output_size = reynolds_cfg.get('output_size', 200)
        sequence_length = reynolds_cfg.get('sequence_length', 5)
        
        # Forward loader-related performance options from config
        num_workers = config.get('data', {}).get('num_workers', 0)
        pin_memory = config.get('data', {}).get('pin_memory', torch.cuda.is_available())
        prefetch_factor = config.get('performance', {}).get('data_loading', {}).get('prefetch_factor', 2)
        
        train_loader, valid_loader, test_loader, _ = create_reynolds_loaders(
            data_path=data_path,
            batch_size=batch_size,
            use_time_sequence=use_time_sequence,
            input_size=input_size,
            output_size=output_size,
            sequence_length=sequence_length,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
        return train_loader, valid_loader, test_loader
    
    # 使用原有的数据加载方式
    data_path = config.get('data', {}).get('path')
    batch_size = config.get('data', {}).get('batch_size', 128)
    use_augmentation = config.get('data', {}).get('use_augmentation', False)
    crop_size = tuple(config.get('data', {}).get('crop_size', [128, 128]))
    
    return get_loaders(data_path, batch_size, use_augmentation, crop_size)
