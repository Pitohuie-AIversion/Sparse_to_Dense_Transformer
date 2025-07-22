#!/usr/bin/env python3
"""统一数据适配器

支持多种数据格式：
1. 原始20x20→200x200压力场预测数据集
2. PDEBench标准数据集（Darcy Flow, Navier-Stokes等）
3. 自定义数据格式
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 导入现有的数据集类
from .dataset import PressureDataset
from .pdebench_adapter import PDEBenchDataset, PDEBenchDataLoader

# 导入新的压力场适配器
try:
    from .pressure_field_adapter import (
        PressureFieldDataset, 
        PressureFieldDataLoader,
        create_pressure_field_datasets,
        analyze_pressure_field_data
    )
except ImportError:
    PressureFieldDataset = None
    PressureFieldDataLoader = None
    create_pressure_field_datasets = None
    analyze_pressure_field_data = None


class UnifiedDataAdapter:
    """统一数据适配器
    
    自动检测数据格式并创建相应的数据加载器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化统一数据适配器
        
        Args:
            config: 配置字典，包含数据路径、批次大小等信息
        """
        self.config = config
        self.data_path = config.get('data', {}).get('path', '')
        self.batch_size = config.get('data', {}).get('batch_size', 32)
        self.normalize = config.get('data', {}).get('normalize', True)
        self.use_pdebench = config.get('data', {}).get('use_pdebench', False)
        
        # 检测数据类型
        self.data_type = self._detect_data_type()
        logging.info(f"检测到数据类型: {self.data_type}")
    
    def _detect_data_type(self) -> str:
        """检测数据类型"""
        if not self.data_path:
            raise ValueError("数据路径未指定")
        
        data_path = Path(self.data_path)
        
        # 检查文件是否存在
        if not data_path.exists():
            # 尝试相对路径
            relative_path = Path.cwd() / self.data_path
            if relative_path.exists():
                self.data_path = str(relative_path)
                data_path = relative_path
            else:
                raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        # 根据文件扩展名和内容检测类型
        if data_path.suffix == '.pt':
            # PyTorch文件，检查内容
            try:
                data = torch.load(data_path, map_location='cpu')
                if isinstance(data, dict) and 'in_pressure' in data and 'pressure' in data:
                    return 'pressure_field'  # 20x20→200x200压力场数据
                else:
                    return 'pytorch_tensor'  # 其他PyTorch张量数据
            except Exception as e:
                logging.warning(f"无法加载PyTorch文件: {e}")
                return 'unknown'
        
        elif data_path.suffix in ['.h5', '.hdf5']:
            # HDF5文件，可能是PDEBench格式
            return 'pdebench_hdf5'
        
        elif data_path.suffix in ['.npy', '.npz']:
            return 'numpy_array'
        
        else:
            return 'unknown'
    
    def create_datasets(self) -> Dict[str, Union[Dataset, DataLoader]]:
        """创建数据集"""
        if self.data_type == 'pressure_field':
            # 优先使用新的压力场适配器
            if create_pressure_field_datasets is not None:
                return self._create_pressure_field_datasets_advanced()
            else:
                return self._create_pressure_field_datasets()
        elif self.data_type == 'pdebench_hdf5':
            return self._create_pdebench_datasets()
        else:
            raise ValueError(f"不支持的数据类型: {self.data_type}")
    
    def _create_pressure_field_datasets(self) -> Dict[str, DataLoader]:
        """创建20x20→200x200压力场数据集"""
        logging.info("创建压力场数据集 (20x20→200x200)")
        
        # 创建完整数据集
        full_dataset = PressureDataset(self.data_path)
        
        # 数据分割
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        # 创建索引
        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 创建子数据集
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get('data', {}).get('num_workers', 4),
            pin_memory=self.config.get('data', {}).get('pin_memory', True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.get('data', {}).get('num_workers', 4),
            pin_memory=self.config.get('data', {}).get('pin_memory', True)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.get('data', {}).get('num_workers', 4),
            pin_memory=self.config.get('data', {}).get('pin_memory', True)
        )
        
        logging.info(f"压力场数据集创建完成: 训练{len(train_dataset)}, 验证{len(val_dataset)}, 测试{len(test_dataset)}")
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'info': {
                'data_type': 'pressure_field',
                'input_dim': 400,  # 20x20
                'output_dim': 40000,  # 200x200
                'input_shape': [20, 20],
                'output_shape': [200, 200],
                'total_samples': total_size,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset)
            }
        }
    
    def _create_pdebench_datasets(self) -> Dict[str, Union[PDEBenchDataLoader, Dict]]:
        """创建PDEBench数据集"""
        logging.info("创建PDEBench数据集")
        
        # 获取PDE配置
        current_pde = self.config.get('current_pde', 'darcy_flow')
        pde_config = self.config.get('pdebench', {}).get('pde_configs', {}).get(current_pde, {})
        
        if not pde_config:
            raise ValueError(f"未找到PDE配置: {current_pde}")
        
        # 创建数据集
        train_dataset = PDEBenchDataset(
            data_path=self.data_path,
            pde_type=current_pde,
            split='train',
            sequence_length=pde_config.get('sequence_length', 1),
            spatial_resolution=pde_config.get('spatial_resolution', [64, 64]),
            normalize=self.normalize
        )
        
        val_dataset = PDEBenchDataset(
            data_path=self.data_path,
            pde_type=current_pde,
            split='val',
            sequence_length=pde_config.get('sequence_length', 1),
            spatial_resolution=pde_config.get('spatial_resolution', [64, 64]),
            normalize=self.normalize
        )
        
        test_dataset = PDEBenchDataset(
            data_path=self.data_path,
            pde_type=current_pde,
            split='test',
            sequence_length=pde_config.get('sequence_length', 1),
            spatial_resolution=pde_config.get('spatial_resolution', [64, 64]),
            normalize=self.normalize
        )
        
        # 创建数据加载器
        train_loader = PDEBenchDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get('data', {}).get('num_workers', 4),
            pin_memory=self.config.get('data', {}).get('pin_memory', True)
        )
        
        val_loader = PDEBenchDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.get('data', {}).get('num_workers', 4),
            pin_memory=self.config.get('data', {}).get('pin_memory', True)
        )
        
        test_loader = PDEBenchDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.get('data', {}).get('num_workers', 4),
            pin_memory=self.config.get('data', {}).get('pin_memory', True)
        )
        
        # 获取数据信息
        data_info = train_dataset.get_data_info()
        
        logging.info(f"PDEBench数据集创建完成: {current_pde}")
        logging.info(f"空间分辨率: {data_info['spatial_resolution']}")
        logging.info(f"输入维度: {data_info['input_dim']}")
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'info': data_info
        }
    
    def _create_pressure_field_datasets_advanced(self) -> Dict[str, Union[DataLoader, Dict]]:
        """使用高级压力场适配器创建数据集"""
        logging.info("使用高级压力场适配器创建数据集")
        
        # 使用新的压力场适配器
        datasets = create_pressure_field_datasets(
            data_path=self.data_path,
            batch_size=self.batch_size,
            normalize=self.normalize,
            num_workers=self.config.get('data', {}).get('num_workers', 4),
            pin_memory=self.config.get('data', {}).get('pin_memory', True)
        )
        
        logging.info("高级压力场数据集创建完成")
        return datasets
    
    def analyze_data(self) -> Dict[str, Any]:
        """分析数据特性"""
        if self.data_type == 'pressure_field' and analyze_pressure_field_data is not None:
            return analyze_pressure_field_data(self.data_path)
        else:
            # 返回基本信息
            return {
                'data_type': self.data_type,
                'data_path': self.data_path,
                'batch_size': self.batch_size,
                'normalize': self.normalize
            }
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        datasets = self.create_datasets()
        return datasets.get('info', {})


def create_unified_datasets(config: Dict[str, Any]) -> Dict[str, Union[DataLoader, Dict]]:
    """创建统一数据集的便捷函数
    
    Args:
        config: 配置字典
    
    Returns:
        包含训练、验证、测试数据加载器和信息的字典
    """
    adapter = UnifiedDataAdapter(config)
    return adapter.create_datasets()


def analyze_unified_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """分析统一数据的便捷函数
    
    Args:
        config: 配置字典
    
    Returns:
        数据分析结果
    """
    adapter = UnifiedDataAdapter(config)
    return adapter.analyze_data()