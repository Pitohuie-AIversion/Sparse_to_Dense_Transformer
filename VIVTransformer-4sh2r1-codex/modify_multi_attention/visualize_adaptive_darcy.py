#!/usr/bin/env python3
"""
可视化使用自适应裁剪处理的Darcy Flow数据

这个脚本展示:
1. 原始数据
2. 自适应裁剪后的数据
3. 输入输出对比
4. 不同裁剪参数的效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import h5py
import torch
from torch.utils.data import DataLoader

# 导入自定义模块
from data.custom_dataset_adapter import CustomDataset
from data.adaptive_transforms import AdaptiveCrop, create_darcy_flow_transform
from data.dataloader import get_adaptive_loaders

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_original_data(file_path, sample_idx=0):
    """
    加载原始HDF5数据
    """
    with h5py.File(file_path, 'r') as f:
        tensor_data = np.array(f['tensor'][sample_idx])  # (1, 128, 128)
        nu_data = np.array(f['nu'][sample_idx])  # (128, 128)
    
    return tensor_data, nu_data

def create_adaptive_dataset(data_path, crop_multiplier=0.5, crop_type='darcy_flow'):
    """
    创建使用自适应裁剪的数据集
    """
    dataset = CustomDataset(
        data_path=data_path,
        split='train',
        use_adaptive_crop=True,
        crop_multiplier=crop_multiplier,
        crop_type=crop_type,
        spatial_resolution=128,
        input_size=None,  # 不使用传统裁剪
        sequence_length=1,
        normalize_method='standard'
    )
    return dataset

def visualize_comparison(original_tensor, original_nu, cropped_tensor, cropped_nu, 
                        crop_multiplier, save_path=None):
    """
    可视化原始数据和裁剪后数据的对比
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 原始tensor数据
    im1 = axes[0, 0].imshow(original_tensor.squeeze(), cmap='viridis')
    axes[0, 0].set_title(f'原始 Tensor 数据 ({original_tensor.shape[-2]}x{original_tensor.shape[-1]})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 原始nu数据
    im2 = axes[0, 1].imshow(original_nu.squeeze(), cmap='plasma')
    axes[0, 1].set_title(f'原始 Nu 数据 ({original_nu.shape[-2]}x{original_nu.shape[-1]})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 裁剪后tensor数据
    im3 = axes[1, 0].imshow(cropped_tensor.squeeze(), cmap='viridis')
    axes[1, 0].set_title(f'自适应裁剪 Tensor (倍数={crop_multiplier}, {cropped_tensor.shape[-2]}x{cropped_tensor.shape[-1]})')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 裁剪后nu数据
    im4 = axes[1, 1].imshow(cropped_nu.squeeze(), cmap='plasma')
    axes[1, 1].set_title(f'自适应裁剪 Nu (倍数={crop_multiplier}, {cropped_nu.shape[-2]}x{cropped_nu.shape[-1]})')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n=== 数据统计对比 ===")
    print(f"原始 Tensor: 形状={original_tensor.shape}, 范围=[{np.min(original_tensor):.6f}, {np.max(original_tensor):.6f}]")
    print(f"原始 Nu: 形状={original_nu.shape}, 范围=[{np.min(original_nu):.6f}, {np.max(original_nu):.6f}]")
    print(f"裁剪 Tensor: 形状={cropped_tensor.shape}, 范围=[{np.min(cropped_tensor):.6f}, {np.max(cropped_tensor):.6f}]")
    print(f"裁剪 Nu: 形状={cropped_nu.shape}, 范围=[{np.min(cropped_nu):.6f}, {np.max(cropped_nu):.6f}]")

def visualize_multiple_crops(data_path, sample_idx=0, crop_multipliers=[0.3, 0.5, 0.7, 1.0]):
    """
    可视化不同裁剪倍数的效果
    """
    # 加载原始数据
    original_tensor, original_nu = load_original_data(data_path, sample_idx)
    
    fig, axes = plt.subplots(2, len(crop_multipliers), figsize=(20, 10))
    
    for i, crop_mult in enumerate(crop_multipliers):
        # 创建自适应裁剪变换
        transform = create_darcy_flow_transform(crop_multiplier=crop_mult)
        
        # 应用变换
        tensor_torch = torch.from_numpy(original_tensor).float()
        nu_torch = torch.from_numpy(original_nu).float()
        
        cropped_tensor = transform(tensor_torch).numpy()
        cropped_nu = transform(nu_torch).numpy()
        
        # 绘制tensor
        im1 = axes[0, i].imshow(cropped_tensor.squeeze(), cmap='viridis')
        axes[0, i].set_title(f'Tensor (倍数={crop_mult})\n{cropped_tensor.shape[-2]}x{cropped_tensor.shape[-1]}')
        axes[0, i].set_xlabel('X')
        axes[0, i].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, i])
        
        # 绘制nu
        im2 = axes[1, i].imshow(cropped_nu.squeeze(), cmap='plasma')
        axes[1, i].set_title(f'Nu (倍数={crop_mult})\n{cropped_nu.shape[-2]}x{cropped_nu.shape[-1]}')
        axes[1, i].set_xlabel('X')
        axes[1, i].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig('darcy_multiple_crops_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== 多种裁剪倍数效果对比 ===")
    for crop_mult in crop_multipliers:
        crop_size = int(128 * crop_mult)
        print(f"裁剪倍数 {crop_mult}: 输出尺寸 {crop_size}x{crop_size}")

def demonstrate_dataloader_integration(data_path):
    """
    演示与DataLoader的集成
    """
    print("\n=== DataLoader集成演示 ===")
    
    # 创建配置
    config = {
        'data': {
            'data_path': data_path,
            'loader_type': 'custom',
            'custom_dataset_params': {
                'use_adaptive_crop': True,
                'crop_multiplier': 0.6,
                'crop_type': 'darcy_flow',
                'spatial_resolution': 128,
                'sequence_length': 1,
                'normalize_method': 'standard'
            }
        },
        'training': {
            'batch_size': 4
        }
    }
    
    try:
        # 获取数据加载器
        train_loader, val_loader, test_loader = get_adaptive_loaders(config)
        
        # 获取一个批次的数据
        batch = next(iter(train_loader))
        inputs, targets = batch
        
        print(f"批次输入形状: {inputs.shape}")
        print(f"批次目标形状: {targets.shape}")
        
        # 可视化批次中的第一个样本
        sample_input = inputs[0].numpy()
        sample_target = targets[0].numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = axes[0].imshow(sample_input.squeeze(), cmap='viridis')
        axes[0].set_title(f'批次输入样本\n{sample_input.shape}')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(sample_target.squeeze(), cmap='plasma')
        axes[1].set_title(f'批次目标样本\n{sample_target.shape}')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('darcy_dataloader_batch.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"DataLoader集成演示失败: {e}")
        print("请确保数据路径正确且数据格式符合要求")

def main():
    # 数据文件路径
    data_path = 'PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5'
    
    if not Path(data_path).exists():
        print(f"错误: 数据文件不存在 {data_path}")
        return
    
    print("=== Darcy Flow 自适应裁剪可视化演示 ===")
    
    # 1. 基本对比可视化
    print("\n1. 基本对比可视化")
    sample_idx = 0
    crop_multiplier = 0.5
    
    # 加载原始数据
    original_tensor, original_nu = load_original_data(data_path, sample_idx)
    
    # 应用自适应裁剪
    transform = create_darcy_flow_transform(crop_multiplier=crop_multiplier)
    tensor_torch = torch.from_numpy(original_tensor).float()
    nu_torch = torch.from_numpy(original_nu).float()
    
    cropped_tensor = transform(tensor_torch).numpy()
    cropped_nu = transform(nu_torch).numpy()
    
    # 可视化对比
    visualize_comparison(original_tensor, original_nu, cropped_tensor, cropped_nu, 
                        crop_multiplier, 'darcy_basic_comparison.png')
    
    # 2. 多种裁剪倍数对比
    print("\n2. 多种裁剪倍数对比")
    visualize_multiple_crops(data_path, sample_idx)
    
    # 3. DataLoader集成演示
    print("\n3. DataLoader集成演示")
    demonstrate_dataloader_integration(data_path)
    
    print("\n=== 可视化完成 ===")
    print("生成的图像文件:")
    print("- darcy_basic_comparison.png: 基本对比")
    print("- darcy_multiple_crops_comparison.png: 多种裁剪倍数对比")
    print("- darcy_dataloader_batch.png: DataLoader批次示例")

if __name__ == '__main__':
    main()