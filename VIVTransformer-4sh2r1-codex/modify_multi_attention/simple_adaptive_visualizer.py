#!/usr/bin/env python3
"""
简化的Darcy Flow自适应裁剪可视化脚本

专门用于展示自适应裁剪处理后的数据输入输出效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import h5py
import torch
from pathlib import Path

# 导入自定义模块
from data.adaptive_transforms import create_darcy_flow_transform

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_darcy_sample(file_path, sample_idx=0):
    """
    加载Darcy Flow数据样本
    """
    print(f"正在加载样本 {sample_idx} 从文件: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # 获取数据信息
        tensor_shape = f['tensor'].shape
        nu_shape = f['nu'].shape
        
        print(f"数据集信息:")
        print(f"  Tensor形状: {tensor_shape}")
        print(f"  Nu形状: {nu_shape}")
        
        # 加载指定样本
        tensor_data = np.array(f['tensor'][sample_idx])  # (1, 128, 128)
        nu_data = np.array(f['nu'][sample_idx])  # (128, 128)
        
        print(f"\n样本 {sample_idx} 数据:")
        print(f"  Tensor: {tensor_data.shape}, 范围=[{np.min(tensor_data):.6f}, {np.max(tensor_data):.6f}]")
        print(f"  Nu: {nu_data.shape}, 范围=[{np.min(nu_data):.6f}, {np.max(nu_data):.6f}]")
    
    return tensor_data, nu_data

def apply_adaptive_crop(data, crop_multiplier=0.5):
    """
    应用自适应裁剪变换
    """
    print(f"\n应用自适应裁剪，倍数: {crop_multiplier}")
    
    # 创建变换
    transform = create_darcy_flow_transform(crop_multiplier=crop_multiplier)
    
    # 转换为tensor并应用变换
    data_tensor = torch.from_numpy(data).float()
    cropped_tensor = transform(data_tensor)
    cropped_data = cropped_tensor.numpy()
    
    print(f"  原始形状: {data.shape}")
    print(f"  裁剪后形状: {cropped_data.shape}")
    print(f"  裁剪后范围: [{np.min(cropped_data):.6f}, {np.max(cropped_data):.6f}]")
    
    return cropped_data

def create_comparison_visualization(original_tensor, original_nu, 
                                 cropped_tensor, cropped_nu, 
                                 crop_multiplier, save_path):
    """
    创建对比可视化图像
    """
    print(f"\n创建对比可视化图像: {save_path}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 原始tensor数据
    im1 = axes[0, 0].imshow(original_tensor.squeeze(), cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'Original Tensor Data\n{original_tensor.shape[-2]}x{original_tensor.shape[-1]}', fontsize=12)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # 原始nu数据
    im2 = axes[0, 1].imshow(original_nu.squeeze(), cmap='plasma', aspect='auto')
    axes[0, 1].set_title(f'Original Nu Data\n{original_nu.shape[-2]}x{original_nu.shape[-1]}', fontsize=12)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # 裁剪后tensor数据
    im3 = axes[1, 0].imshow(cropped_tensor.squeeze(), cmap='viridis', aspect='auto')
    axes[1, 0].set_title(f'Adaptive Cropped Tensor\nMultiplier={crop_multiplier}, {cropped_tensor.shape[-2]}x{cropped_tensor.shape[-1]}', fontsize=12)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # 裁剪后nu数据
    im4 = axes[1, 1].imshow(cropped_nu.squeeze(), cmap='plasma', aspect='auto')
    axes[1, 1].set_title(f'Adaptive Cropped Nu\nMultiplier={crop_multiplier}, {cropped_nu.shape[-2]}x{cropped_nu.shape[-1]}', fontsize=12)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像以释放内存
    
    print(f"对比图像已保存到: {save_path}")

def create_multiple_crops_visualization(original_tensor, original_nu, 
                                      crop_multipliers, save_path):
    """
    创建多种裁剪倍数的对比可视化
    """
    print(f"\n创建多种裁剪倍数对比图像: {save_path}")
    
    fig, axes = plt.subplots(2, len(crop_multipliers), figsize=(20, 10))
    
    for i, crop_mult in enumerate(crop_multipliers):
        # 应用裁剪
        cropped_tensor = apply_adaptive_crop(original_tensor, crop_mult)
        cropped_nu = apply_adaptive_crop(original_nu, crop_mult)
        
        # 绘制tensor
        im1 = axes[0, i].imshow(cropped_tensor.squeeze(), cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Tensor\nMult={crop_mult}\n{cropped_tensor.shape[-2]}x{cropped_tensor.shape[-1]}', fontsize=10)
        axes[0, i].set_xlabel('X')
        if i == 0:
            axes[0, i].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
        
        # 绘制nu
        im2 = axes[1, i].imshow(cropped_nu.squeeze(), cmap='plasma', aspect='auto')
        axes[1, i].set_title(f'Nu\nMult={crop_mult}\n{cropped_nu.shape[-2]}x{cropped_nu.shape[-1]}', fontsize=10)
        axes[1, i].set_xlabel('X')
        if i == 0:
            axes[1, i].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1, i], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"多种裁剪倍数对比图像已保存到: {save_path}")

def print_data_summary(original_tensor, original_nu, cropped_tensor, cropped_nu, crop_multiplier):
    """
    打印数据处理摘要
    """
    print("\n" + "="*60)
    print("           DARCY FLOW 自适应裁剪处理摘要")
    print("="*60)
    
    print(f"\n📊 原始数据:")
    print(f"   Tensor: {original_tensor.shape} | 范围: [{np.min(original_tensor):.6f}, {np.max(original_tensor):.6f}]")
    print(f"   Nu:     {original_nu.shape} | 范围: [{np.min(original_nu):.6f}, {np.max(original_nu):.6f}]")
    
    print(f"\n🔄 自适应裁剪 (倍数={crop_multiplier}):")
    print(f"   Tensor: {cropped_tensor.shape} | 范围: [{np.min(cropped_tensor):.6f}, {np.max(cropped_tensor):.6f}]")
    print(f"   Nu:     {cropped_nu.shape} | 范围: [{np.min(cropped_nu):.6f}, {np.max(cropped_nu):.6f}]")
    
    # 计算数据压缩比
    original_size = np.prod(original_tensor.shape)
    cropped_size = np.prod(cropped_tensor.shape)
    compression_ratio = cropped_size / original_size
    
    print(f"\n📈 处理效果:")
    print(f"   数据压缩比: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
    print(f"   尺寸变化: {original_tensor.shape[-2]}x{original_tensor.shape[-1]} → {cropped_tensor.shape[-2]}x{cropped_tensor.shape[-1]}")
    
    print("\n" + "="*60)

def main():
    # 数据文件路径
    data_path = 'PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5'
    
    if not Path(data_path).exists():
        print(f"❌ 错误: 数据文件不存在 {data_path}")
        return
    
    print("🚀 开始Darcy Flow自适应裁剪可视化演示")
    print("="*60)
    
    # 1. 加载原始数据
    sample_idx = 0
    original_tensor, original_nu = load_darcy_sample(data_path, sample_idx)
    
    # 2. 应用自适应裁剪
    crop_multiplier = 0.5
    cropped_tensor = apply_adaptive_crop(original_tensor, crop_multiplier)
    cropped_nu = apply_adaptive_crop(original_nu, crop_multiplier)
    
    # 3. 创建基本对比可视化
    create_comparison_visualization(
        original_tensor, original_nu, 
        cropped_tensor, cropped_nu, 
        crop_multiplier, 
        'darcy_adaptive_comparison.png'
    )
    
    # 4. 创建多种裁剪倍数对比
    crop_multipliers = [0.3, 0.5, 0.7, 1.0]
    create_multiple_crops_visualization(
        original_tensor, original_nu, 
        crop_multipliers, 
        'darcy_multiple_crops.png'
    )
    
    # 5. 打印数据处理摘要
    print_data_summary(original_tensor, original_nu, cropped_tensor, cropped_nu, crop_multiplier)
    
    print("\n✅ 可视化完成!")
    print("\n📁 生成的文件:")
    print("   - darcy_adaptive_comparison.png: 基本对比图")
    print("   - darcy_multiple_crops.png: 多种裁剪倍数对比图")
    
    print("\n💡 说明:")
    print("   - 自适应裁剪可以在训练过程中动态应用")
    print("   - 无需预处理数据文件，直接集成到DataLoader中")
    print("   - 支持多种裁剪策略和参数配置")
    print("   - 保持数据的空间特征和物理意义")

if __name__ == '__main__':
    main()