#!/usr/bin/env python3
"""
简单的Darcy Flow数据可视化脚本
专门用于生成可视化图片文件
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_darcy_sample(file_path, sample_idx=0, output_dir="."):
    """
    可视化指定样本的Darcy Flow数据
    """
    print(f"正在加载文件: {file_path}")
    print(f"样本索引: {sample_idx}")
    
    with h5py.File(file_path, 'r') as f:
        # 获取数据
        tensor_data = np.array(f['tensor'][sample_idx])  # shape: (1, 128, 128)
        nu_data = np.array(f['nu'][sample_idx])          # shape: (128, 128)
        
        print(f"Tensor数据形状: {tensor_data.shape}")
        print(f"Nu数据形状: {nu_data.shape}")
        print(f"Tensor范围: [{np.min(tensor_data):.6f}, {np.max(tensor_data):.6f}]")
        print(f"Nu范围: [{np.min(nu_data):.6f}, {np.max(nu_data):.6f}]")
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示tensor数据 (解的场)
        im1 = axes[0].imshow(tensor_data.squeeze(), cmap='viridis', origin='lower')
        axes[0].set_title(f'Solution Field (Sample {sample_idx})', fontsize=14)
        axes[0].set_xlabel('X', fontsize=12)
        axes[0].set_ylabel('Y', fontsize=12)
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Solution Value', fontsize=12)
        
        # 显示nu数据 (扩散系数)
        im2 = axes[1].imshow(nu_data, cmap='plasma', origin='lower')
        axes[1].set_title(f'Diffusion Coefficient (Sample {sample_idx})', fontsize=14)
        axes[1].set_xlabel('X', fontsize=12)
        axes[1].set_ylabel('Y', fontsize=12)
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Nu Value', fontsize=12)
        
        plt.tight_layout()
        
        # 保存图片
        output_path = Path(output_dir) / f"darcy_flow_sample_{sample_idx}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {output_path}")
        
        plt.close()
        
        # 创建单独的tensor可视化
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(tensor_data.squeeze(), cmap='viridis', origin='lower')
        ax.set_title(f'Darcy Flow Solution Field (Sample {sample_idx})', fontsize=16)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Solution Value', fontsize=14)
        
        output_path_tensor = Path(output_dir) / f"darcy_flow_tensor_sample_{sample_idx}.png"
        plt.savefig(output_path_tensor, dpi=300, bbox_inches='tight')
        print(f"Tensor可视化已保存到: {output_path_tensor}")
        
        plt.close()
        
        # 创建单独的nu可视化
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(nu_data, cmap='plasma', origin='lower')
        ax.set_title(f'Darcy Flow Diffusion Coefficient (Sample {sample_idx})', fontsize=16)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nu Value', fontsize=14)
        
        output_path_nu = Path(output_dir) / f"darcy_flow_nu_sample_{sample_idx}.png"
        plt.savefig(output_path_nu, dpi=300, bbox_inches='tight')
        print(f"Nu可视化已保存到: {output_path_nu}")
        
        plt.close()

def visualize_multiple_samples(file_path, num_samples=4, output_dir="."):
    """
    可视化多个样本
    """
    print(f"正在可视化 {num_samples} 个样本...")
    
    with h5py.File(file_path, 'r') as f:
        total_samples = f['tensor'].shape[0]
        print(f"数据集总共有 {total_samples} 个样本")
        
        # 选择要可视化的样本索引
        if num_samples > total_samples:
            num_samples = total_samples
        
        sample_indices = np.linspace(0, total_samples-1, num_samples, dtype=int)
        
        # 创建网格布局
        rows = 2
        cols = num_samples
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 8))
        
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i, sample_idx in enumerate(sample_indices):
            tensor_data = np.array(f['tensor'][sample_idx]).squeeze()
            nu_data = np.array(f['nu'][sample_idx])
            
            # 第一行显示tensor
            im1 = axes[0, i].imshow(tensor_data, cmap='viridis', origin='lower')
            axes[0, i].set_title(f'Solution {sample_idx}', fontsize=10)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # 第二行显示nu
            im2 = axes[1, i].imshow(nu_data, cmap='plasma', origin='lower')
            axes[1, i].set_title(f'Nu {sample_idx}', fontsize=10)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / f"darcy_flow_multiple_samples.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"多样本可视化已保存到: {output_path}")
        
        plt.close()

def main():
    # 默认文件路径
    file_path = "X:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codex/modify_multi_attention/PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5"
    
    if not Path(file_path).exists():
        print(f"错误: 文件不存在 {file_path}")
        return
    
    print("开始可视化Darcy Flow数据...")
    
    # 可视化第一个样本
    visualize_darcy_sample(file_path, sample_idx=0)
    
    # 可视化第100个样本
    visualize_darcy_sample(file_path, sample_idx=100)
    
    # 可视化多个样本的对比
    visualize_multiple_samples(file_path, num_samples=6)
    
    print("\n所有可视化完成！")

if __name__ == '__main__':
    main()