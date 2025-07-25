#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建演示数据集用于多尺度训练
生成模拟的2D Darcy Flow数据
"""

import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_darcy_flow_sample(nx=128, ny=128):
    """
    生成一个模拟的2D Darcy Flow样本
    
    Args:
        nx, ny: 网格分辨率
    
    Returns:
        permeability: 渗透率场 [nx, ny]
        pressure: 压力场 [nx, ny]
    """
    # 生成随机渗透率场
    # 使用多个高斯核的叠加来创建复杂的渗透率分布
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    permeability = np.ones((nx, ny))
    
    # 添加多个高斯分布的渗透率异常
    num_anomalies = np.random.randint(3, 8)
    for _ in range(num_anomalies):
        # 随机中心位置
        cx = np.random.uniform(0.2, 0.8)
        cy = np.random.uniform(0.2, 0.8)
        
        # 随机大小和强度
        sigma_x = np.random.uniform(0.05, 0.15)
        sigma_y = np.random.uniform(0.05, 0.15)
        amplitude = np.random.uniform(0.1, 2.0)
        
        # 添加高斯异常
        anomaly = amplitude * np.exp(
            -((X - cx)**2 / (2 * sigma_x**2) + (Y - cy)**2 / (2 * sigma_y**2))
        )
        permeability += anomaly
    
    # 确保渗透率为正值
    permeability = np.maximum(permeability, 0.01)
    
    # 生成对应的压力场（简化的Darcy方程求解）
    # 这里使用简化的方法：压力与渗透率的倒数相关
    pressure = np.zeros_like(permeability)
    
    # 边界条件：左边界高压，右边界低压
    pressure[:, 0] = 1.0  # 左边界
    pressure[:, -1] = 0.0  # 右边界
    
    # 简化的迭代求解（Jacobi迭代）
    for _ in range(100):
        pressure_new = pressure.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # 简化的有限差分
                k_avg = (permeability[i, j] + permeability[i+1, j] + 
                        permeability[i-1, j] + permeability[i, j+1] + 
                        permeability[i, j-1]) / 5
                
                pressure_new[i, j] = 0.25 * (
                    pressure[i+1, j] + pressure[i-1, j] + 
                    pressure[i, j+1] + pressure[i, j-1]
                ) * k_avg
        
        pressure = pressure_new
    
    return permeability, pressure

def create_demo_dataset(output_path, num_samples=1000, resolution=(128, 128)):
    """
    创建演示数据集
    
    Args:
        output_path: 输出文件路径
        num_samples: 样本数量
        resolution: 数据分辨率
    """
    print(f"创建演示数据集: {output_path}")
    print(f"样本数量: {num_samples}")
    print(f"分辨率: {resolution[0]}×{resolution[1]}")
    
    nx, ny = resolution
    
    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成数据
    permeability_data = np.zeros((num_samples, 1, nx, ny), dtype=np.float32)
    pressure_data = np.zeros((num_samples, 1, nx, ny), dtype=np.float32)
    
    print("生成数据样本...")
    for i in tqdm(range(num_samples)):
        perm, press = generate_darcy_flow_sample(nx, ny)
        permeability_data[i, 0] = perm
        pressure_data[i, 0] = press
    
    # 保存为HDF5格式
    print(f"保存数据到: {output_path}")
    with h5py.File(output_path, 'w') as f:
        # 创建数据组
        f.create_dataset('tensor', data=permeability_data, compression='gzip')
        f.create_dataset('pressure', data=pressure_data, compression='gzip')
        
        # 添加元数据
        f.attrs['description'] = 'Demo 2D Darcy Flow Dataset'
        f.attrs['num_samples'] = num_samples
        f.attrs['resolution'] = resolution
        f.attrs['data_format'] = 'NCHW (samples, channels, height, width)'
        f.attrs['tensor_description'] = 'Permeability field'
        f.attrs['pressure_description'] = 'Pressure field'
        
        # 添加时间维度（模拟时间序列）
        time_steps = np.arange(num_samples, dtype=np.float32)
        f.create_dataset('t', data=time_steps)
    
    print("数据集创建完成！")
    
    # 可视化几个样本
    visualize_samples(output_path, num_vis=4)

def visualize_samples(dataset_path, num_vis=4):
    """
    可视化数据集样本
    """
    print("可视化数据样本...")
    
    with h5py.File(dataset_path, 'r') as f:
        permeability = f['tensor'][:num_vis, 0]  # [num_vis, nx, ny]
        pressure = f['pressure'][:num_vis, 0]    # [num_vis, nx, ny]
    
    fig, axes = plt.subplots(2, num_vis, figsize=(4*num_vis, 8))
    
    for i in range(num_vis):
        # 渗透率场
        im1 = axes[0, i].imshow(permeability[i], cmap='viridis', aspect='equal')
        axes[0, i].set_title(f'Permeability {i+1}')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i])
        
        # 压力场
        im2 = axes[1, i].imshow(pressure[i], cmap='plasma', aspect='equal')
        axes[1, i].set_title(f'Pressure {i+1}')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    
    # 保存可视化结果
    vis_path = dataset_path.parent / 'demo_dataset_visualization.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果保存到: {vis_path}")

def main():
    """
    主函数
    """
    # 配置
    output_path = "PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5"
    num_samples = 1000  # 创建1000个样本用于演示
    resolution = (128, 128)  # 128×128分辨率
    
    print("=" * 60)
    print("创建演示数据集")
    print("=" * 60)
    
    # 创建数据集
    create_demo_dataset(output_path, num_samples, resolution)
    
    print("\n✅ 演示数据集创建完成！")
    print(f"文件位置: {output_path}")
    print(f"数据格式: HDF5")
    print(f"样本数量: {num_samples}")
    print(f"分辨率: {resolution[0]}×{resolution[1]}")
    print("\n现在可以运行多尺度训练脚本了！")

if __name__ == "__main__":
    main()