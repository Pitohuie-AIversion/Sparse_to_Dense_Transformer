#!/usr/bin/env python3
"""
分析2D Darcy Flow数据集内容
"""

import h5py
import numpy as np

def analyze_darcy_dataset(file_path):
    """
    分析Darcy Flow数据集的详细内容
    """
    with h5py.File(file_path, 'r') as f:
        print("=== 2D Darcy Flow 数据集分析 ===")
        print(f"文件路径: {file_path}")
        print(f"数据集键: {list(f.keys())}")
        
        # 基本信息
        tensor_shape = f['tensor'].shape
        nu_shape = f['nu'].shape
        x_coord_shape = f['x-coordinate'].shape
        y_coord_shape = f['y-coordinate'].shape
        
        print(f"\n=== 数据维度信息 ===")
        print(f"样本数量: {tensor_shape[0]}")
        print(f"解场(tensor)形状: {tensor_shape}")
        print(f"扩散系数(nu)形状: {nu_shape}")
        print(f"x坐标形状: {x_coord_shape}")
        print(f"y坐标形状: {y_coord_shape}")
        
        # 坐标信息
        x_coords = f['x-coordinate'][:]
        y_coords = f['y-coordinate'][:]
        print(f"\n=== 空间坐标信息 ===")
        print(f"x坐标范围: [{x_coords.min():.6f}, {x_coords.max():.6f}]")
        print(f"y坐标范围: [{y_coords.min():.6f}, {y_coords.max():.6f}]")
        print(f"网格分辨率: {len(x_coords)} x {len(y_coords)}")
        
        # 分析前10个样本
        print(f"\n=== 样本统计分析 (前10个样本) ===")
        print("样本ID | 解场范围 | 解场均值 | 解场标准差 | 扩散系数范围 | 扩散系数均值")
        print("-" * 80)
        
        for i in range(min(10, tensor_shape[0])):
            data = f['tensor'][i]
            nu = f['nu'][i]
            
            print(f"{i:6d} | [{data.min():.6f}, {data.max():.6f}] | {data.mean():.6f} | {data.std():.6f} | [{nu.min():.6f}, {nu.max():.6f}] | {nu.mean():.6f}")
        
        # 整体统计
        print(f"\n=== 整体数据集统计 ===")
        
        # 随机采样100个样本进行统计
        sample_indices = np.random.choice(tensor_shape[0], min(100, tensor_shape[0]), replace=False)
        
        all_data_stats = []
        all_nu_stats = []
        
        for idx in sample_indices:
            data = f['tensor'][idx]
            nu = f['nu'][idx]
            all_data_stats.append([data.min(), data.max(), data.mean(), data.std()])
            all_nu_stats.append([nu.min(), nu.max(), nu.mean(), nu.std()])
        
        all_data_stats = np.array(all_data_stats)
        all_nu_stats = np.array(all_nu_stats)
        
        print(f"解场统计 (基于{len(sample_indices)}个样本):")
        print(f"  最小值范围: [{all_data_stats[:, 0].min():.6f}, {all_data_stats[:, 0].max():.6f}]")
        print(f"  最大值范围: [{all_data_stats[:, 1].min():.6f}, {all_data_stats[:, 1].max():.6f}]")
        print(f"  均值范围: [{all_data_stats[:, 2].min():.6f}, {all_data_stats[:, 2].max():.6f}]")
        print(f"  标准差范围: [{all_data_stats[:, 3].min():.6f}, {all_data_stats[:, 3].max():.6f}]")
        
        print(f"\n扩散系数统计 (基于{len(sample_indices)}个样本):")
        print(f"  最小值范围: [{all_nu_stats[:, 0].min():.6f}, {all_nu_stats[:, 0].max():.6f}]")
        print(f"  最大值范围: [{all_nu_stats[:, 1].min():.6f}, {all_nu_stats[:, 1].max():.6f}]")
        print(f"  均值范围: [{all_nu_stats[:, 2].min():.6f}, {all_nu_stats[:, 2].max():.6f}]")
        print(f"  标准差范围: [{all_nu_stats[:, 3].min():.6f}, {all_nu_stats[:, 3].max():.6f}]")
        
        # 数据类型和存储信息
        print(f"\n=== 数据类型和存储信息 ===")
        print(f"解场数据类型: {f['tensor'].dtype}")
        print(f"扩散系数数据类型: {f['nu'].dtype}")
        print(f"坐标数据类型: {f['x-coordinate'].dtype}")
        
        # 估算文件大小
        tensor_size = np.prod(tensor_shape) * 4  # float32 = 4 bytes
        nu_size = np.prod(nu_shape) * 4
        coord_size = (len(x_coords) + len(y_coords)) * 4
        total_size = tensor_size + nu_size + coord_size
        
        print(f"\n=== 存储大小估算 ===")
        print(f"解场数据大小: {tensor_size / (1024**3):.2f} GB")
        print(f"扩散系数大小: {nu_size / (1024**3):.2f} GB")
        print(f"坐标数据大小: {coord_size / (1024**2):.2f} MB")
        print(f"总估算大小: {total_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    file_path = "PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5"
    analyze_darcy_dataset(file_path)