#!/usr/bin/env python3
"""
加载和可视化Darcy Flow HDF5数据的脚本

这个脚本可以:
1. 加载指定的HDF5文件
2. 显示数据的基本信息
3. 可视化数据内容
4. 保存可视化结果
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_and_inspect_hdf5(file_path):
    """
    加载HDF5文件并显示其结构和基本信息
    """
    print(f"正在加载文件: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        print("\n=== HDF5文件结构 ===")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"数据集: {name}, 形状: {obj.shape}, 数据类型: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"组: {name}")
        
        f.visititems(print_structure)
        
        # 获取数据的基本统计信息
        print("\n=== 数据统计信息 ===")
        
        # 检查常见的数据键
        common_keys = ['tensor', 'nu', 'data', 'x-coordinate', 'y-coordinate']
        available_keys = list(f.keys())
        
        print(f"可用的顶级键: {available_keys}")
        
        # 如果有数字键（样本索引），检查第一个样本
        numeric_keys = [k for k in available_keys if k.isdigit()]
        if numeric_keys:
            sample_key = numeric_keys[0]
            print(f"\n检查样本 {sample_key}:")
            sample_group = f[sample_key]
            for key in sample_group.keys():
                data = sample_group[key]
                if isinstance(data, h5py.Dataset):
                    print(f"  {key}: 形状={data.shape}, 类型={data.dtype}")
                    if data.size > 0:
                        arr = np.array(data)
                        print(f"    范围: [{np.min(arr):.6f}, {np.max(arr):.6f}]")
                        print(f"    均值: {np.mean(arr):.6f}, 标准差: {np.std(arr):.6f}")
        
        # 如果直接有tensor和nu键
        elif 'tensor' in f and 'nu' in f:
            print("\n检查tensor和nu数据:")
            tensor_data = np.array(f['tensor'])
            nu_data = np.array(f['nu'])
            
            print(f"tensor: 形状={tensor_data.shape}, 类型={tensor_data.dtype}")
            print(f"  范围: [{np.min(tensor_data):.6f}, {np.max(tensor_data):.6f}]")
            print(f"  均值: {np.mean(tensor_data):.6f}, 标准差: {np.std(tensor_data):.6f}")
            
            print(f"nu: 形状={nu_data.shape}, 类型={nu_data.dtype}")
            print(f"  范围: [{np.min(nu_data):.6f}, {np.max(nu_data):.6f}]")
            print(f"  均值: {np.mean(nu_data):.6f}, 标准差: {np.std(nu_data):.6f}")

def visualize_darcy_data(file_path, sample_idx=0, save_path=None):
    """
    可视化Darcy Flow数据
    """
    with h5py.File(file_path, 'r') as f:
        # 尝试不同的数据访问方式
        tensor_data = None
        nu_data = None
        
        # 方式1: 直接访问tensor和nu
        if 'tensor' in f and 'nu' in f:
            tensor_data = np.array(f['tensor'][sample_idx])
            nu_data = np.array(f['nu'][sample_idx])
        
        # 方式2: 通过样本索引访问
        else:
            available_keys = list(f.keys())
            numeric_keys = [k for k in available_keys if k.isdigit()]
            
            if numeric_keys:
                sample_key = str(sample_idx).zfill(4)  # 补零到4位
                if sample_key in f:
                    sample_group = f[sample_key]
                    if 'tensor' in sample_group:
                        tensor_data = np.array(sample_group['tensor'])
                    if 'nu' in sample_group:
                        nu_data = np.array(sample_group['nu'])
                    elif 'data' in sample_group:
                        tensor_data = np.array(sample_group['data'])
        
        if tensor_data is None:
            print("错误: 无法找到tensor数据")
            return
        
        # 创建可视化
        if nu_data is not None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 显示tensor数据
            im1 = axes[0].imshow(tensor_data.squeeze(), cmap='viridis')
            axes[0].set_title(f'Tensor Data (样本 {sample_idx})')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[0])
            
            # 显示nu数据
            im2 = axes[1].imshow(nu_data.squeeze(), cmap='plasma')
            axes[1].set_title(f'Nu (扩散系数) (样本 {sample_idx})')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[1])
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.imshow(tensor_data.squeeze(), cmap='viridis')
            ax.set_title(f'Tensor Data (样本 {sample_idx})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()
        
        # 打印数据信息
        print(f"\n=== 样本 {sample_idx} 数据信息 ===")
        print(f"Tensor形状: {tensor_data.shape}")
        print(f"Tensor范围: [{np.min(tensor_data):.6f}, {np.max(tensor_data):.6f}]")
        if nu_data is not None:
            print(f"Nu形状: {nu_data.shape}")
            print(f"Nu范围: [{np.min(nu_data):.6f}, {np.max(nu_data):.6f}]")

def main():
    parser = argparse.ArgumentParser(description='加载和可视化Darcy Flow HDF5数据')
    parser.add_argument('--file_path', type=str, 
                       default='X:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codex/modify_multi_attention/PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5',
                       help='HDF5文件路径')
    parser.add_argument('--sample_idx', type=int, default=0, help='要可视化的样本索引')
    parser.add_argument('--inspect_only', action='store_true', help='只检查文件结构，不进行可视化')
    parser.add_argument('--save_path', type=str, help='保存可视化结果的路径')
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        print(f"错误: 文件不存在 {file_path}")
        return
    
    # 检查文件结构
    load_and_inspect_hdf5(file_path)
    
    if not args.inspect_only:
        # 可视化数据
        print("\n开始可视化...")
        save_path = args.save_path or f"darcy_visualization_sample_{args.sample_idx}.png"
        visualize_darcy_data(file_path, args.sample_idx, save_path)

if __name__ == '__main__':
    main()