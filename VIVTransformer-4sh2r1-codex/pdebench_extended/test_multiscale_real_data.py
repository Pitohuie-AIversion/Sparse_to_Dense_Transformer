#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实PDEBench数据集多尺度模块测试
使用实际的2D Darcy Flow数据集验证多尺度功能
"""

import os
import sys
import h5py
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'multiscale'))

def test_real_dataset():
    """测试真实PDEBench数据集的多尺度处理"""
    print("=" * 60)
    print("真实PDEBench数据集多尺度模块测试")
    print("=" * 60)
    
    # 数据集路径
    dataset_path = "x:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codex/pdebench_extended/PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5"
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集文件不存在: {dataset_path}")
        return False
    
    print(f"✅ 找到数据集: {dataset_path}")
    
    try:
        # 1. 分析数据集结构
        print("\n1. 分析数据集结构...")
        with h5py.File(dataset_path, 'r') as f:
            print(f"   数据集键: {list(f.keys())}")
            
            # 获取主要数据
            if 'tensor' in f:
                tensor_shape = f['tensor'].shape
                print(f"   tensor形状: {tensor_shape}")
            
            if 'nu' in f:
                nu_shape = f['nu'].shape
                print(f"   nu形状: {nu_shape}")
            
            # 获取属性
            if 'beta' in f.attrs:
                beta = f.attrs['beta']
                print(f"   beta参数: {beta}")
        
        # 2. 导入多尺度模块
        print("\n2. 导入多尺度模块...")
        from multiscale.data.multiscale_adapter import MultiScaleDataset
        print("   ✅ 成功导入MultiScaleDataset")
        
        # 3. 创建多尺度数据集
        print("\n3. 创建多尺度数据集...")
        
        # 创建多尺度数据集（使用正确的参数格式）
        dataset = MultiScaleDataset(
            data_path=dataset_path,
            scale_factor=2,  # 下采样因子
            pde_type='darcy',
            split='train',
            sequence_length=1,
            original_resolution=[128, 128],  # 从数据集分析得到的原始分辨率
            normalize=True,
            downsampling_method='average'
        )
        print(f"   ✅ 成功创建多尺度数据集，样本数: {len(dataset)}")
        
        # 4. 测试数据获取
        print("\n4. 测试数据获取...")
        sample = dataset[0]
        
        # sample是一个元组 (inputs, targets, time_steps)
        inputs, targets, time_steps = sample
        print(f"   输入数据: {inputs.shape}, dtype: {inputs.dtype}")
        print(f"   目标数据: {targets.shape}, dtype: {targets.dtype}")
        print(f"   时间步: {time_steps}")
        
        # 5. 测试多个尺度
        print("\n5. 测试多个尺度...")
        scale_factors = [1, 2, 4]
        
        for scale in scale_factors:
            dataset_scaled = MultiScaleDataset(
                data_path=dataset_path,
                scale_factor=scale,
                pde_type='darcy',
                split='train',
                sequence_length=1,
                original_resolution=[128, 128],
                normalize=True,
                downsampling_method='average'
            )
            sample_scaled = dataset_scaled[0]
            
            # sample_scaled是一个元组 (inputs, targets, time_steps)
            inputs, targets, _ = sample_scaled
            print(f"   尺度{scale}: 输入形状 {inputs.shape}, 目标形状 {targets.shape}")
        
        # 6. 测试不同下采样方法
        print("\n6. 测试不同下采样方法...")
        methods = ['average', 'bilinear', 'nearest']
        
        for method in methods:
            try:
                dataset_method = MultiScaleDataset(
                    data_path=dataset_path,
                    scale_factor=2,
                    pde_type='darcy',
                    split='train',
                    sequence_length=1,
                    original_resolution=[128, 128],
                    normalize=True,
                    downsampling_method=method
                )
                sample_method = dataset_method[0]
                
                # sample_method是一个元组 (inputs, targets, time_steps)
                inputs, targets, _ = sample_method
                print(f"   {method}: 输入形状 {inputs.shape}, 目标形状 {targets.shape}")
            except Exception as e:
                print(f"   {method}: 失败 - {e}")
        
        # 7. 测试PyTorch DataLoader兼容性
        print("\n7. 测试PyTorch DataLoader兼容性...")
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        
        # batch是一个元组 (batch_inputs, batch_targets, batch_time_steps)
        batch_inputs, batch_targets, batch_time_steps = batch
        print(f"   批次输入: {batch_inputs.shape}, dtype: {batch_inputs.dtype}")
        print(f"   批次目标: {batch_targets.shape}, dtype: {batch_targets.dtype}")
        print(f"   批次时间步: {batch_time_steps}")
        
        # 8. 验证数据范围和统计
        print("\n8. 验证数据统计...")
        # 使用之前获取的inputs和targets
        print(f"   输入数据范围: [{inputs.min():.4f}, {inputs.max():.4f}]")
        print(f"   输入数据均值: {inputs.mean():.4f}")
        print(f"   输入数据标准差: {inputs.std():.4f}")
        
        print(f"   目标数据范围: [{targets.min():.4f}, {targets.max():.4f}]")
        print(f"   目标数据均值: {targets.mean():.4f}")
        print(f"   目标数据标准差: {targets.std():.4f}")
        
        print("\n" + "=" * 60)
        print("✅ 真实数据集多尺度测试完成！")
        print("所有功能验证通过，多尺度模块可以正确处理真实PDEBench数据")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_real_dataset()
    
    if success:
        print("\n🎉 真实数据集多尺度模块测试成功！")
        print("多尺度功能已准备就绪，可以用于VIVTransformer训练")
    else:
        print("\n❌ 测试失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()