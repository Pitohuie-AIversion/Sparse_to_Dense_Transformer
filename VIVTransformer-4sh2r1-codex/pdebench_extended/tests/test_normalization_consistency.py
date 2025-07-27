#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试归一化一致性
验证输入和输出数据是否使用相同的归一化参数
"""

import torch
import numpy as np
from multiscale.data.multiscale_adapter import MultiScaleDataset
from data.pdebench_adapter import PDEBenchDataset
import matplotlib.pyplot as plt

def test_normalization_consistency():
    """测试归一化一致性"""
    print("=== 归一化一致性测试 ===")
    
    # 创建基础数据集
    base_dataset = PDEBenchDataset(
        pde_type="burgers",
        data_dir="data",
        split="train",
        normalize=False  # 不在基础数据集中归一化
    )
    
    # 创建多尺度数据集
    multiscale_dataset = MultiScaleDataset(
        base_dataset=base_dataset,
        scale_factor=4,
        downsampling_method="bilinear",
        normalize=True,  # 在多尺度数据集中归一化
        enable_center_crop=True,
        center_crop_input_resolution=[28, 28],
        center_crop_output_resolution=[112, 112]
    )
    
    print(f"\n数据集信息:")
    print(f"- 基础数据集大小: {len(base_dataset)}")
    print(f"- 多尺度数据集大小: {len(multiscale_dataset)}")
    print(f"- 归一化参数: mean={multiscale_dataset.data_mean:.6f}, std={multiscale_dataset.data_std:.6f}")
    
    # 获取几个样本进行测试
    num_samples = 5
    input_stats = []
    output_stats = []
    
    print(f"\n=== 分析前{num_samples}个样本的统计特性 ===")
    
    for i in range(num_samples):
        try:
            inputs, targets, _ = multiscale_dataset[i]
            
            # 计算统计量
            input_mean = torch.mean(inputs).item()
            input_std = torch.std(inputs).item()
            target_mean = torch.mean(targets).item()
            target_std = torch.std(targets).item()
            
            input_stats.append((input_mean, input_std))
            output_stats.append((target_mean, target_std))
            
            print(f"\n样本 {i+1}:")
            print(f"  输入形状: {inputs.shape}")
            print(f"  输出形状: {targets.shape}")
            print(f"  输入统计: mean={input_mean:.6f}, std={input_std:.6f}")
            print(f"  输出统计: mean={target_mean:.6f}, std={target_std:.6f}")
            print(f"  均值差异: {abs(input_mean - target_mean):.6f}")
            print(f"  标准差差异: {abs(input_std - target_std):.6f}")
            
        except Exception as e:
            print(f"样本 {i+1} 处理失败: {e}")
            continue
    
    # 计算整体统计
    if input_stats and output_stats:
        avg_input_mean = np.mean([s[0] for s in input_stats])
        avg_input_std = np.mean([s[1] for s in input_stats])
        avg_output_mean = np.mean([s[0] for s in output_stats])
        avg_output_std = np.mean([s[1] for s in output_stats])
        
        print(f"\n=== 整体统计分析 ===")
        print(f"输入数据平均统计: mean={avg_input_mean:.6f}, std={avg_input_std:.6f}")
        print(f"输出数据平均统计: mean={avg_output_mean:.6f}, std={avg_output_std:.6f}")
        print(f"均值差异: {abs(avg_input_mean - avg_output_mean):.6f}")
        print(f"标准差差异: {abs(avg_input_std - avg_output_std):.6f}")
        
        # 判断归一化一致性
        mean_diff = abs(avg_input_mean - avg_output_mean)
        std_diff = abs(avg_input_std - avg_output_std)
        
        print(f"\n=== 归一化一致性评估 ===")
        if mean_diff < 0.1 and std_diff < 0.1:
            print("✅ 归一化一致性良好：输入和输出数据的统计特性基本一致")
        elif mean_diff < 0.5 and std_diff < 0.5:
            print("⚠️  归一化一致性一般：输入和输出数据存在一定差异")
        else:
            print("❌ 归一化一致性差：输入和输出数据的统计特性差异较大")
            print("   这可能导致模型训练不稳定和预测不准确")
        
        print(f"\n建议:")
        print(f"- 确保输入和输出使用相同的归一化参数")
        print(f"- 考虑在下采样后重新计算归一化统计量")
        print(f"- 或者使用原始数据的统计量但确保处理过程不改变分布")

if __name__ == "__main__":
    test_normalization_consistency()