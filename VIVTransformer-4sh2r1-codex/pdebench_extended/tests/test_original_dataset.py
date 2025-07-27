#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from data.dataset import PressureDataset
from torch.utils.data import DataLoader

def test_original_dataset():
    """测试原始20x20到200x200数据集的形状"""
    print("=== 测试原始数据集形状 ===")
    
    # 原始数据路径
    data_path = "X:/2025/Graduation_project/simulation_results/merged_all_pressures_separated_normalized.pt"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    try:
        # 创建数据集
        dataset = PressureDataset(data_path)
        print(f"✓ 数据集创建成功")
        print(f"数据集大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 获取一个批次
        batch = next(iter(dataloader))
        inputs, targets, time_steps = batch
        
        print(f"输入形状: {inputs.shape}")
        print(f"目标形状: {targets.shape}")
        print(f"时间步形状: {time_steps.shape}")
        
        # 检查维度
        batch_size, input_dim = inputs.shape
        _, output_dim = targets.shape
        
        print(f"批次大小: {batch_size}")
        print(f"输入维度: {input_dim} (应该是400 = 20x20)")
        print(f"输出维度: {output_dim} (应该是40000 = 200x200)")
        
        # 验证维度
        if input_dim == 400 and output_dim == 40000:
            print("✓ 数据维度正确！这是20x20到200x200的预测任务")
        else:
            print(f"❌ 数据维度不匹配，期望输入400，输出40000，实际输入{input_dim}，输出{output_dim}")
        
        # 测试重塑
        sample_input = inputs[0].view(20, 20)
        sample_output = targets[0].view(200, 200)
        print(f"重塑后输入形状: {sample_input.shape}")
        print(f"重塑后输出形状: {sample_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_original_dataset()