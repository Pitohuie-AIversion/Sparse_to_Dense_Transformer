#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.pdebench_adapter import create_pdebench_datasets
from utils.config import load_config

def test_data_shape():
    """测试修复后的数据形状"""
    print("=== 测试数据形状 ===")
    
    # 加载配置
    config = load_config('configs/darcy_flow_config.yaml')
    print(f"配置加载成功: {config['experiment']['name']}")
    
    # 创建数据集
    datasets = create_pdebench_datasets(config)
    train_loader = datasets['train']
    
    # 获取一个批次
    batch = next(iter(train_loader))
    inputs, targets = batch
    
    print(f"输入形状: {inputs.shape}")
    print(f"目标形状: {targets.shape}")
    
    # 检查形状是否正确
    batch_size, seq_len, input_dim = inputs.shape
    print(f"批次大小: {batch_size}")
    print(f"序列长度: {seq_len}")
    print(f"输入维度: {input_dim}")
    
    if seq_len > 0:
        print("✓ 数据形状修复成功！")
    else:
        print("✗ 数据形状仍有问题")
    
    return inputs.shape, targets.shape

if __name__ == "__main__":
    test_data_shape()