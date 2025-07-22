#!/usr/bin/env python3
"""测试统一数据适配器

验证统一数据适配器能否正确处理：
1. 原始20x20→200x200压力场数据集
2. PDEBench标准数据集
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import logging
from utils.config import load_config
from data.unified_adapter import UnifiedDataAdapter, create_unified_datasets

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_pressure_field_dataset():
    """测试20x20→200x200压力场数据集"""
    print("\n=== 测试压力场数据集 (20x20→200x200) ===")
    
    # 创建配置
    config = {
        'data': {
            'path': 'X:/2025/Graduation_project/simulation_results/merged_all_pressures_separated_normalized.pt',
            'batch_size': 32,
            'normalize': True,
            'num_workers': 4,
            'pin_memory': True
        }
    }
    
    try:
        # 创建适配器
        adapter = UnifiedDataAdapter(config)
        print(f"✓ 数据类型检测: {adapter.data_type}")
        
        # 创建数据集
        datasets = adapter.create_datasets()
        
        # 获取数据信息
        info = datasets['info']
        print(f"数据类型: {info['data_type']}")
        print(f"输入维度: {info['input_dim']} (应该是400 = 20x20)")
        print(f"输出维度: {info['output_dim']} (应该是40000 = 200x200)")
        print(f"输入形状: {info['input_shape']}")
        print(f"输出形状: {info['output_shape']}")
        print(f"总样本数: {info['total_samples']}")
        print(f"训练样本: {info['train_samples']}")
        print(f"验证样本: {info['val_samples']}")
        print(f"测试样本: {info['test_samples']}")
        
        # 测试数据加载
        train_loader = datasets['train']
        for batch_idx, (inputs, targets, time_steps) in enumerate(train_loader):
            print(f"\n批次 {batch_idx + 1}:")
            print(f"  输入形状: {inputs.shape}")
            print(f"  目标形状: {targets.shape}")
            print(f"  时间步形状: {time_steps.shape}")
            
            # 验证维度
            assert inputs.shape[1] == 400, f"输入维度错误: {inputs.shape[1]} != 400"
            assert targets.shape[1] == 40000, f"输出维度错误: {targets.shape[1]} != 40000"
            
            # 测试重塑
            input_reshaped = inputs[0].view(20, 20)
            target_reshaped = targets[0].view(200, 200)
            print(f"  重塑后输入形状: {input_reshaped.shape}")
            print(f"  重塑后目标形状: {target_reshaped.shape}")
            
            break  # 只测试第一个批次
        
        print("✓ 压力场数据集测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 压力场数据集测试失败: {e}")
        return False

def test_pdebench_dataset():
    """测试PDEBench数据集"""
    print("\n=== 测试PDEBench数据集 ===")
    
    try:
        # 加载PDEBench配置
        config = load_config('configs/darcy_flow_config.yaml')
        
        # 创建适配器
        adapter = UnifiedDataAdapter(config)
        print(f"✓ 数据类型检测: {adapter.data_type}")
        
        # 创建数据集
        datasets = adapter.create_datasets()
        
        # 获取数据信息
        info = datasets['info']
        print(f"数据类型: PDEBench")
        print(f"PDE类型: {info.get('pde_type', 'unknown')}")
        print(f"空间分辨率: {info.get('spatial_resolution', 'unknown')}")
        print(f"输入维度: {info.get('input_dim', 'unknown')}")
        print(f"序列长度: {info.get('sequence_length', 'unknown')}")
        print(f"训练样本: {info.get('train_samples', 'unknown')}")
        print(f"验证样本: {info.get('val_samples', 'unknown')}")
        print(f"测试样本: {info.get('test_samples', 'unknown')}")
        
        # 测试数据加载
        train_loader = datasets['train']
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, tuple) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get('input', batch.get('inputs'))
                targets = batch.get('target', batch.get('targets'))
            else:
                print(f"未知的批次格式: {type(batch)}")
                break
            
            print(f"\n批次 {batch_idx + 1}:")
            print(f"  输入形状: {inputs.shape}")
            print(f"  目标形状: {targets.shape}")
            
            break  # 只测试第一个批次
        
        print("✓ PDEBench数据集测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ PDEBench数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_unified_function():
    """测试统一创建函数"""
    print("\n=== 测试统一创建函数 ===")
    
    # 测试压力场数据集
    config1 = {
        'data': {
            'path': 'X:/2025/Graduation_project/simulation_results/merged_all_pressures_separated_normalized.pt',
            'batch_size': 16,
            'normalize': True
        }
    }
    
    try:
        datasets1 = create_unified_datasets(config1)
        info1 = datasets1['info']
        print(f"✓ 压力场数据集: {info1['data_type']}, 输入维度: {info1['input_dim']}, 输出维度: {info1['output_dim']}")
    except Exception as e:
        print(f"✗ 压力场数据集创建失败: {e}")
    
    # 测试PDEBench数据集
    try:
        config2 = load_config('configs/darcy_flow_config.yaml')
        datasets2 = create_unified_datasets(config2)
        info2 = datasets2['info']
        print(f"✓ PDEBench数据集: PDE类型: {info2.get('pde_type')}, 空间分辨率: {info2.get('spatial_resolution')}")
    except Exception as e:
        print(f"✗ PDEBench数据集创建失败: {e}")

def main():
    """主函数"""
    print("开始测试统一数据适配器...")
    
    # 测试压力场数据集
    success1 = test_pressure_field_dataset()
    
    # 测试PDEBench数据集
    success2 = test_pdebench_dataset()
    
    # 测试统一创建函数
    test_unified_function()
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"压力场数据集: {'✓ 通过' if success1 else '✗ 失败'}")
    print(f"PDEBench数据集: {'✓ 通过' if success2 else '✗ 失败'}")
    
    if success1 and success2:
        print("\n🎉 所有测试通过！统一数据适配器工作正常。")
    else:
        print("\n⚠️ 部分测试失败，请检查配置和数据路径。")

if __name__ == '__main__':
    main()