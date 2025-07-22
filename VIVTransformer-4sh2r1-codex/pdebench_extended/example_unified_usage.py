#!/usr/bin/env python3
"""统一数据适配器使用示例

展示如何在实际项目中使用统一数据适配器来处理不同类型的数据集
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from utils.config import load_config
from data.unified_adapter import create_unified_datasets
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_simple_model(input_dim: int, output_dim: int) -> nn.Module:
    """创建一个简单的模型用于演示"""
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, output_dim)
    )

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 处理不同的批次格式
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                continue
        elif isinstance(batch, dict):
            inputs = batch.get('input', batch.get('inputs'))
            targets = batch.get('target', batch.get('targets'))
            if inputs is None or targets is None:
                continue
        else:
            print(f"未知的批次格式: {type(batch)}")
            continue
        
        # 移动到设备
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 只训练几个批次作为演示
        if batch_idx >= 2:
            break
    
    return total_loss / max(num_batches, 1)

def evaluate_model(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 处理不同的批次格式
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    continue
            elif isinstance(batch, dict):
                inputs = batch.get('input', batch.get('inputs'))
                targets = batch.get('target', batch.get('targets'))
                if inputs is None or targets is None:
                    continue
            else:
                continue
            
            # 移动到设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 只评估几个批次作为演示
            if batch_idx >= 2:
                break
    
    return total_loss / max(num_batches, 1)

def demo_pressure_field_training():
    """演示压力场数据集训练"""
    print("\n=== 压力场数据集训练演示 (20x20→200x200) ===")
    
    # 配置
    config = {
        'data': {
            'path': 'X:/2025/Graduation_project/simulation_results/merged_all_pressures_separated_normalized.pt',
            'batch_size': 16,
            'normalize': True,
            'num_workers': 2,
            'pin_memory': True
        }
    }
    
    try:
        # 创建数据集
        datasets = create_unified_datasets(config)
        info = datasets['info']
        
        print(f"数据类型: {info['data_type']}")
        print(f"输入维度: {info['input_dim']}, 输出维度: {info['output_dim']}")
        print(f"训练样本: {info['train_samples']}, 验证样本: {info['val_samples']}")
        
        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_simple_model(info['input_dim'], info['output_dim']).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print(f"使用设备: {device}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 训练几个epoch
        for epoch in range(3):
            train_loss = train_one_epoch(model, datasets['train'], criterion, optimizer, device)
            val_loss = evaluate_model(model, datasets['val'], criterion, device)
            print(f"Epoch {epoch+1}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f}")
        
        print("✓ 压力场数据集训练演示完成")
        
    except Exception as e:
        print(f"✗ 压力场数据集训练演示失败: {e}")
        import traceback
        traceback.print_exc()

def demo_pdebench_training():
    """演示PDEBench数据集训练"""
    print("\n=== PDEBench数据集训练演示 (Darcy Flow) ===")
    
    try:
        # 加载配置
        config = load_config('configs/darcy_flow_config.yaml')
        config['data']['batch_size'] = 16  # 减小批次大小用于演示
        
        # 创建数据集
        datasets = create_unified_datasets(config)
        info = datasets['info']
        
        print(f"PDE类型: {info.get('pde_type')}")
        print(f"空间分辨率: {info.get('spatial_resolution')}")
        print(f"输入维度: {info.get('input_dim')}")
        print(f"序列长度: {info.get('sequence_length')}")
        
        # 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = info.get('input_dim', 16384)
        model = create_simple_model(input_dim, input_dim).to(device)  # 输入输出维度相同
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print(f"使用设备: {device}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 训练几个epoch
        for epoch in range(3):
            train_loss = train_one_epoch(model, datasets['train'], criterion, optimizer, device)
            val_loss = evaluate_model(model, datasets['val'], criterion, device)
            print(f"Epoch {epoch+1}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f}")
        
        print("✓ PDEBench数据集训练演示完成")
        
    except Exception as e:
        print(f"✗ PDEBench数据集训练演示失败: {e}")
        import traceback
        traceback.print_exc()

def demo_automatic_detection():
    """演示自动数据类型检测"""
    print("\n=== 自动数据类型检测演示 ===")
    
    # 测试不同的数据路径
    test_configs = [
        {
            'name': '压力场数据',
            'config': {
                'data': {
                    'path': 'X:/2025/Graduation_project/simulation_results/merged_all_pressures_separated_normalized.pt',
                    'batch_size': 8
                }
            }
        },
        {
            'name': 'PDEBench数据',
            'config': load_config('configs/darcy_flow_config.yaml')
        }
    ]
    
    for test in test_configs:
        try:
            print(f"\n测试 {test['name']}:")
            datasets = create_unified_datasets(test['config'])
            info = datasets['info']
            
            if info.get('data_type') == 'pressure_field':
                print(f"  ✓ 检测为压力场数据: {info['input_shape']} → {info['output_shape']}")
            else:
                print(f"  ✓ 检测为PDEBench数据: {info.get('pde_type')}, 分辨率: {info.get('spatial_resolution')}")
                
        except Exception as e:
            print(f"  ✗ 检测失败: {e}")

def main():
    """主函数"""
    print("统一数据适配器使用演示")
    print("=" * 50)
    
    # 演示自动检测
    demo_automatic_detection()
    
    # 演示压力场数据集训练
    demo_pressure_field_training()
    
    # 演示PDEBench数据集训练
    demo_pdebench_training()
    
    print("\n=== 演示总结 ===")
    print("✓ 统一数据适配器可以自动检测和处理不同类型的数据集")
    print("✓ 支持原始20x20→200x200压力场预测任务")
    print("✓ 支持PDEBench标准数据集（Darcy Flow等）")
    print("✓ 提供统一的接口，简化数据处理流程")
    print("\n🎉 统一数据适配器演示完成！")

if __name__ == '__main__':
    main()