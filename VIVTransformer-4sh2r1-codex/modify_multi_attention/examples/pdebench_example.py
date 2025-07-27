#!/usr/bin/env python3
"""PDEBench数据集使用示例

本脚本展示如何使用PDEBench数据集进行训练和评估。
包括数据加载、模型训练、结果可视化等功能。
"""

import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from data.dataloader import get_pdebench_loaders, get_adaptive_loaders
    from mymodels.model_factory import create_model
    from utils.config import load_config
    from utils.logger import setup_logger
    from utils.system import set_seed
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("这是一个演示脚本，需要完整的项目环境才能运行")
    print("请确保在项目根目录下运行，或者安装完整的项目依赖")
    
    # 创建模拟函数用于演示
    def get_adaptive_loaders(*args, **kwargs):
        raise FileNotFoundError("演示模式：数据文件未找到")
    
    def create_model(*args, **kwargs):
        return None
    
    def load_config(*args, **kwargs):
        return {}
    
    def setup_logger(*args, **kwargs):
        return None
    
    def set_seed(*args, **kwargs):
        pass


def create_pdebench_config() -> Dict[str, Any]:
    """创建PDEBench配置示例"""
    config = {
        'global': {
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'result_dir': './results/pdebench_example'
        },
        'data': {
            'use_pdebench': True,
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
            'normalize': True,
            'use_augmentation': False
        },
        'current_pde': 'darcy_flow',  # 当前使用的PDE类型
        'pdebench': {
            'data_root': './PDEBench/pdebench/data_download/data/2D/DarcyFlow',
            'pde_configs': {
                'ns_incom': {
                    'data_file': 'ns_incom_inhom_2d.h5',
                    'spatial_resolution': [64, 64],
                    'sequence_length': 49,
                    'input_dim': 4096,  # 64*64
                    'output_dim': 4096,
                    'description': 'Navier-Stokes incompressible flow'
                },
                'darcy_flow': {
                    'data_file': '2D_DarcyFlow_beta0.1_Train.hdf5',
                    'spatial_resolution': [32, 32],
                    'sequence_length': 1,
                    'input_dim': 1024,  # 32*32
                    'output_dim': 1024,
                    'description': 'Darcy flow equation'
                },
                'shallow_water': {
                    'data_file': 'shallow_water_2d.h5',
                    'spatial_resolution': [128, 128],
                    'sequence_length': 40,
                    'input_dim': 16384,  # 128*128
                    'output_dim': 16384,
                    'description': 'Shallow water equations'
                }
            }
        },
        'model': {
            'input_dim': 4096,
            'output_dim': 4096,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'attention_type': 'MultiHeadAttention'
        },
        'training': {
            'epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'scheduler': 'cosine'
        },
        'attention_test': {
            'types': ['MultiHeadAttention', 'ECAAttention', 'SEAttention']
        }
    }
    return config


def demonstrate_data_loading(config: Dict[str, Any]):
    """演示PDEBench数据加载"""
    print("\n=== PDEBench数据加载演示 ===")
    
    # 获取当前PDE类型
    pde_type = config['current_pde']
    pde_config = config['pdebench']['pde_configs'][pde_type]
    
    print(f"当前PDE类型: {pde_type}")
    print(f"描述: {pde_config['description']}")
    print(f"空间分辨率: {pde_config['spatial_resolution']}")
    print(f"序列长度: {pde_config['sequence_length']}")
    
    try:
        # 使用自适应数据加载器
        train_loader, valid_loader, test_loader = get_adaptive_loaders(config)
        
        print(f"\n数据加载成功!")
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(valid_loader)}")
        print(f"测试集批次数: {len(test_loader)}")
        
        # 检查数据形状
        for batch_idx, (inputs, targets, time_steps) in enumerate(train_loader):
            print(f"\n第一个批次数据形状:")
            print(f"输入: {inputs.shape}")
            print(f"目标: {targets.shape}")
            print(f"时间步: {time_steps.shape if time_steps is not None else 'None'}")
            break
            
        return train_loader, valid_loader, test_loader
        
    except FileNotFoundError as e:
        print(f"\n⚠️ 数据文件未找到: {e}")
        print("请确保PDEBench数据文件已下载到正确位置")
        return None, None, None
    except Exception as e:
        print(f"\n❌ 数据加载失败: {e}")
        return None, None, None


def demonstrate_model_creation(config: Dict[str, Any]):
    """演示模型创建"""
    print("\n=== 模型创建演示 ===")
    
    try:
        # 获取当前PDE配置
        pde_type = config['current_pde']
        pde_config = config['pdebench']['pde_configs'][pde_type]
        
        # 更新模型配置
        config['model']['input_dim'] = pde_config['input_dim']
        config['model']['output_dim'] = pde_config['output_dim']
        
        # 创建模型
        model = create_model(
            config['model'],
            config['model']['attention_type'],
            config['global']['device']
        )
        
        print(f"模型创建成功!")
        print(f"模型类型: {type(model).__name__}")
        print(f"输入维度: {config['model']['input_dim']}")
        print(f"输出维度: {config['model']['output_dim']}")
        print(f"注意力机制: {config['model']['attention_type']}")
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型参数统计:")
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None


def demonstrate_training_loop(model, train_loader, valid_loader, config: Dict[str, Any]):
    """演示训练循环"""
    print("\n=== 训练循环演示 ===")
    
    if model is None or train_loader is None:
        print("⚠️ 模型或数据加载器未准备好，跳过训练演示")
        return
    
    device = torch.device(config['global']['device'])
    model = model.to(device)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = torch.nn.MSELoss()
    
    print(f"设备: {device}")
    print(f"优化器: Adam")
    print(f"学习率: {config['training']['learning_rate']}")
    print(f"损失函数: MSE")
    
    # 简单的训练循环演示（只训练几个批次）
    model.train()
    total_loss = 0.0
    num_batches = min(5, len(train_loader))  # 只训练前5个批次作为演示
    
    print(f"\n开始训练演示（{num_batches}个批次）...")
    
    for batch_idx, (inputs, targets, time_steps) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"批次 {batch_idx + 1}/{num_batches}, 损失: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    print(f"\n平均训练损失: {avg_loss:.6f}")
    
    # 验证演示
    if valid_loader is not None:
        model.eval()
        val_loss = 0.0
        num_val_batches = min(3, len(valid_loader))
        
        print(f"\n开始验证演示（{num_val_batches}个批次）...")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, time_steps) in enumerate(valid_loader):
                if batch_idx >= num_val_batches:
                    break
                    
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                print(f"验证批次 {batch_idx + 1}/{num_val_batches}, 损失: {loss.item():.6f}")
        
        avg_val_loss = val_loss / num_val_batches
        print(f"\n平均验证损失: {avg_val_loss:.6f}")


def save_example_config(config: Dict[str, Any], output_path: str):
    """保存示例配置文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n示例配置已保存到: {output_path}")


def main():
    """主函数"""
    print("🚀 PDEBench数据集使用示例")
    print("=" * 50)
    
    # 设置随机种子
    set_seed(42)
    
    # 创建配置
    config = create_pdebench_config()
    
    # 保存示例配置
    config_path = project_root / "configs" / "pdebench_example.yaml"
    save_example_config(config, config_path)
    
    # 演示数据加载
    train_loader, valid_loader, test_loader = demonstrate_data_loading(config)
    
    # 演示模型创建
    model = demonstrate_model_creation(config)
    
    # 演示训练循环
    demonstrate_training_loop(model, train_loader, valid_loader, config)
    
    print("\n" + "=" * 50)
    print("✅ PDEBench示例演示完成!")
    print("\n📝 使用说明:")
    print("1. 下载PDEBench数据集到 ./data/pdebench/ 目录")
    print("2. 修改配置文件中的数据路径和PDE类型")
    print("3. 运行完整训练: python main.py -c configs/pdebench_example.yaml")
    print("4. 查看结果和可视化")
    
    print("\n🔗 相关资源:")
    print("- PDEBench GitHub: https://github.com/pdebench/PDEBench")
    print("- 数据下载: 请参考PDEBench官方文档")
    print("- 配置文件: configs/pdebench_example.yaml")


if __name__ == "__main__":
    main()