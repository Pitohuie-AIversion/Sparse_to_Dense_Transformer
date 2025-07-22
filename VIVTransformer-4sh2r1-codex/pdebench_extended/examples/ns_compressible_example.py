#!/usr/bin/env python3
"""Compressible Navier-Stokes 示例脚本

本脚本演示如何使用VIVTransformer处理Compressible Navier-Stokes数据集。
Compressible Navier-Stokes方程描述可压缩流体的运动，是双曲型偏微分方程。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging
from typing import Dict, Any

from data.pdebench_adapter import create_pdebench_loaders
from mymodels.model_factory import create_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ns_compressible_config() -> Dict[str, Any]:
    """加载Compressible Navier-Stokes配置"""
    config_path = Path(__file__).parent.parent / "configs" / "ns_compressible_config.yaml"
    
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def check_data_availability(data_path: str) -> bool:
    """检查数据文件是否存在"""
    data_file = Path(data_path)
    if not data_file.exists():
        logger.error(f"Compressible Navier-Stokes数据文件不存在: {data_file}")
        logger.info("请确保已下载PDEBench数据集并放置在正确位置")
        return False
    
    logger.info(f"找到Compressible Navier-Stokes数据文件: {data_file}")
    return True


def create_ns_compressible_model(config: Dict[str, Any]) -> torch.nn.Module:
    """创建Compressible Navier-Stokes模型"""
    model_config = config['model']
    
    # 从配置中获取模型参数
    model = create_model(
        attention_type=model_config['attention_type'],
        input_dim=model_config['input_dim'],
        output_dim=model_config['output_dim'],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        dropout=model_config.get('dropout', 0.1),
        device=config['global']['device']
    )
    
    return model


def test_ns_compressible_data_loading(config: Dict[str, Any]):
    """测试Compressible Navier-Stokes数据加载"""
    logger.info("开始测试Compressible Navier-Stokes数据加载...")
    
    # 获取数据路径
    data_path = config['data']['path']
    
    # 检查数据可用性
    if not check_data_availability(data_path):
        return None
    
    try:
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_pdebench_loaders(
            data_path=data_path,
            pde_type=config['current_pde'],
            batch_size=config['data']['batch_size'],
            sequence_length=config['pdebench']['pde_configs']['ns_compressible']['sequence_length'],
            spatial_resolution=config['pdebench']['pde_configs']['ns_compressible']['spatial_resolution'],
            normalize=config['data']['normalize'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
        
        # 获取数据信息
        data_info = train_loader.get_data_info()
        logger.info(f"数据集信息: {data_info}")
        
        # 测试数据批次
        for i, (inputs, targets, time_steps) in enumerate(train_loader):
            logger.info(f"批次 {i+1}:")
            logger.info(f"  输入形状: {inputs.shape}")
            logger.info(f"  目标形状: {targets.shape}")
            logger.info(f"  时间步形状: {time_steps.shape if time_steps is not None else None}")
            
            # 检查数据范围
            logger.info(f"  输入数据范围: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
            logger.info(f"  目标数据范围: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
            
            if i >= 2:  # 只测试前3个批次
                break
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return None


def test_ns_compressible_model(config: Dict[str, Any], data_loaders):
    """测试Compressible Navier-Stokes模型"""
    if data_loaders is None:
        logger.error("数据加载器为空，跳过模型测试")
        return
    
    logger.info("开始测试Compressible Navier-Stokes模型...")
    
    train_loader, val_loader, test_loader = data_loaders
    
    try:
        # 创建模型
        model = create_ns_compressible_model(config)
        logger.info(f"模型创建成功: {model.__class__.__name__}")
        
        # 设置设备
        device = torch.device(config['global']['device'] if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets, time_steps) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                
                logger.info(f"前向传播测试 {i+1}:")
                logger.info(f"  输入形状: {inputs.shape}")
                logger.info(f"  输出形状: {outputs.shape}")
                logger.info(f"  目标形状: {targets.shape}")
                
                # 计算损失
                mse_loss = torch.nn.functional.mse_loss(outputs, targets)
                mae_loss = torch.nn.functional.l1_loss(outputs, targets)
                logger.info(f"  MSE损失: {mse_loss.item():.6f}")
                logger.info(f"  MAE损失: {mae_loss.item():.6f}")
                
                if i >= 1:  # 只测试前2个批次
                    break
        
        logger.info("模型测试完成")
        return model
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return None


def visualize_ns_compressible_sample(data_loaders, config: Dict[str, Any]):
    """可视化Compressible Navier-Stokes样本"""
    if data_loaders is None:
        logger.error("数据加载器为空，跳过可视化")
        return
    
    logger.info("开始可视化Compressible Navier-Stokes样本...")
    
    train_loader, _, _ = data_loaders
    
    try:
        # 获取一个批次的数据
        inputs, targets, time_steps = next(iter(train_loader))
        
        # 选择第一个样本的多个时间步
        sample_input = inputs[0].numpy()  # [T, H*W]
        sample_target = targets[0].numpy()  # [T, H*W]
        
        # 重塑为2D
        spatial_res = config['pdebench']['pde_configs']['ns_compressible']['spatial_resolution']
        H, W = spatial_res
        
        # 选择几个时间步进行可视化
        time_indices = [0, len(sample_input)//4, len(sample_input)//2, -1]
        
        fig, axes = plt.subplots(2, len(time_indices), figsize=(20, 10))
        
        for i, t_idx in enumerate(time_indices):
            # 输入时间步
            input_2d = sample_input[t_idx].reshape(H, W)
            im1 = axes[0, i].imshow(input_2d, cmap='viridis')
            axes[0, i].set_title(f'输入 t={t_idx}')
            axes[0, i].set_xlabel('x')
            if i == 0:
                axes[0, i].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0, i])
            
            # 目标时间步
            target_2d = sample_target[t_idx].reshape(H, W)
            im2 = axes[1, i].imshow(target_2d, cmap='plasma')
            axes[1, i].set_title(f'目标 t={t_idx}')
            axes[1, i].set_xlabel('x')
            if i == 0:
                axes[1, i].set_ylabel('y')
            plt.colorbar(im2, ax=axes[1, i])
        
        plt.tight_layout()
        
        # 保存图像
        save_path = Path("ns_compressible_sample_visualization.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存: {save_path}")
        
        plt.show()
        
        # 创建时间序列可视化
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # 计算每个时间步的平均值
        input_means = [sample_input[t].mean() for t in range(len(sample_input))]
        target_means = [sample_target[t].mean() for t in range(len(sample_target))]
        
        time_axis = range(len(input_means))
        ax.plot(time_axis, input_means, label='输入平均值', marker='o', markersize=3)
        ax.plot(time_axis, target_means, label='目标平均值', marker='s', markersize=3)
        ax.set_xlabel('时间步')
        ax.set_ylabel('场平均值')
        ax.set_title('Compressible Navier-Stokes 时间演化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存时间序列图
        save_path_ts = Path("ns_compressible_time_series.png")
        plt.savefig(save_path_ts, dpi=300, bbox_inches='tight')
        logger.info(f"时间序列可视化已保存: {save_path_ts}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"可视化失败: {e}")


def analyze_data_statistics(data_loaders):
    """分析数据统计信息"""
    if data_loaders is None:
        logger.error("数据加载器为空，跳过统计分析")
        return
    
    logger.info("开始分析数据统计信息...")
    
    train_loader, _, _ = data_loaders
    
    try:
        all_inputs = []
        all_targets = []
        
        # 收集前几个批次的数据
        for i, (inputs, targets, _) in enumerate(train_loader):
            all_inputs.append(inputs.numpy())
            all_targets.append(targets.numpy())
            
            if i >= 4:  # 只分析前5个批次
                break
        
        # 合并数据
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 计算统计量
        logger.info("数据统计信息:")
        logger.info(f"  输入数据形状: {all_inputs.shape}")
        logger.info(f"  输入均值: {all_inputs.mean():.6f}")
        logger.info(f"  输入标准差: {all_inputs.std():.6f}")
        logger.info(f"  输入最小值: {all_inputs.min():.6f}")
        logger.info(f"  输入最大值: {all_inputs.max():.6f}")
        
        logger.info(f"  目标数据形状: {all_targets.shape}")
        logger.info(f"  目标均值: {all_targets.mean():.6f}")
        logger.info(f"  目标标准差: {all_targets.std():.6f}")
        logger.info(f"  目标最小值: {all_targets.min():.6f}")
        logger.info(f"  目标最大值: {all_targets.max():.6f}")
        
    except Exception as e:
        logger.error(f"统计分析失败: {e}")


def main():
    """主函数"""
    logger.info("=== Compressible Navier-Stokes 示例脚本 ===")
    
    # 加载配置
    config = load_ns_compressible_config()
    if config is None:
        logger.error("配置加载失败")
        return
    
    logger.info(f"配置加载成功: {config['experiment']['name']}")
    
    # 测试数据加载
    data_loaders = test_ns_compressible_data_loading(config)
    
    # 分析数据统计
    analyze_data_statistics(data_loaders)
    
    # 测试模型
    model = test_ns_compressible_model(config, data_loaders)
    
    # 可视化样本
    visualize_ns_compressible_sample(data_loaders, config)
    
    logger.info("=== Compressible Navier-Stokes 示例完成 ===")


if __name__ == "__main__":
    main()