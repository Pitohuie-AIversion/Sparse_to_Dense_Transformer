#!/usr/bin/env python3
"""2D Darcy Flow 示例脚本

本脚本演示如何使用VIVTransformer处理2D Darcy Flow数据集。
Darcy Flow是描述多孔介质中流体流动的椭圆型偏微分方程。
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


def load_darcy_flow_config() -> Dict[str, Any]:
    """加载Darcy Flow配置"""
    config_path = Path(__file__).parent.parent / "configs" / "darcy_flow_config.yaml"
    
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
        logger.error(f"Darcy Flow数据文件不存在: {data_file}")
        logger.info("请确保已下载PDEBench数据集并放置在正确位置")
        return False
    
    logger.info(f"找到Darcy Flow数据文件: {data_file}")
    return True


def create_darcy_flow_model(config: Dict[str, Any]) -> torch.nn.Module:
    """创建Darcy Flow模型"""
    model_config = config['model']
    
    # 添加缺失的配置项
    if 'max_time_steps' not in model_config:
        model_config['max_time_steps'] = config['pdebench']['pde_configs']['darcy_flow']['sequence_length']
    
    # 从配置中获取模型参数
    device = config['global']['device']
    model = create_model(
        config=config,
        attention_type=model_config['attention_type'],
        device=device
    )
    
    return model


def test_darcy_flow_data_loading(config: Dict[str, Any]):
    """测试Darcy Flow数据加载"""
    logger.info("开始测试Darcy Flow数据加载...")
    
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
            sequence_length=config['pdebench']['pde_configs']['darcy_flow']['sequence_length'],
            spatial_resolution=config['pdebench']['pde_configs']['darcy_flow']['spatial_resolution'],
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
            
            if i >= 2:  # 只测试前3个批次
                break
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return None


def test_darcy_flow_model(config: Dict[str, Any], data_loaders):
    """测试Darcy Flow模型"""
    if data_loaders is None:
        logger.error("数据加载器为空，跳过模型测试")
        return
    
    logger.info("开始测试Darcy Flow模型...")
    
    train_loader, val_loader, test_loader = data_loaders
    
    try:
        # 创建模型
        model = create_darcy_flow_model(config)
        logger.info(f"模型创建成功: {model.__class__.__name__}")
        
        # 设置设备
        device = torch.device(config['global']['device'] if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
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
                loss = torch.nn.functional.mse_loss(outputs, targets)
                logger.info(f"  MSE损失: {loss.item():.6f}")
                
                if i >= 1:  # 只测试前2个批次
                    break
        
        logger.info("模型测试完成")
        return model
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return None


def visualize_darcy_flow_sample(data_loaders, config: Dict[str, Any]):
    """可视化Darcy Flow样本"""
    if data_loaders is None:
        logger.error("数据加载器为空，跳过可视化")
        return
    
    logger.info("开始可视化Darcy Flow样本...")
    
    train_loader, _, _ = data_loaders
    
    try:
        # 获取一个批次的数据
        inputs, targets, _ = next(iter(train_loader))
        
        # 选择第一个样本
        sample_input = inputs[0, 0].numpy()  # [H*W]
        sample_target = targets[0, 0].numpy()  # [H*W]
        
        # 重塑为2D
        spatial_res = config['pdebench']['pde_configs']['darcy_flow']['spatial_resolution']
        H, W = spatial_res
        
        input_2d = sample_input.reshape(H, W)
        target_2d = sample_target.reshape(H, W)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 输入（渗透率场）
        im1 = axes[0].imshow(input_2d, cmap='viridis')
        axes[0].set_title('输入: 渗透率场')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        # 目标（压力场）
        im2 = axes[1].imshow(target_2d, cmap='plasma')
        axes[1].set_title('目标: 压力场')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        # 差异
        diff = target_2d - input_2d
        im3 = axes[2].imshow(diff, cmap='RdBu_r')
        axes[2].set_title('差异: 目标 - 输入')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        # 保存图像
        save_path = Path("darcy_flow_sample_visualization.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"可视化失败: {e}")


def main():
    """主函数"""
    logger.info("=== 2D Darcy Flow 示例脚本 ===")
    
    # 加载配置
    config = load_darcy_flow_config()
    if config is None:
        logger.error("配置加载失败")
        return
    
    logger.info(f"配置加载成功: {config['experiment']['name']}")
    
    # 测试数据加载
    data_loaders = test_darcy_flow_data_loading(config)
    
    # 测试模型
    model = test_darcy_flow_model(config, data_loaders)
    
    # 可视化样本
    visualize_darcy_flow_sample(data_loaders, config)
    
    logger.info("=== Darcy Flow 示例完成 ===")


if __name__ == "__main__":
    main()