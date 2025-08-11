#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控示例脚本

这个脚本展示了如何使用增强的训练监控功能，包括：
- 硬件监控
- 增强日志记录
- 训练可视化
- 自动报告生成
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger
from utils.system import set_seed, create_timestamped_dir
from utils.hardware_monitor import HardwareMonitor
from utils.enhanced_logger import EnhancedTrainingLogger
from utils.training_visualizer import TrainingVisualizer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_training_environment(config: dict) -> tuple:
    """设置训练环境
    
    Returns:
        tuple: (device, result_dir, logger)
    """
    # 设置随机种子
    set_seed(config['global']['seed'])
    
    # 设置设备
    device = torch.device(config['global']['device'])
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 创建结果目录
    result_dir = create_timestamped_dir(
        base_dir=config['global']['result_dir'],
        prefix="training_with_monitoring"
    )
    
    # 设置日志
    logger = setup_logger(
        name="training_monitor",
        log_file=str(result_dir / "training.log"),
        level=logging.INFO
    )
    
    return device, result_dir, logger


def demonstrate_hardware_monitoring(result_dir: Path, config: dict):
    """演示硬件监控功能"""
    print("\n=== 硬件监控演示 ===")
    
    # 初始化硬件监控器
    hardware_monitor = HardwareMonitor(
        log_dir=str(result_dir / "hardware_logs"),
        enable_gpu_monitoring=config.get("hardware_monitoring", {}).get("enable_gpu_monitoring", True)
    )
    
    # 开始训练监控
    hardware_monitor.start_training()
    
    # 模拟训练过程
    for epoch in range(3):
        print(f"模拟 Epoch {epoch + 1}")
        hardware_monitor.start_epoch(epoch + 1)
        
        # 模拟批次训练
        for batch in range(5):
            hardware_monitor.start_batch(batch + 1)
            
            # 模拟一些计算
            if torch.cuda.is_available():
                x = torch.randn(32, 100, device='cuda')
                y = torch.matmul(x, x.transpose(-2, -1))
                torch.cuda.synchronize()
            
            hardware_monitor.end_batch()
        
        hardware_monitor.end_epoch()
    
    # 结束训练监控
    hardware_monitor.end_training()
    
    # 获取训练摘要
    summary = hardware_monitor.get_training_summary()
    print(f"训练摘要: {summary}")
    
    return hardware_monitor


def demonstrate_enhanced_logging(result_dir: Path):
    """演示增强日志记录功能"""
    print("\n=== 增强日志记录演示 ===")
    
    # 初始化增强日志记录器
    enhanced_logger = EnhancedTrainingLogger(
        log_dir=str(result_dir / "enhanced_logs")
    )
    
    # 记录训练开始
    enhanced_logger.log_training_start({
        'model_name': 'VIVTransformer',
        'dataset': 'demo_dataset',
        'batch_size': 32,
        'learning_rate': 0.001
    })
    
    # 模拟记录训练过程
    for epoch in range(3):
        for batch in range(5):
            # 记录批次指标
            enhanced_logger.log_batch_metrics(
                epoch=epoch + 1,
                batch=batch + 1,
                metrics={
                    'loss': 0.5 - (epoch * 0.1 + batch * 0.01),
                    'accuracy': 0.7 + (epoch * 0.05 + batch * 0.005),
                    'learning_rate': 0.001 * (0.9 ** epoch)
                }
            )
        
        # 记录epoch摘要
        enhanced_logger.log_epoch_summary(
            epoch=epoch + 1,
            train_loss=0.5 - epoch * 0.1,
            val_loss=0.6 - epoch * 0.08,
            metrics={'accuracy': 0.7 + epoch * 0.05}
        )
    
    # 记录训练结束
    enhanced_logger.log_training_end({
        'final_loss': 0.2,
        'best_accuracy': 0.85,
        'total_epochs': 3
    })
    
    # 保存日志
    enhanced_logger.save_logs()
    
    return enhanced_logger


def demonstrate_training_visualization(result_dir: Path):
    """演示训练可视化功能"""
    print("\n=== 训练可视化演示 ===")
    
    try:
        # 初始化训练可视化器
        visualizer = TrainingVisualizer(str(result_dir / "hardware_logs"))
        
        # 生成完整报告
        visualizer.generate_complete_report()
        
        print(f"可视化报告已生成到: {result_dir / 'hardware_logs' / 'visualizations'}")
        
        # 列出生成的文件
        viz_dir = result_dir / 'hardware_logs' / 'visualizations'
        if viz_dir.exists():
            print("生成的文件:")
            for file in viz_dir.iterdir():
                print(f"  - {file.name}")
        
        return visualizer
        
    except Exception as e:
        print(f"可视化演示失败: {e}")
        return None


def main():
    """主函数"""
    print("🚀 训练监控功能演示")
    
    # 加载配置
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        print("请确保在项目根目录下有 config.yaml 文件")
        return
    
    config = load_config(str(config_path))
    
    # 设置训练环境
    device, result_dir, logger = setup_training_environment(config)
    
    logger.info(f"训练监控演示开始，结果目录: {result_dir}")
    logger.info(f"使用设备: {device}")
    
    try:
        # 1. 演示硬件监控
        hardware_monitor = demonstrate_hardware_monitoring(result_dir, config)
        
        # 2. 演示增强日志记录
        enhanced_logger = demonstrate_enhanced_logging(result_dir)
        
        # 3. 演示训练可视化
        visualizer = demonstrate_training_visualization(result_dir)
        
        print(f"\n✅ 所有演示完成！")
        print(f"📁 结果保存在: {result_dir}")
        print(f"📊 查看可视化报告: {result_dir / 'hardware_logs' / 'visualizations' / 'training_dashboard.png'}")
        print(f"📋 查看HTML报告: {result_dir / 'hardware_logs' / 'visualizations' / 'training_report.html'}")
        
        logger.info("训练监控演示成功完成")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        print(f"❌ 演示失败: {e}")
        raise


if __name__ == "__main__":
    main()