#!/usr/bin/env python3
"""
简单的训练监控系统测试脚本
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modify_multi_attention.utils.hardware_monitor import HardwareMonitor
from modify_multi_attention.utils.enhanced_logger import EnhancedTrainingLogger
from modify_multi_attention.utils.training_visualizer import TrainingVisualizer

def test_hardware_monitor():
    """测试硬件监控功能"""
    print("\n=== 测试硬件监控功能 ===")
    
    monitor = HardwareMonitor(
        log_dir="test_logs",
        enable_gpu_monitoring=True
    )
    
    # 开始监控
    monitor.start_training()
    
    # 模拟一些计算负载
    print("模拟计算负载...")
    for i in range(3):
        monitor.start_epoch(i)
        
        # 创建一些张量进行计算
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x.T)
            del x, y
            torch.cuda.empty_cache()
        
        time.sleep(1)
        
        # 结束epoch监控
        train_loss = 0.5 - i * 0.1
        valid_loss = 0.6 - i * 0.08
        monitor.end_epoch(i, train_loss, valid_loss)
    
    # 获取当前指标
    metrics = monitor.get_current_metrics()
    
    # 结束训练监控
    monitor.end_training()
    
    print(f"监控结果:")
    print(f"- CPU使用率: {metrics.get('cpu_percent', 'N/A')}%")
    print(f"- 内存使用: {metrics.get('memory_percent', 'N/A')}%")
    if torch.cuda.is_available():
        print(f"- GPU使用率: {metrics.get('gpu_percent', 'N/A')}%")
        print(f"- GPU内存: {metrics.get('gpu_memory_percent', 'N/A')}%")
    
    return metrics

def test_enhanced_logger():
    """测试增强日志功能"""
    print("\n=== 测试增强日志功能 ===")
    
    # 创建日志目录
    log_dir = Path("test_logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = EnhancedTrainingLogger(
        log_dir=log_dir,
        experiment_name="test_experiment"
    )
    
    # 模拟一些日志记录
    print("记录训练开始...")
    
    for epoch in range(3):
        epoch_info = logger.log_epoch_start(epoch, 5, 0.001)
        
        for batch in range(2):
            # 模拟batch训练进度
            loss = 1.0 - epoch * 0.2 - batch * 0.1
            batch_time = 0.5
            data_time = 0.1
            gpu_memory = 1024.0 if torch.cuda.is_available() else None
            gpu_util = 85.0 if torch.cuda.is_available() else None
            
            logger.log_batch_progress(
                epoch=epoch,
                batch_idx=batch,
                total_batches=2,
                loss=loss,
                batch_time=batch_time,
                data_time=data_time,
                gpu_memory_mb=gpu_memory,
                gpu_utilization=gpu_util
            )
        
        # 记录epoch总结
        logger.log_epoch_summary(
            epoch=epoch,
            epoch_start_info=epoch_info,
            train_loss=1.0 - epoch * 0.2,
            valid_loss=1.1 - epoch * 0.15,
            test_loss=1.05 - epoch * 0.18,
            epoch_time=10.0,
            best_loss=1.1 - epoch * 0.15,
            patience_counter=0,
            model_saved=epoch == 2,
            gradient_norm=0.5,
            learning_rate=0.001
        )
    
    print("记录训练结束...")
    
    print("增强日志测试完成，日志文件保存在 test_logs/ 目录")
    return log_dir

def test_training_visualizer(log_dir):
    """测试训练可视化功能"""
    print("\n=== 测试训练可视化功能 ===")
    
    # 创建模拟的训练数据
    training_data = {
        'train_losses': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val_losses': [1.1, 0.9, 0.7, 0.6, 0.5],
        'train_accuracies': [0.6, 0.7, 0.75, 0.8, 0.85],
        'val_accuracies': [0.55, 0.65, 0.72, 0.78, 0.83],
        'learning_rates': [0.001, 0.001, 0.0008, 0.0006, 0.0004],
        'epochs': list(range(1, 6))
    }
    
    # 创建模拟的硬件数据
    hardware_data = {
        'timestamps': [f"2025-01-{i+1:02d} 10:00:00" for i in range(5)],
        'cpu_percent': [45.2, 52.1, 48.7, 51.3, 49.8],
        'memory_percent': [68.5, 72.1, 70.3, 73.8, 71.2],
        'gpu_percent': [85.2, 88.7, 86.1, 89.3, 87.5] if torch.cuda.is_available() else [],
        'gpu_memory_percent': [76.3, 79.8, 77.5, 81.2, 78.9] if torch.cuda.is_available() else []
    }
    
    visualizer = TrainingVisualizer(log_dir)
    
    # 测试各种可视化功能
    print("生成训练损失曲线...")
    loss_curve_path = visualizer.plot_loss_curves()
    
    print("生成硬件使用情况图表...")
    hardware_usage_path = visualizer.plot_hardware_usage()
    
    print("生成训练仪表板...")
    dashboard_path = visualizer.create_training_dashboard()
    
    print("生成HTML报告...")
    report_path = visualizer.generate_html_report()
    
    print(f"训练可视化测试完成！")
    print(f"- 损失曲线: {loss_curve_path}")
    print(f"- 硬件使用: {hardware_usage_path}")
    print(f"- 仪表板: {dashboard_path}")
    print(f"- 完整报告: {report_path}")
    
    return dashboard_path, report_path

def main():
    """主测试函数"""
    print("开始测试VIVTransformer训练监控系统...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
    
    try:
        # 测试硬件监控
        hardware_metrics = test_hardware_monitor()
        
        # 测试增强日志
        log_dir = test_enhanced_logger()
        
        # 测试训练可视化
        dashboard_path, report_path = test_training_visualizer(log_dir)
        
        print("\n=== 测试总结 ===")
        print("✅ 硬件监控功能正常")
        print("✅ 增强日志功能正常")
        print("✅ 训练可视化功能正常")
        print(f"\n📊 查看结果:")
        print(f"- 日志目录: {log_dir.absolute()}")
        print(f"- 训练仪表板: {dashboard_path}")
        print(f"- 完整报告: {report_path}")
        
        print("\n🎉 所有训练监控功能测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)