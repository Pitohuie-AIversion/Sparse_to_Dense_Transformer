#!/usr/bin/env python3
"""
增强版可视化训练脚本

集成了固定轮数预测可视化功能的完整训练脚本。
在训练过程中自动生成：
- 实时训练指标可视化
- 固定轮数预测结果对比
- 误差分析和训练进度跟踪
- 硬件监控和性能分析
"""

import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# 导入现有模块
from utils.training_visualizer import TrainingVisualizer
from utils.hardware_monitor import HardwareMonitor
from create_prediction_visualization import PredictionVisualizer

# 设置matplotlib后端和中文字体
plt.switch_backend('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """
    简单的神经网络模型用于演示
    """
    
    def __init__(self, input_size: int = 400, hidden_size: int = 512, output_size: int = 40000):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


class EnhancedTrainingLogger:
    """
    增强版训练日志记录器
    
    集成了传统训练可视化和预测结果可视化功能
    """
    
    def __init__(self, output_dir: str = "enhanced_training_output"):
        """
        初始化增强版训练日志记录器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建子目录
        self.plots_dir = self.output_dir / "plots"
        self.predictions_dir = self.output_dir / "predictions"
        self.logs_dir = self.output_dir / "logs"
        self.hardware_dir = self.output_dir / "hardware_logs"
        
        for dir_path in [self.plots_dir, self.predictions_dir, self.logs_dir, self.hardware_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 初始化各种可视化器
        self.training_visualizer = TrainingVisualizer(str(self.plots_dir))
        self.prediction_visualizer = PredictionVisualizer(str(self.predictions_dir))
        self.hardware_monitor = HardwareMonitor(str(self.hardware_dir))
        
        # 训练数据记录
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.learning_rates = []
        self.training_times = []
        
        # 预测历史记录
        self.prediction_history = []
        self.prediction_epochs = []
        
        # 硬件监控数据
        self.hardware_data = []
        
        logger.info(f"增强版训练日志记录器初始化完成，输出目录: {self.output_dir}")
    
    def start_hardware_monitoring(self):
        """
        开始硬件监控
        """
        try:
            self.hardware_monitor.start_monitoring()
            logger.info("硬件监控已启动")
        except Exception as e:
            logger.warning(f"硬件监控启动失败: {e}")
    
    def stop_hardware_monitoring(self):
        """
        停止硬件监控
        """
        try:
            self.hardware_monitor.stop_monitoring()
            logger.info("硬件监控已停止")
        except Exception as e:
            logger.warning(f"硬件监控停止失败: {e}")
    
    def log_epoch(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float = None, 
        learning_rate: float = None,
        epoch_time: float = None
    ):
        """
        记录单个epoch的训练数据
        
        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            val_loss: 验证损失（可选）
            learning_rate: 学习率（可选）
            epoch_time: epoch训练时间（可选）
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        if epoch_time is not None:
            self.training_times.append(epoch_time)
        
        # 记录硬件数据
        try:
            hw_data = self.hardware_monitor.get_current_stats()
            if hw_data:
                hw_data['epoch'] = epoch
                hw_data['timestamp'] = time.time()
                self.hardware_data.append(hw_data)
        except Exception as e:
            logger.debug(f"获取硬件数据失败: {e}")
        
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss_str}")
    
    def log_prediction(
        self,
        epoch: int,
        input_data: np.ndarray,
        target_data: np.ndarray,
        prediction_data: np.ndarray,
        sample_idx: int = 0
    ):
        """
        记录预测结果并生成可视化
        
        Args:
            epoch: 当前epoch
            input_data: 输入数据
            target_data: 目标数据
            prediction_data: 预测数据
            sample_idx: 样本索引
        """
        # 保存预测历史
        self.prediction_history.append({
            'epoch': epoch,
            'input': input_data.copy(),
            'target': target_data.copy(),
            'prediction': prediction_data.copy()
        })
        self.prediction_epochs.append(epoch)
        
        # 生成单样本预测可视化
        self.prediction_visualizer.visualize_single_prediction(
            input_data, target_data, prediction_data,
            epoch=epoch, sample_idx=sample_idx
        )
        
        # 每5个epoch生成一次误差分析
        if epoch % 5 == 0:
            self.prediction_visualizer.visualize_prediction_with_error(
                input_data, target_data, prediction_data,
                epoch=epoch, sample_idx=sample_idx
            )
        
        logger.info(f"Epoch {epoch}: 预测结果已记录和可视化")
    
    def log_batch_predictions(
        self,
        epoch: int,
        inputs_batch: np.ndarray,
        targets_batch: np.ndarray,
        predictions_batch: np.ndarray,
        num_samples: int = 4
    ):
        """
        记录批次预测结果
        
        Args:
            epoch: 当前epoch
            inputs_batch: 输入批次
            targets_batch: 目标批次
            predictions_batch: 预测批次
            num_samples: 要可视化的样本数量
        """
        self.prediction_visualizer.visualize_batch_predictions(
            inputs_batch, targets_batch, predictions_batch,
            epoch=epoch, num_samples=num_samples
        )
        
        logger.info(f"Epoch {epoch}: 批次预测结果已可视化")
    
    def generate_training_plots(self):
        """
        生成训练过程可视化图表
        """
        if not self.epochs:
            logger.warning("没有训练数据可供可视化")
            return
        
        # 使用TrainingVisualizer生成传统训练图表
        training_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses if self.val_losses else None,
            'learning_rates': self.learning_rates if self.learning_rates else None,
            'training_times': self.training_times if self.training_times else None,
            'hardware_data': self.hardware_data if self.hardware_data else None
        }
        
        # 生成训练图表
        try:
            report_files = self.training_visualizer.generate_complete_report(
                output_dir=str(self.output_dir),
                include_plots=True,
                include_html=True,
                include_csv=True
            )
            loss_plot_path = report_files.get('loss_plot')
            hw_plot_path = report_files.get('hardware_plot')
            dashboard_path = report_files.get('dashboard_plot')
            logger.info(f"训练图表已生成: {report_files}")
        except Exception as e:
            logger.warning(f"生成训练图表失败: {e}")
            loss_plot_path = hw_plot_path = dashboard_path = None
    
    def generate_prediction_progress(self, sample_idx: int = 0):
        """
        生成预测进度可视化
        
        Args:
            sample_idx: 要跟踪的样本索引
        """
        if len(self.prediction_history) < 2:
            logger.warning("预测历史数据不足，无法生成进度可视化")
            return
        
        # 准备预测历史数据
        history_data = []
        epochs = []
        
        for record in self.prediction_history:
            history_data.append({
                'input': record['input'],
                'target': record['target'],
                'prediction': record['prediction']
            })
            epochs.append(record['epoch'])
        
        # 生成训练进度可视化
        progress_path = self.prediction_visualizer.visualize_training_progress(
            history_data, epochs, sample_idx=sample_idx
        )
        
        logger.info(f"预测进度可视化已生成: {progress_path}")
    
    def generate_summary_report(self):
        """
        生成完整的训练摘要报告
        """
        if not self.epochs:
            logger.warning("没有训练数据可供生成报告")
            return
        
        # 生成传统训练摘要
        training_summary = self.training_visualizer.generate_stats_text(
            str(self.output_dir)
        )
        
        # 生成预测可视化摘要
        if self.prediction_history:
            last_epoch = self.prediction_history[-1]['epoch']
            prediction_summary = self.prediction_visualizer.generate_summary_report(
                last_epoch, len(self.prediction_history)
            )
        
        # 生成HTML报告
        html_report_path = self.training_visualizer.generate_html_report(
            str(self.output_dir)
        )
        
        # 生成综合摘要文档
        self._generate_comprehensive_summary()
        
        logger.info("完整训练摘要报告已生成")
    
    def _generate_comprehensive_summary(self):
        """
        生成综合摘要文档
        """
        summary_content = f"""
# 增强版可视化训练完整报告

## 📊 训练概览

- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总训练轮数**: {len(self.epochs)}
- **预测可视化次数**: {len(self.prediction_history)}
- **输出目录**: {self.output_dir}

## 📈 训练性能

### 损失变化
- **初始训练损失**: {f'{self.train_losses[0]:.6f}' if self.train_losses else 'N/A'}
- **最终训练损失**: {f'{self.train_losses[-1]:.6f}' if self.train_losses else 'N/A'}
- **最佳验证损失**: {f'{min(self.val_losses):.6f}' if self.val_losses else 'N/A'}
- **损失改善**: {f'{((self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0] * 100):.2f}%' if self.train_losses and len(self.train_losses) > 1 else 'N/A'}

### 训练效率
- **平均每轮时间**: {f'{np.mean(self.training_times):.2f}秒' if self.training_times else 'N/A'}
- **总训练时间**: {f'{sum(self.training_times):.2f}秒' if self.training_times else 'N/A'}

## 🎯 预测质量分析

### 预测可视化
- **单样本预测对比**: {len([f for f in self.predictions_dir.glob('single_samples/*.png')])} 个文件
- **批次预测对比**: {len([f for f in self.predictions_dir.glob('batch_comparisons/*.png')])} 个文件
- **误差分析图表**: {len([f for f in self.predictions_dir.glob('error_analysis/*.png')])} 个文件
- **训练进度跟踪**: {len([f for f in self.predictions_dir.glob('time_series/*.png')])} 个文件

## 📁 生成的文件结构

```
{self.output_dir.name}/
├── plots/                    # 传统训练可视化
│   ├── training_curves.png
│   ├── hardware_usage.png
│   └── training_times.png
├── predictions/              # 预测结果可视化
│   ├── single_samples/       # 单样本预测对比
│   ├── batch_comparisons/    # 批次预测对比
│   ├── error_analysis/       # 误差分析
│   └── time_series/          # 训练进度跟踪
├── logs/                     # 训练日志
│   ├── training_log.txt
│   └── training_summary.txt
└── hardware_logs/            # 硬件监控数据
    ├── training_report.html
    └── training_stats.txt
```

## 🔧 技术特性

### 实现的功能
- ✅ 实时训练指标监控
- ✅ 固定轮数预测可视化
- ✅ 详细误差分析
- ✅ 批次预测对比
- ✅ 训练进度跟踪
- ✅ 硬件性能监控
- ✅ 自动报告生成

### 可视化类型
1. **训练监控**: 损失曲线、学习率变化、训练时间
2. **预测对比**: 输入-目标-预测三联图
3. **误差分析**: 绝对误差、相对误差、统计分布
4. **进度跟踪**: 多轮次预测质量变化
5. **硬件监控**: CPU、内存、GPU使用率

## 📋 使用建议

1. **训练监控**: 实时查看训练进度和硬件使用情况
2. **质量评估**: 通过预测可视化评估模型性能
3. **问题诊断**: 利用误差分析发现模型问题
4. **参数调优**: 根据可视化结果调整训练参数

---

**生成工具**: EnhancedTrainingLogger v1.0
**项目**: VIVTransformer 压力场重建
**集成模块**: TrainingVisualizer + PredictionVisualizer + HardwareMonitor
        """
        
        summary_path = self.output_dir / "comprehensive_training_report.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        logger.info(f"综合训练报告已生成: {summary_path}")


def generate_sample_data(batch_size: int = 8, input_size: int = 400, output_size: int = 40000):
    """
    生成模拟训练数据
    
    Args:
        batch_size: 批次大小
        input_size: 输入数据大小
        output_size: 输出数据大小
    
    Returns:
        输入和目标数据
    """
    # 生成模拟的压力场数据
    inputs = torch.randn(batch_size, input_size)
    targets = torch.randn(batch_size, output_size)
    
    return inputs, targets


def enhanced_training_demo(num_epochs: int = 20, output_dir: str = "enhanced_training_output"):
    """
    增强版训练演示
    
    Args:
        num_epochs: 训练轮数
        output_dir: 输出目录
    """
    logger.info("开始增强版可视化训练演示...")
    
    # 初始化模型和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 初始化增强版训练日志记录器
    training_logger = EnhancedTrainingLogger(output_dir)
    
    # 开始硬件监控
    training_logger.start_hardware_monitoring()
    
    try:
        # 训练循环
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 生成训练数据
            inputs, targets = generate_sample_data()
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, targets)
            
            # 反向传播
            train_loss.backward()
            optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = generate_sample_data(batch_size=4)
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录训练数据
            training_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss.item(),
                val_loss=val_loss.item(),
                learning_rate=optimizer.param_groups[0]['lr'],
                epoch_time=epoch_time
            )
            
            # 记录预测结果（每2个epoch记录一次）
            if epoch % 2 == 0:
                # 转换为numpy数组用于可视化
                input_np = val_inputs[0].cpu().numpy()
                target_np = val_targets[0].cpu().numpy()
                prediction_np = val_outputs[0].cpu().numpy()
                
                training_logger.log_prediction(
                    epoch=epoch,
                    input_data=input_np,
                    target_data=target_np,
                    prediction_data=prediction_np,
                    sample_idx=0
                )
            
            # 记录批次预测（每5个epoch记录一次）
            if epoch % 5 == 0:
                inputs_np = val_inputs.cpu().numpy()
                targets_np = val_targets.cpu().numpy()
                predictions_np = val_outputs.cpu().numpy()
                
                training_logger.log_batch_predictions(
                    epoch=epoch,
                    inputs_batch=inputs_np,
                    targets_batch=targets_np,
                    predictions_batch=predictions_np,
                    num_samples=4
                )
        
        # 生成所有可视化图表
        logger.info("生成训练过程可视化...")
        training_logger.generate_training_plots()
        
        logger.info("生成预测进度可视化...")
        training_logger.generate_prediction_progress(sample_idx=0)
        
        logger.info("生成完整摘要报告...")
        training_logger.generate_summary_report()
        
    finally:
        # 停止硬件监控
        training_logger.stop_hardware_monitoring()
    
    logger.info(f"增强版训练演示完成！所有结果已保存到: {training_logger.output_dir}")
    return training_logger.output_dir


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='增强版可视化训练脚本')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--output-dir', type=str, default='enhanced_training_output', 
                       help='输出目录')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    
    args = parser.parse_args()
    
    if args.demo:
        enhanced_training_demo(args.epochs, args.output_dir)
    else:
        logger.info("请使用 --demo 参数运行演示")
        logger.info("或在代码中调用 enhanced_training_demo() 函数")


if __name__ == '__main__':
    main()