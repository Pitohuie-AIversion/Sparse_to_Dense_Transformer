#!/usr/bin/env python3
"""
带可视化功能的压力场重建训练脚本

提供实时训练监控和可视化功能，包括：
- 实时损失曲线显示
- 训练进度可视化
- 硬件监控图表
- 预测结果对比
- TensorBoard集成
"""

import os
import sys
import yaml
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from data.pressure_field_adapter import (
    create_pressure_field_datasets,
    analyze_pressure_field_data
)
from utils.hardware_monitor import HardwareMonitor
from utils.training_visualizer import TrainingVisualizer
from utils.visualization import (
    plot_comparison_figure,
    plot_difference_figure,
    plot_losses
)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealTimeVisualizer:
    """
    实时训练可视化器
    
    提供训练过程中的实时图表更新
    """
    
    def __init__(self, save_dir: Path, enable_realtime: bool = True):
        self.save_dir = save_dir
        self.enable_realtime = enable_realtime
        
        # 训练数据存储
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.epochs = []
        self.learning_rates = []
        
        # 硬件数据存储
        self.gpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        
        if self.enable_realtime:
            self._setup_realtime_plots()
    
    def _setup_realtime_plots(self):
        """设置实时绘图"""
        plt.ion()  # 开启交互模式
        
        # 创建图形和子图
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = GridSpec(3, 3, figure=self.fig)
        
        # 损失曲线
        self.ax_loss = self.fig.add_subplot(self.gs[0, :])
        self.ax_loss.set_title('训练损失曲线', fontsize=14, fontweight='bold')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # GPU使用率
        self.ax_gpu = self.fig.add_subplot(self.gs[1, 0])
        self.ax_gpu.set_title('GPU使用率', fontsize=12)
        self.ax_gpu.set_ylabel('使用率 (%)')
        self.ax_gpu.set_ylim(0, 100)
        
        # 内存使用率
        self.ax_memory = self.fig.add_subplot(self.gs[1, 1])
        self.ax_memory.set_title('内存使用率', fontsize=12)
        self.ax_memory.set_ylabel('使用率 (%)')
        self.ax_memory.set_ylim(0, 100)
        
        # 学习率变化
        self.ax_lr = self.fig.add_subplot(self.gs[1, 2])
        self.ax_lr.set_title('学习率变化', fontsize=12)
        self.ax_lr.set_ylabel('学习率')
        
        # 预测结果对比（占用底部两个位置）
        self.ax_pred = self.fig.add_subplot(self.gs[2, :])
        self.ax_pred.set_title('预测结果对比', fontsize=12)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_loss_data(self, epoch: int, train_loss: float, val_loss: float = None, test_loss: float = None):
        """更新损失数据"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if test_loss is not None:
            self.test_losses.append(test_loss)
        
        if self.enable_realtime:
            self._update_loss_plot()
    
    def update_hardware_data(self, gpu_usage: float, memory_usage: float):
        """更新硬件数据"""
        self.gpu_usage.append(gpu_usage)
        self.memory_usage.append(memory_usage)
        self.timestamps.append(time.time())
        
        # 只保留最近100个数据点
        if len(self.gpu_usage) > 100:
            self.gpu_usage = self.gpu_usage[-100:]
            self.memory_usage = self.memory_usage[-100:]
            self.timestamps = self.timestamps[-100:]
        
        if self.enable_realtime:
            self._update_hardware_plots()
    
    def update_learning_rate(self, lr: float):
        """更新学习率"""
        self.learning_rates.append(lr)
        
        if self.enable_realtime:
            self._update_lr_plot()
    
    def update_prediction_comparison(self, input_data: np.ndarray, target: np.ndarray, prediction: np.ndarray):
        """更新预测结果对比"""
        if self.enable_realtime:
            self._update_prediction_plot(input_data, target, prediction)
    
    def _update_loss_plot(self):
        """更新损失曲线"""
        self.ax_loss.clear()
        self.ax_loss.set_title('训练损失曲线', fontsize=14, fontweight='bold')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        if self.train_losses:
            self.ax_loss.plot(self.epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        if self.val_losses:
            self.ax_loss.plot(self.epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        if self.test_losses:
            self.ax_loss.plot(self.epochs, self.test_losses, 'g-', label='测试损失', linewidth=2)
        
        self.ax_loss.legend()
        self.ax_loss.set_yscale('log')
    
    def _update_hardware_plots(self):
        """更新硬件监控图表"""
        if not self.gpu_usage:
            return
        
        # GPU使用率
        self.ax_gpu.clear()
        self.ax_gpu.set_title('GPU使用率', fontsize=12)
        self.ax_gpu.set_ylabel('使用率 (%)')
        self.ax_gpu.set_ylim(0, 100)
        self.ax_gpu.plot(range(len(self.gpu_usage)), self.gpu_usage, 'g-', linewidth=2)
        self.ax_gpu.fill_between(range(len(self.gpu_usage)), self.gpu_usage, alpha=0.3, color='green')
        
        # 内存使用率
        self.ax_memory.clear()
        self.ax_memory.set_title('内存使用率', fontsize=12)
        self.ax_memory.set_ylabel('使用率 (%)')
        self.ax_memory.set_ylim(0, 100)
        self.ax_memory.plot(range(len(self.memory_usage)), self.memory_usage, 'orange', linewidth=2)
        self.ax_memory.fill_between(range(len(self.memory_usage)), self.memory_usage, alpha=0.3, color='orange')
    
    def _update_lr_plot(self):
        """更新学习率图表"""
        if not self.learning_rates:
            return
        
        self.ax_lr.clear()
        self.ax_lr.set_title('学习率变化', fontsize=12)
        self.ax_lr.set_ylabel('学习率')
        self.ax_lr.plot(self.epochs, self.learning_rates, 'purple', linewidth=2, marker='o')
        self.ax_lr.set_yscale('log')
    
    def _update_prediction_plot(self, input_data: np.ndarray, target: np.ndarray, prediction: np.ndarray):
        """更新预测结果对比"""
        self.ax_pred.clear()
        self.ax_pred.set_title('预测结果对比 (最新样本)', fontsize=12)
        
        # 重塑数据为2D图像
        if len(target.shape) == 1:
            size = int(np.sqrt(len(target)))
            target_2d = target.reshape(size, size)
            prediction_2d = prediction.reshape(size, size)
        else:
            target_2d = target
            prediction_2d = prediction
        
        # 创建子图
        self.ax_pred.axis('off')
        
        # 显示目标和预测
        im1 = self.ax_pred.imshow(np.hstack([target_2d, prediction_2d]), cmap='coolwarm')
        self.ax_pred.axvline(x=target_2d.shape[1]-0.5, color='white', linewidth=2)
        self.ax_pred.text(target_2d.shape[1]//2, -5, '目标', ha='center', fontsize=10, fontweight='bold')
        self.ax_pred.text(target_2d.shape[1] + target_2d.shape[1]//2, -5, '预测', ha='center', fontsize=10, fontweight='bold')
        
        plt.colorbar(im1, ax=self.ax_pred, fraction=0.046, pad=0.04)
    
    def save_final_plots(self):
        """保存最终的图表"""
        if self.enable_realtime:
            # 保存当前图形
            final_plot_path = self.save_dir / 'final_training_visualization.png'
            self.fig.savefig(final_plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"最终可视化图表已保存到: {final_plot_path}")
        
        # 保存单独的损失曲线
        plt.figure(figsize=(10, 6))
        if self.train_losses:
            plt.plot(self.epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        if self.test_losses:
            plt.plot(self.epochs, self.test_losses, 'g-', label='测试损失', linewidth=2)
        
        plt.title('训练损失曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        loss_plot_path = self.save_dir / 'loss_curves.png'
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"损失曲线已保存到: {loss_plot_path}")
    
    def close(self):
        """关闭可视化"""
        if self.enable_realtime:
            plt.ioff()
            plt.close(self.fig)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]) -> nn.Module:
    """创建模型"""
    # 这里应该根据配置创建实际的模型
    # 为了演示，创建一个简单的MLP
    input_dim = config['data']['input_dim']
    output_dim = config['data']['output_dim']
    hidden_dim = config['model']['transformer']['d_model']
    
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    
    return model


def train_with_visualization(
    config_path: str,
    enable_realtime_vis: bool = True,
    enable_tensorboard: bool = True
):
    """
    带可视化功能的训练主函数
    
    Args:
        config_path: 配置文件路径
        enable_realtime_vis: 是否启用实时可视化
        enable_tensorboard: 是否启用TensorBoard
    """
    
    # 加载配置
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建可视化目录
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # 初始化可视化器
    realtime_vis = RealTimeVisualizer(vis_dir, enable_realtime_vis)
    
    # 初始化TensorBoard
    if enable_tensorboard:
        tb_dir = output_dir / 'tensorboard'
        tb_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard日志目录: {tb_dir}")
        logger.info(f"启动TensorBoard: tensorboard --logdir={tb_dir} --port=6006")
    
    # 初始化硬件监控
    hardware_monitor = HardwareMonitor(str(output_dir / 'hardware_logs'))
    hardware_monitor.start_training()
    
    # 创建模型
    model = create_model(config).to(device)
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集（这里需要根据实际情况实现）
    logger.info("创建数据集...")
    # train_loader, val_loader, test_loader = create_pressure_field_datasets(config)
    
    # 为了演示，创建虚拟数据
    batch_size = config['training']['batch_size']
    input_dim = config['data']['input_dim']
    output_dim = config['data']['output_dim']
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['num_epochs']
    )
    
    # 训练循环
    num_epochs = config['training']['num_epochs']
    logger.info(f"开始训练，共 {num_epochs} 个epoch")
    
    for epoch in range(num_epochs):
        hardware_monitor.start_epoch(epoch)
        
        # 模拟训练数据
        model.train()
        train_loss = 0.0
        num_batches = 10  # 模拟10个batch
        
        # 训练阶段
        for batch_idx in range(num_batches):
            hardware_monitor.start_batch(epoch, batch_idx)
            
            # 生成模拟数据
            inputs = torch.randn(batch_size, input_dim).to(device)
            targets = torch.randn(batch_size, output_dim).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 更新硬件监控
            if torch.cuda.is_available():
                try:
                    gpu_usage = torch.cuda.utilization()
                    memory_usage = torch.cuda.memory_percent()
                except Exception:
                    # 如果pynvml不可用，使用简单的GPU内存监控
                    gpu_usage = 50  # 模拟GPU使用率
                    memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
            else:
                gpu_usage = 0
                memory_usage = 0
            
            realtime_vis.update_hardware_data(gpu_usage, memory_usage)
            
            hardware_monitor.end_batch(epoch, batch_idx, loss.item())
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(5):  # 模拟5个验证batch
                inputs = torch.randn(batch_size, input_dim).to(device)
                targets = torch.randn(batch_size, output_dim).to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        # 计算平均损失
        train_loss /= num_batches
        val_loss /= 5
        
        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # 更新可视化
        realtime_vis.update_loss_data(epoch + 1, train_loss, val_loss)
        realtime_vis.update_learning_rate(current_lr)
        
        # 更新预测结果对比（使用最后一个batch的数据）
        if epoch % 5 == 0:  # 每5个epoch更新一次预测对比
            with torch.no_grad():
                sample_input = torch.randn(1, input_dim).to(device)
                sample_target = torch.randn(1, output_dim).to(device)
                sample_pred = model(sample_input)
                
                realtime_vis.update_prediction_comparison(
                    sample_input.cpu().numpy().flatten(),
                    sample_target.cpu().numpy().flatten(),
                    sample_pred.cpu().numpy().flatten()
                )
        
        # TensorBoard记录
        if enable_tensorboard:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            if torch.cuda.is_available():
                writer.add_scalar('Hardware/GPU_Usage', gpu_usage, epoch)
                writer.add_scalar('Hardware/Memory_Usage', memory_usage, epoch)
        
        # 打印进度
        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, "
            f"LR: {current_lr:.2e}"
        )
        
        hardware_monitor.end_epoch(epoch, train_loss, val_loss)
        
        # 强制更新图表
        if enable_realtime_vis:
            plt.pause(0.01)
    
    # 训练结束
    hardware_monitor.end_training()
    
    # 保存最终可视化
    realtime_vis.save_final_plots()
    realtime_vis.close()
    
    # 生成完整的训练报告
    try:
        training_visualizer = TrainingVisualizer(str(output_dir / 'hardware_logs'))
        training_visualizer.generate_complete_report()
        logger.info(f"完整训练报告已生成到: {output_dir / 'hardware_logs' / 'visualizations'}")
    except Exception as e:
        logger.warning(f"生成训练报告失败: {e}")
    
    # 关闭TensorBoard
    if enable_tensorboard:
        writer.close()
    
    logger.info("训练完成！")
    logger.info(f"结果保存在: {output_dir}")
    if enable_tensorboard:
        logger.info(f"查看TensorBoard: tensorboard --logdir={tb_dir} --port=6006")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='带可视化功能的压力场重建训练')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/pressure_field_training.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--no-realtime-vis', 
        action='store_true',
        help='禁用实时可视化'
    )
    parser.add_argument(
        '--no-tensorboard', 
        action='store_true',
        help='禁用TensorBoard'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 启动带可视化功能的压力场重建训练")
    logger.info("=" * 60)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"实时可视化: {'禁用' if args.no_realtime_vis else '启用'}")
    logger.info(f"TensorBoard: {'禁用' if args.no_tensorboard else '启用'}")
    logger.info("=" * 60)
    
    try:
        train_with_visualization(
            str(config_path),
            enable_realtime_vis=not args.no_realtime_vis,
            enable_tensorboard=not args.no_tensorboard
        )
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    main()