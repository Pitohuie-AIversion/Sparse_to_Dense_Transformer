#!/usr/bin/env python3
"""
简化版可视化训练脚本

专为无GUI环境优化的训练脚本，提供：
- 控制台实时进度显示
- 训练指标记录和保存
- 简单的图表生成
- 完整的训练日志
"""

import os
import sys
import yaml
import logging
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# 设置matplotlib后端为非交互式
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTrainingLogger:
    """
    简单的训练日志记录器
    """
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 训练数据存储
        self.metrics = {
            'epochs': [],
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'gpu_usage': [],
            'memory_usage': [],
            'timestamps': []
        }
        
        # 创建日志文件
        self.log_file = self.save_dir / 'training_log.txt'
        self.metrics_file = self.save_dir / 'training_metrics.json'
        
        # 初始化日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  lr: float, gpu_usage: float = 0, memory_usage: float = 0):
        """记录epoch数据"""
        
        # 存储数据
        self.metrics['epochs'].append(epoch)
        self.metrics['train_losses'].append(train_loss)
        self.metrics['val_losses'].append(val_loss)
        self.metrics['learning_rates'].append(lr)
        self.metrics['gpu_usage'].append(gpu_usage)
        self.metrics['memory_usage'].append(memory_usage)
        self.metrics['timestamps'].append(time.time())
        
        # 写入日志文件
        log_entry = (
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {lr:.2e} | "
            f"GPU: {gpu_usage:5.1f}% | "
            f"Mem: {memory_usage:5.1f}%\n"
        )
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # 控制台输出
        print(log_entry.strip())
        
        # 保存指标到JSON
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
    
    def generate_plots(self):
        """生成训练图表"""
        if not self.metrics['epochs']:
            logger.warning("没有训练数据，无法生成图表")
            return
        
        # 创建图表目录
        plots_dir = self.save_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. 损失曲线
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['epochs'], self.metrics['train_losses'], 'b-', label='训练损失', linewidth=2)
        plt.plot(self.metrics['epochs'], self.metrics['val_losses'], 'r-', label='验证损失', linewidth=2)
        plt.title('训练损失曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 2. 学习率变化
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['epochs'], self.metrics['learning_rates'], 'g-', linewidth=2, marker='o')
        plt.title('学习率变化', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 3. GPU使用率
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['epochs'], self.metrics['gpu_usage'], 'orange', linewidth=2)
        plt.fill_between(self.metrics['epochs'], self.metrics['gpu_usage'], alpha=0.3, color='orange')
        plt.title('GPU使用率', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('GPU使用率 (%)')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        
        # 4. 内存使用率
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['epochs'], self.metrics['memory_usage'], 'purple', linewidth=2)
        plt.fill_between(self.metrics['epochs'], self.metrics['memory_usage'], alpha=0.3, color='purple')
        plt.title('内存使用率', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('内存使用率 (%)')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = plots_dir / 'training_overview.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练图表已保存到: {plot_path}")
        
        # 单独保存损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['epochs'], self.metrics['train_losses'], 'b-', label='训练损失', linewidth=2)
        plt.plot(self.metrics['epochs'], self.metrics['val_losses'], 'r-', label='验证损失', linewidth=2)
        plt.title('训练损失曲线', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        loss_plot_path = plots_dir / 'loss_curves.png'
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"损失曲线已保存到: {loss_plot_path}")
    
    def generate_summary_report(self):
        """生成训练摘要报告"""
        if not self.metrics['epochs']:
            return
        
        # 计算统计信息
        final_train_loss = self.metrics['train_losses'][-1]
        final_val_loss = self.metrics['val_losses'][-1]
        best_train_loss = min(self.metrics['train_losses'])
        best_val_loss = min(self.metrics['val_losses'])
        
        avg_gpu_usage = np.mean(self.metrics['gpu_usage']) if self.metrics['gpu_usage'] else 0
        max_gpu_usage = max(self.metrics['gpu_usage']) if self.metrics['gpu_usage'] else 0
        avg_memory_usage = np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
        max_memory_usage = max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
        
        # 生成报告
        report = f"""
训练摘要报告
{'=' * 60}
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

训练配置:
- 总Epoch数: {len(self.metrics['epochs'])}
- 最终训练损失: {final_train_loss:.6f}
- 最终验证损失: {final_val_loss:.6f}
- 最佳训练损失: {best_train_loss:.6f}
- 最佳验证损失: {best_val_loss:.6f}

硬件使用情况:
- 平均GPU使用率: {avg_gpu_usage:.1f}%
- 最大GPU使用率: {max_gpu_usage:.1f}%
- 平均内存使用率: {avg_memory_usage:.1f}%
- 最大内存使用率: {max_memory_usage:.1f}%

训练改善:
- 训练损失改善: {(self.metrics['train_losses'][0] - final_train_loss) / self.metrics['train_losses'][0] * 100:.2f}%
- 验证损失改善: {(self.metrics['val_losses'][0] - final_val_loss) / self.metrics['val_losses'][0] * 100:.2f}%

{'=' * 60}
"""
        
        # 保存报告
        report_path = self.save_dir / 'training_summary.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"训练摘要报告已保存到: {report_path}")
        print(report)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_simple_model(config: Dict[str, Any]) -> nn.Module:
    """创建简单的演示模型"""
    input_dim = config['data']['input_dim']
    output_dim = config['data']['output_dim']
    hidden_dim = config['model']['transformer']['d_model']
    
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim // 2, output_dim)
    )
    
    return model


def get_hardware_usage():
    """获取硬件使用情况"""
    gpu_usage = 0
    memory_usage = 0
    
    if torch.cuda.is_available():
        try:
            # 尝试获取GPU使用率
            gpu_usage = torch.cuda.utilization()
            memory_usage = torch.cuda.memory_percent()
        except Exception:
            # 如果无法获取，使用简单的内存监控
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            if cached > 0:
                memory_usage = (allocated / cached) * 100
            else:
                memory_usage = 0
            gpu_usage = min(50 + memory_usage / 2, 100)  # 估算GPU使用率
    
    return gpu_usage, memory_usage


def simple_visual_training(config_path: str):
    """
    简化版可视化训练主函数
    """
    
    # 加载配置
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 初始化训练日志记录器
    training_logger = SimpleTrainingLogger(output_dir)
    
    # 创建模型
    model = create_simple_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数量: {total_params:,}")
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['num_epochs']
    )
    
    # 训练参数
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    input_dim = config['data']['input_dim']
    output_dim = config['data']['output_dim']
    
    logger.info(f"开始训练，共 {num_epochs} 个epoch")
    print("\n" + "=" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>10} | {'LR':>10} | {'GPU':>5} | {'Mem':>5}")
    print("=" * 80)
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = 10  # 模拟10个batch
        
        # 使用tqdm显示进度
        with tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
            for batch_idx in pbar:
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
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
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
        
        # 获取当前学习率
        current_lr = scheduler.get_last_lr()[0]
        
        # 获取硬件使用情况
        gpu_usage, memory_usage = get_hardware_usage()
        
        # 记录epoch数据
        training_logger.log_epoch(
            epoch + 1, train_loss, val_loss, current_lr, gpu_usage, memory_usage
        )
        
        # 更新学习率
        scheduler.step()
    
    print("=" * 80)
    logger.info("训练完成！")
    
    # 生成图表和报告
    logger.info("生成训练图表...")
    training_logger.generate_plots()
    
    logger.info("生成训练摘要报告...")
    training_logger.generate_summary_report()
    
    logger.info(f"所有结果已保存到: {output_dir}")
    
    return output_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简化版可视化训练')
    parser.add_argument(
        '--config', 
        type=str, 
        default='demo_config.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return 1
    
    print("\n" + "=" * 70)
    print("🎯 简化版可视化训练")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)
    
    try:
        output_dir = simple_visual_training(str(config_path))
        
        print("\n" + "=" * 70)
        print("✅ 训练成功完成！")
        print("=" * 70)
        print(f"结果保存在: {output_dir}")
        print(f"查看图表: {output_dir / 'plots'}")
        print(f"查看日志: {output_dir / 'training_log.txt'}")
        print(f"查看摘要: {output_dir / 'training_summary.txt'}")
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
        return 0
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)