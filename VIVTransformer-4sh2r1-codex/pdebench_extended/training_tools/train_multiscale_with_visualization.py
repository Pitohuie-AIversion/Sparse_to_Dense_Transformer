#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多尺度VIVTransformer训练脚本（带可视化）
使用32×32输入分辨率，5个epoch训练，完整可视化监控
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import h5py
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'multiscale'))
sys.path.insert(0, str(project_root / 'mymodels'))

# 导入多尺度模块
from multiscale.data.multiscale_adapter import MultiScaleDataset

# 导入模型（简化版VIVTransformer）
class SimpleVIVTransformer(nn.Module):
    """简化版VIVTransformer用于多尺度训练"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_encoding.expand(batch_size, seq_len, -1)
        
        # Transformer编码
        x = self.transformer(x)  # [batch, seq_len, hidden_dim]
        
        # 输出投影
        x = self.output_projection(x)  # [batch, seq_len, output_dim]
        
        return x

class MultiScaleTrainer:
    """多尺度训练器（带可视化）"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        # 初始化数据集
        self.setup_datasets()
        
        # 初始化模型
        self.setup_model()
        
        # 初始化优化器
        self.setup_optimizer()
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        self.logger.info(f"多尺度训练器初始化完成，设备: {self.device}")
        
    def setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_datasets(self):
        """设置数据集"""
        self.logger.info("创建多尺度数据集...")
        
        # 训练数据集 - 启用中心裁剪
        self.train_dataset = MultiScaleDataset(
            data_path=self.config['data_path'],
            scale_factor=self.config['scale_factor'],
            pde_type='darcy',
            split='train',
            sequence_length=1,
            original_resolution=self.config['original_resolution'],
            normalize=True,
            downsampling_method='center_crop',  # 使用中心裁剪方法
            enable_center_crop=True,  # 启用中心裁剪
            center_crop_input_resolution=self.config.get('input_resolution', [64, 64]),  # 输入分辨率
            center_crop_output_resolution=self.config.get('output_resolution', [96, 96])  # 输出分辨率
        )
        
        # 验证数据集（使用相同配置）
        self.val_dataset = MultiScaleDataset(
            data_path=self.config['data_path'],
            scale_factor=self.config['scale_factor'],
            pde_type='darcy',
            split='train',  # 使用训练集的一部分作为验证
            sequence_length=1,
            original_resolution=self.config['original_resolution'],
            normalize=True,
            downsampling_method='center_crop',  # 使用中心裁剪方法
            enable_center_crop=True,  # 启用中心裁剪
            center_crop_input_resolution=self.config.get('input_resolution', [64, 64]),  # 输入分辨率
            center_crop_output_resolution=self.config.get('output_resolution', [96, 96])  # 输出分辨率
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,  # Windows兼容性
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # 验证集使用训练集的后20%
        val_size = len(self.val_dataset) // 5
        val_indices = list(range(len(self.val_dataset) - val_size, len(self.val_dataset)))
        val_subset = torch.utils.data.Subset(self.val_dataset, val_indices)
        
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.logger.info(f"训练样本数: {len(self.train_dataset)}")
        self.logger.info(f"验证样本数: {len(val_subset)}")
        
    def setup_model(self):
        """设置模型"""
        # 获取数据维度
        sample_input, sample_target, _ = self.train_dataset[0]
        input_dim = sample_input.shape[-1]  # 最后一维是特征维度
        output_dim = sample_target.shape[-1]
        
        self.logger.info(f"输入维度: {input_dim}, 输出维度: {output_dim}")
        
        # 创建模型
        self.model = SimpleVIVTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads']
        ).to(self.device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型参数总数: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}")
        
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Train]",
            leave=False
        )
        
        for batch_idx, (inputs, targets, _) in enumerate(progress_bar):
            inputs = inputs.to(self.device)  # [batch, seq_len, input_dim]
            targets = targets.to(self.device)  # [batch, seq_len, output_dim]
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss/num_batches:.6f}'
            })
            
            # 记录到TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            
            # 每100个batch可视化一次
            if batch_idx % 100 == 0:
                self.visualize_predictions(inputs, targets, outputs, epoch, batch_idx)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Val]",
                leave=False
            )
            
            for inputs, targets, _ in progress_bar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Val Loss': f'{loss.item():.6f}',
                    'Avg Val Loss': f'{total_loss/num_batches:.6f}'
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def visualize_predictions(self, inputs, targets, outputs, epoch, batch_idx):
        """可视化预测结果"""
        if batch_idx % 200 != 0:  # 减少可视化频率
            return
            
        # 取第一个样本进行可视化
        input_sample = inputs[0, 0].cpu().numpy()  # [input_dim]
        target_sample = targets[0, 0].cpu().numpy()  # [output_dim]
        output_sample = outputs[0, 0].detach().cpu().numpy()  # [output_dim]
        
        # 重塑为2D图像
        input_size = int(np.sqrt(len(input_sample)))
        target_size = int(np.sqrt(len(target_sample)))
        
        input_img = input_sample.reshape(input_size, input_size)
        target_img = target_sample.reshape(target_size, target_size)
        output_img = output_sample.reshape(target_size, target_size)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # 输入（低分辨率）
        im1 = axes[0].imshow(input_img, cmap='viridis')
        axes[0].set_title(f'Input ({input_size}×{input_size})')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # 目标（高分辨率）
        im2 = axes[1].imshow(target_img, cmap='viridis')
        axes[1].set_title(f'Target ({target_size}×{target_size})')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # 预测（高分辨率）
        im3 = axes[2].imshow(output_img, cmap='viridis')
        axes[2].set_title(f'Prediction ({target_size}×{target_size})')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        # 误差图
        error_img = np.abs(target_img - output_img)
        im4 = axes[3].imshow(error_img, cmap='Reds')
        axes[3].set_title('Absolute Error')
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3])
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.output_dir / 'visualizations' / f'epoch_{epoch+1}'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'batch_{batch_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 记录到TensorBoard
        self.writer.add_figure(
            f'Predictions/Epoch_{epoch+1}_Batch_{batch_idx}',
            fig,
            global_step=epoch * 1000 + batch_idx
        )
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[1].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 记录到TensorBoard
        self.writer.add_figure('Training/Curves', fig, global_step=len(self.train_losses))
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_checkpoint.pth')
            self.logger.info(f"保存最佳模型检查点 (Epoch {epoch+1})")
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始多尺度训练...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录统计信息
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # 记录到TensorBoard
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # 计算时间
            epoch_time = time.time() - start_time
            
            # 打印统计信息
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.2e}, Time: {epoch_time:.2f}s"
            )
            
            # 保存检查点
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            
            # 绘制训练曲线
            if (epoch + 1) % 1 == 0:  # 每个epoch都绘制
                self.plot_training_curves()
        
        self.logger.info("训练完成！")
        self.logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 关闭TensorBoard writer
        self.writer.close()
        
        # 保存最终统计信息
        stats = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': best_val_loss,
            'config': self.config
        }
        
        with open(self.output_dir / 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    """主函数"""
    # 配置参数
    config = {
        # 数据配置
        'data_path': 'PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5',
        'original_resolution': [128, 128],  # 原始分辨率
        'scale_factor': 4,  # 缩放因子（保持兼容性，但实际使用中心裁剪）
        
        # 中心裁剪配置 - 高分辨率输入输出
        'input_resolution': [96, 96],   # 输入分辨率：从128×128中心裁剪96×96
        'output_resolution': [112, 112], # 输出分辨率：从128×128中心裁剪112×112
        
        # 训练配置
        'num_epochs': 5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        
        # 模型配置
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        
        # 输出配置
        'output_dir': f'./multiscale_training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("=" * 80)
    print("多尺度VIVTransformer训练（带可视化）- 中心裁剪模式")
    print("=" * 80)
    print(f"原始数据分辨率: {config['original_resolution'][0]}×{config['original_resolution'][1]}")
    print(f"输入分辨率（中心裁剪）: {config['input_resolution'][0]}×{config['input_resolution'][1]}")
    print(f"输出分辨率（中心裁剪）: {config['output_resolution'][0]}×{config['output_resolution'][1]}")
    print(f"训练轮数: {config['num_epochs']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"输出目录: {config['output_dir']}")
    print("=" * 80)
    
    # 创建训练器
    trainer = MultiScaleTrainer(config)
    
    # 开始训练
    try:
        trainer.train()
        print("\n🎉 训练成功完成！")
        print(f"结果保存在: {config['output_dir']}")
        print("可以使用以下命令启动TensorBoard查看训练过程:")
        print(f"tensorboard --logdir {config['output_dir']}/tensorboard")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()