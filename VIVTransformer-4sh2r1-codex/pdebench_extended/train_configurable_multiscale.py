#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可配置多尺度VIVTransformer训练脚本
支持不同缩放因子的训练，适用于服务器部署

作者: AI Assistant
日期: 2025-01-25
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'multiscale'))
sys.path.append(str(Path(__file__).parent / 'multiscale' / 'data'))

# 尝试多种导入方式来解决服务器端导入问题
try:
    from multiscale.data.multiscale_adapter import create_multiscale_loaders
except ImportError:
    try:
        # 尝试直接导入
        import multiscale_adapter
        create_multiscale_loaders = multiscale_adapter.create_multiscale_loaders
    except ImportError:
        try:
            # 尝试从当前目录导入
            sys.path.insert(0, str(Path(__file__).parent / 'multiscale' / 'data'))
            from multiscale_adapter import create_multiscale_loaders
        except ImportError as e:
            print(f"导入错误: {e}")
            print("请确保multiscale_adapter.py文件存在于multiscale/data/目录下")
            raise

from mymodels.transformer import TransformerFlowReconstructionModel


class ConfigurableMultiScaleTrainer:
    """可配置的多尺度训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # 初始化数据加载器
        self.setup_data_loaders()
        
        # 初始化模型
        self.setup_model()
        
        # 初始化优化器和调度器
        self.setup_optimizer()
        
        self.logger.info(f"训练器初始化完成，使用设备: {self.device}")
        self.logger.info(f"缩放因子: {config['data']['scale_factor']}")
        self.logger.info(f"输入分辨率: {config['data'].get('input_resolution', 'Auto')}")
        self.logger.info(f"输出分辨率: {config['data'].get('output_resolution', 'Auto')}")
    
    def setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_data_loaders(self):
        """设置数据加载器"""
        self.logger.info("创建数据加载器...")
        
        data_config = self.config['data']
        
        # 创建多尺度数据加载器
        loaders_info = create_multiscale_loaders(
            data_path=data_config.get('path', data_config.get('data_path', 'PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5')),
            scale_factor=data_config['scale_factor'],
            pde_type=data_config.get('pde_type', 'darcy_flow'),
            batch_size=data_config['batch_size'],
            sequence_length=data_config.get('sequence_length', 1),
            original_resolution=data_config['original_resolution'],
            normalize=data_config.get('normalize', True),
            downsampling_method=data_config.get('downsampling_method', 'average'),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            enable_center_crop=data_config.get('enable_center_crop', False),
            center_crop_input_resolution=data_config.get('center_crop_input_resolution'),
            center_crop_output_resolution=data_config.get('center_crop_output_resolution')
        )
        
        self.train_loader = loaders_info['train']
        self.val_loader = loaders_info['val']
        self.test_loader = loaders_info['test']
        self.data_info = loaders_info['info']
        
        # 获取数据维度
        sample_input, sample_target, _ = next(iter(self.train_loader))
        self.input_dim = sample_input.shape[-1]
        self.output_dim = sample_target.shape[-1]
        
        self.logger.info(f"数据加载器创建完成")
        self.logger.info(f"训练集大小: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证集大小: {len(self.val_loader.dataset)}")
        self.logger.info(f"输入维度: {self.input_dim}")
        self.logger.info(f"输出维度: {self.output_dim}")
    
    def setup_model(self):
        """设置模型"""
        model_config = self.config['model']
        
        self.model = TransformerFlowReconstructionModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            d_model=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            max_time_steps=100,
            attention_type='relative',
            seq_len=49
        ).to(self.device)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型创建完成")
        self.logger.info(f"总参数数量: {total_params:,}")
        self.logger.info(f"可训练参数数量: {trainable_params:,}")
    
    def setup_optimizer(self):
        """设置优化器和调度器"""
        train_config = self.config['training']
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config.get('weight_decay', 1e-5)
        )
        
        num_epochs = train_config.get('epochs', train_config.get('num_epochs', 10))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=train_config.get('min_learning_rate', 1e-6)
        )
        
        self.logger.info(f"优化器和调度器设置完成")
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['training'].get('epochs', self.config['training'].get('num_epochs', 'N/A'))} [Train]",
            leave=False
        )
        
        for batch_idx, (inputs, targets, _) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            # 创建时间步张量
            batch_size = inputs.shape[0]
            time_steps = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            outputs = self.model(inputs, time_steps)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            if self.config['training'].get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip_norm']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss/num_batches:.6f}'
            })
            
            # 记录到TensorBoard
            if batch_idx % 100 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            
            # 可视化预测结果
            if batch_idx % 500 == 0:
                self.visualize_predictions(inputs, targets, outputs, epoch, batch_idx)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch: int) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch+1}/{self.config['training'].get('epochs', self.config['training'].get('num_epochs', 'N/A'))} [Val]",
                leave=False
            )
            
            for inputs, targets, _ in progress_bar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 创建时间步张量
                batch_size = inputs.shape[0]
                time_steps = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
                outputs = self.model(inputs, time_steps)
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
        if batch_idx % 1000 != 0:  # 减少可视化频率
            return
        
        # 取第一个样本进行可视化
        input_sample = inputs[0, 0].cpu().numpy()
        target_sample = targets[0, 0].cpu().numpy()
        output_sample = outputs[0, 0].detach().cpu().numpy()
        
        # 重塑为2D图像
        input_size = int(np.sqrt(len(input_sample)))
        target_size = int(np.sqrt(len(target_sample)))
        
        input_img = input_sample.reshape(input_size, input_size)
        target_img = target_sample.reshape(target_size, target_size)
        output_img = output_sample.reshape(target_size, target_size)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # 输入
        im1 = axes[0].imshow(input_img, cmap='viridis')
        axes[0].set_title(f'Input ({input_size}×{input_size})')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # 目标
        im2 = axes[1].imshow(target_img, cmap='viridis')
        axes[1].set_title(f'Target ({target_size}×{target_size})')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # 预测
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
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config,
            'data_info': self.data_info
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
        
        num_epochs = self.config['training'].get('epochs', self.config['training'].get('num_epochs', 10))
        for epoch in range(num_epochs):
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
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {current_lr:.2e}, Time: {epoch_time:.2f}s"
            )
            
            # 保存检查点
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
        
        self.logger.info("训练完成！")
        self.logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 关闭TensorBoard writer
        self.writer.close()
        
        # 保存最终统计信息
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        stats = {
            'train_losses': [float(loss) for loss in self.train_losses],
            'val_losses': [float(loss) for loss in self.val_losses],
            'learning_rates': [float(lr) for lr in self.learning_rates],
            'best_val_loss': float(best_val_loss),
            'config': self.config,
            'data_info': convert_numpy_types(self.data_info)
        }
        
        with open(self.output_dir / 'training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)


def create_config(scale_factor: int = 4, 
                 input_resolution: Optional[List[int]] = None,
                 output_resolution: Optional[List[int]] = None,
                 num_epochs: int = 10,
                 batch_size: int = 8) -> Dict[str, Any]:
    """创建训练配置"""
    
    # 默认分辨率设置
    if input_resolution is None:
        input_resolution = [96, 96]
    if output_resolution is None:
        output_resolution = [112, 112]
    
    config = {
        # 数据配置
        'data': {
            'data_path': 'PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5',
            'original_resolution': [128, 128],
            'scale_factor': scale_factor,
            'pde_type': 'darcy_flow',
            'sequence_length': 1,
            'normalize': True,
            'downsampling_method': 'average',
            'batch_size': batch_size,
            'num_workers': 4,
            'pin_memory': True,
            
            # 中心裁剪配置
            'enable_center_crop': True,
            'center_crop_input_resolution': input_resolution,
            'center_crop_output_resolution': output_resolution
        },
        
        # 模型配置
        'model': {
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.1,
            'activation': 'gelu'
        },
        
        # 训练配置
        'training': {
            'num_epochs': num_epochs,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'min_learning_rate': 1e-6,
            'grad_clip_norm': 1.0
        },
        
        # 输出配置
        'output_dir': f'./configurable_multiscale_results_sf{scale_factor}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='可配置多尺度VIVTransformer训练')
    parser.add_argument('--scale_factor', type=int, default=4, help='缩放因子')
    parser.add_argument('--input_resolution', type=int, nargs=2, default=[96, 96], help='输入分辨率 [H, W]')
    parser.add_argument('--output_resolution', type=int, nargs=2, default=[112, 112], help='输出分辨率 [H, W]')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--config_file', type=str, help='配置文件路径（可选）')
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config_file and os.path.exists(args.config_file):
        import yaml
        with open(args.config_file, 'r', encoding='utf-8') as f:
            if args.config_file.endswith('.yaml') or args.config_file.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        print(f"从配置文件加载: {args.config_file}")
        
        # 确保有output_dir字段
        if 'output_dir' not in config:
            config['output_dir'] = f'./configurable_multiscale_results_sf{config["data"]["scale_factor"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # 命令行参数覆盖配置文件设置
        if hasattr(args, 'scale_factor') and args.scale_factor != 4:  # 4是默认值
            config['data']['scale_factor'] = args.scale_factor
        if hasattr(args, 'input_resolution') and args.input_resolution != [96, 96]:  # [96, 96]是默认值
            config['data']['center_crop_input_resolution'] = args.input_resolution
        if hasattr(args, 'output_resolution') and args.output_resolution != [112, 112]:  # [112, 112]是默认值
            config['data']['center_crop_output_resolution'] = args.output_resolution
        if hasattr(args, 'num_epochs') and args.num_epochs != 10:  # 10是默认值
            if 'epochs' in config['training']:
                config['training']['epochs'] = args.num_epochs
            else:
                config['training']['num_epochs'] = args.num_epochs
        if hasattr(args, 'batch_size') and args.batch_size != 8:  # 8是默认值
            config['data']['batch_size'] = args.batch_size
            
        print("命令行参数已覆盖配置文件中的相应设置")
    else:
        config = create_config(
            scale_factor=args.scale_factor,
            input_resolution=args.input_resolution,
            output_resolution=args.output_resolution,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size
        )
    
    print("=" * 80)
    print("可配置多尺度VIVTransformer训练")
    print("=" * 80)
    print(f"缩放因子: {config['data']['scale_factor']}")
    print(f"原始数据分辨率: {config['data']['original_resolution'][0]}×{config['data']['original_resolution'][1]}")
    print(f"输入分辨率（中心裁剪）: {config['data']['center_crop_input_resolution'][0]}×{config['data']['center_crop_input_resolution'][1]}")
    print(f"输出分辨率（中心裁剪）: {config['data']['center_crop_output_resolution'][0]}×{config['data']['center_crop_output_resolution'][1]}")
    print(f"训练轮数: {config['training'].get('epochs', config['training'].get('num_epochs', 'N/A'))}")
    print(f"批次大小: {config['data']['batch_size']}")
    print(f"输出目录: {config['output_dir']}")
    print("=" * 80)
    
    # 保存配置
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 创建训练器
    trainer = ConfigurableMultiScaleTrainer(config)
    
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