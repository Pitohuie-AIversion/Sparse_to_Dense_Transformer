#!/usr/bin/env python3
"""多尺度超分辨率重构训练脚本

使用VIVTransformer进行多尺度超分辨率重构训练：
- 从低分辨率输入重构到高分辨率输出
- 支持多种缩放因子（2x, 4x, 8x等）
- 支持多种注意力机制
- 包含完整的训练、验证和测试流程
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import time
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.dataloader import get_multiscale_loaders
from mymodels.model_factory import create_model
from utils.svd10_loss import TotalLossWithSVD
from utils.visualization import create_prediction_visualization

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiScaleTrainer:
    """多尺度超分辨率重构训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = self._setup_device()
        self.setup_directories()
        self.setup_logging()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        logger.info(f"多尺度训练器初始化完成，设备: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        logger.info(f"使用设备: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU信息: {torch.cuda.get_device_name()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        return device
    
    def setup_directories(self):
        """设置输出目录"""
        self.output_dir = Path(self.config.get('output', {}).get('save_dir', 'results/multiscale'))
        self.log_dir = Path(self.config.get('logging', {}).get('log_dir', 'logs/multiscale'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.visualization_dir = self.output_dir / 'visualizations'
        
        # 创建目录
        for dir_path in [self.output_dir, self.log_dir, self.checkpoint_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"输出目录: {self.output_dir}")
    
    def setup_logging(self):
        """设置日志记录"""
        # 文件日志处理器
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加到根日志记录器
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"日志文件: {log_file}")
    
    def load_data(self) -> Tuple[Any, Any, Any, Dict]:
        """加载多尺度数据
        
        Returns:
            训练、验证、测试数据加载器和数据信息
        """
        logger.info("开始加载多尺度数据...")
        
        try:
            loaders_info = get_multiscale_loaders(self.config)
            
            train_loader = loaders_info['train']
            val_loader = loaders_info['val']
            test_loader = loaders_info['test']
            data_info = loaders_info['info']
            
            logger.info(f"数据加载完成:")
            logger.info(f"  训练集: {len(train_loader)} 批次")
            logger.info(f"  验证集: {len(val_loader)} 批次")
            logger.info(f"  测试集: {len(test_loader)} 批次")
            logger.info(f"  输入维度: {data_info['input_dim']}")
            logger.info(f"  输出维度: {data_info['output_dim']}")
            logger.info(f"  缩放因子: {data_info['scale_factor']}")
            logger.info(f"  压缩比: {data_info['compression_ratio']:.2f}")
            
            return train_loader, val_loader, test_loader, data_info
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def create_model(self, data_info: Dict) -> nn.Module:
        """创建模型
        
        Args:
            data_info: 数据信息字典
            
        Returns:
            创建的模型
        """
        logger.info("开始创建模型...")
        
        # 更新配置中的维度信息
        model_config = self.config.get('model', {}).copy()
        model_config['input_dim'] = data_info['input_dim']
        model_config['output_dim'] = data_info['output_dim']
        
        # 创建模型
        model = create_model(model_config, attention_type='standard')
        model = model.to(self.device)
        
        # 多GPU支持
        if torch.cuda.device_count() > 1 and model_config.get('use_data_parallel', True):
            model = nn.DataParallel(model)
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型创建完成:")
        logger.info(f"  总参数数: {total_params:,}")
        logger.info(f"  可训练参数数: {trainable_params:,}")
        logger.info(f"  模型大小: {total_params * 4 / 1e6:.1f} MB")
        
        return model
    
    def create_criterion(self) -> nn.Module:
        """创建损失函数
        
        Returns:
            损失函数
        """
        loss_config = self.config.get('loss', {})
        
        # 主损失函数
        primary_loss = loss_config.get('primary', {}).get('type', 'MSELoss')
        
        if primary_loss == 'MSELoss':
            criterion = nn.MSELoss()
        elif primary_loss == 'L1Loss':
            criterion = nn.L1Loss()
        else:
            # 使用TotalLossWithSVD作为默认，传入 grid 和 svd_enabled 参数
            model_cfg = self.config.get('model', {})
            grid_height = model_cfg.get('grid_height', None)
            grid_width = model_cfg.get('grid_width', None)
            svd_enabled = loss_config.get('svd_enabled', True)
            
            criterion = TotalLossWithSVD(
                grid_height=grid_height,
                grid_width=grid_width,
                svd_enabled=svd_enabled
            )
        
        criterion = criterion.to(self.device)
        
        logger.info(f"损失函数: {type(criterion).__name__}")
        return criterion
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """创建优化器
        
        Args:
            model: 模型
            
        Returns:
            优化器
        """
        training_config = self.config.get('training', {})
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.get('learning_rate', 0.0001),
            weight_decay=training_config.get('weight_decay', 0.0001)
        )
        
        logger.info(f"优化器: Adam, 学习率: {training_config.get('learning_rate', 0.0001)}")
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """创建学习率调度器
        
        Args:
            optimizer: 优化器
            
        Returns:
            学习率调度器
        """
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        
        if not scheduler_config:
            return None
        
        scheduler_type = scheduler_config.get('type', 'StepLR')
        
        if scheduler_type == 'StepLR':
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('training', {}).get('epochs', 100)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            logger.warning(f"未知的调度器类型: {scheduler_type}")
            return None
        
        logger.info(f"学习率调度器: {scheduler_type}")
        return scheduler
    
    def train_epoch(self, model: nn.Module, train_loader: Any, criterion: nn.Module, 
                   optimizer: optim.Optimizer) -> float:
        """训练一个epoch
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            
        Returns:
            平均训练损失
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (inputs, targets, _) in enumerate(progress_bar):
            # 数据移到设备
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{total_loss / num_batches:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, model: nn.Module, val_loader: Any, criterion: nn.Module) -> float:
        """验证一个epoch
        
        Args:
            model: 模型
            val_loader: 验证数据加载器
            criterion: 损失函数
            
        Returns:
            平均验证损失
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            
            for inputs, targets, _ in progress_bar:
                # 数据移到设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 统计
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg Loss': f'{total_loss / num_batches:.6f}'
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                      scheduler: Optional[Any], epoch: int, val_loss: float, 
                      is_best: bool = False):
        """保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            val_loss: 验证损失
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
        
        # 定期保存
        if epoch % self.config.get('training', {}).get('checkpoint_interval', 10) == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if len(self.train_losses) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 学习率曲线
        if self.learning_rates:
            ax2.plot(epochs, self.learning_rates, 'g-', label='学习率', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('学习率变化')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存图像
        plot_path = self.visualization_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存: {plot_path}")
    
    def train(self):
        """主训练循环"""
        logger.info("开始多尺度超分辨率重构训练...")
        
        # 加载数据
        train_loader, val_loader, test_loader, data_info = self.load_data()
        
        # 创建模型
        model = self.create_model(data_info)
        
        # 创建损失函数、优化器和调度器
        criterion = self.create_criterion()
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        
        # 训练配置
        num_epochs = self.config.get('training', {}).get('epochs', 100)
        early_stopping_patience = self.config.get('training', {}).get('early_stopping', {}).get('patience', 15)
        early_stopping_counter = 0
        
        logger.info(f"开始训练，总epoch数: {num_epochs}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # 验证
            val_loss = self.validate_epoch(model, val_loader, criterion)
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 学习率调度
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
            
            # 检查是否为最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # 保存检查点
            if self.config.get('training', {}).get('save_checkpoint', True):
                self.save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, is_best)
            
            # 日志记录
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Best Val Loss: {self.best_val_loss:.6f}"
            )
            
            # 早停检查
            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"早停触发，在epoch {epoch + 1}停止训练")
                break
            
            # 绘制训练曲线
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"训练完成，总用时: {total_time / 3600:.2f} 小时")
        
        # 最终绘制训练曲线
        self.plot_training_curves()
        
        # 测试最佳模型
        self.test_model(model, test_loader, criterion, data_info)
        
        return model
    
    def test_model(self, model: nn.Module, test_loader: Any, criterion: nn.Module, data_info: Dict):
        """测试模型
        
        Args:
            model: 模型
            test_loader: 测试数据加载器
            criterion: 损失函数
            data_info: 数据信息
        """
        logger.info("开始测试模型...")
        
        # 加载最佳模型
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("已加载最佳模型权重")
        
        model.eval()
        total_loss = 0.0
        num_batches = 0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(test_loader, desc="Testing"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # 保存预测结果（仅保存前几个批次用于可视化）
                if len(predictions) < 5:
                    predictions.append(outputs.cpu().numpy())
                    targets_list.append(targets.cpu().numpy())
        
        avg_test_loss = total_loss / num_batches
        logger.info(f"测试完成，平均测试损失: {avg_test_loss:.6f}")
        
        # 保存测试结果
        test_results = {
            'test_loss': avg_test_loss,
            'best_val_loss': self.best_val_loss,
            'data_info': data_info,
            'config': self.config
        }
        
        results_path = self.output_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        logger.info(f"测试结果已保存: {results_path}")
        
        # 可视化预测结果
        if predictions and self.config.get('evaluation', {}).get('visualization', {}).get('enabled', True):
            self.visualize_predictions(predictions, targets_list, data_info)
    
    def visualize_predictions(self, predictions: List[np.ndarray], targets: List[np.ndarray], data_info: Dict):
        """可视化预测结果
        
        Args:
            predictions: 预测结果列表
            targets: 目标结果列表
            data_info: 数据信息
        """
        logger.info("开始生成预测可视化...")
        
        try:
            # 获取分辨率信息
            low_res = data_info['low_resolution']
            high_res = data_info['original_resolution']
            
            for i, (pred_batch, target_batch) in enumerate(zip(predictions, targets)):
                # 取第一个样本和第一个时间步
                pred_sample = pred_batch[0, 0]  # [H*W*C]
                target_sample = target_batch[0, 0]  # [H*W*C]
                
                # 重塑为2D
                pred_2d = pred_sample.reshape(high_res[0], high_res[1])
                target_2d = target_sample.reshape(high_res[0], high_res[1])
                
                # 创建可视化
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 预测结果
                im1 = axes[0].imshow(pred_2d, cmap='viridis')
                axes[0].set_title(f'预测结果 ({high_res[0]}x{high_res[1]})')
                axes[0].set_xlabel('X')
                axes[0].set_ylabel('Y')
                plt.colorbar(im1, ax=axes[0])
                
                # 真实目标
                im2 = axes[1].imshow(target_2d, cmap='viridis')
                axes[1].set_title(f'真实目标 ({high_res[0]}x{high_res[1]})')
                axes[1].set_xlabel('X')
                axes[1].set_ylabel('Y')
                plt.colorbar(im2, ax=axes[1])
                
                # 误差图
                error = np.abs(pred_2d - target_2d)
                im3 = axes[2].imshow(error, cmap='hot')
                axes[2].set_title('绝对误差')
                axes[2].set_xlabel('X')
                axes[2].set_ylabel('Y')
                plt.colorbar(im3, ax=axes[2])
                
                plt.tight_layout()
                
                # 保存图像
                vis_path = self.visualization_dir / f'prediction_sample_{i}.png'
                plt.savefig(vis_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"预测可视化已保存: {vis_path}")
                
                if i >= 4:  # 只保存前5个样本
                    break
        
        except Exception as e:
            logger.error(f"预测可视化失败: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    # 配置文件路径
    config_path = "configs/multiscale_config.yaml"
    
    if not Path(config_path).exists():
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    try:
        # 加载配置
        config = load_config(config_path)
        
        # 设置随机种子
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 创建训练器并开始训练
        trainer = MultiScaleTrainer(config)
        model = trainer.train()
        
        logger.info("多尺度超分辨率重构训练完成！")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


if __name__ == "__main__":
    main()