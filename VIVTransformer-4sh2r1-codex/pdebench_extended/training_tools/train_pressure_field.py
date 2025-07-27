#!/usr/bin/env python3
"""
压力场重建训练脚本

使用新的压力场适配器训练20x20→200x200压力场重建模型
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from data.pressure_field_adapter import (
    create_pressure_field_datasets,
    analyze_pressure_field_data
)
from data.unified_adapter import UnifiedDataAdapter

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PressureFieldTransformer(nn.Module):
    """
    压力场重建Transformer模型
    
    专门用于20x20→200x200压力场重建任务
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 模型参数
        model_config = config['model']['transformer']
        self.d_model = model_config['d_model']
        self.nhead = model_config['nhead']
        self.num_encoder_layers = model_config['num_encoder_layers']
        self.num_decoder_layers = model_config['num_decoder_layers']
        self.dim_feedforward = model_config['dim_feedforward']
        self.dropout = model_config['dropout']
        
        # 输入输出维度
        self.input_dim = config['data']['input_dim']  # 400
        self.output_dim = config['data']['output_dim']  # 40000
        
        # 输入投影
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # 位置编码
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=model_config['activation'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )
        
        # 输出投影
        output_config = model_config['output_projection']
        if output_config['type'] == 'mlp':
            layers = []
            hidden_dims = output_config['hidden_dims']
            
            # 从d_model开始
            prev_dim = self.d_model
            for hidden_dim in hidden_dims[:-1]:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(output_config['dropout'])
                ])
                prev_dim = hidden_dim
            
            # 最后一层
            layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
            self.output_projection = nn.Sequential(*layers)
        else:
            self.output_projection = nn.Linear(self.d_model, self.output_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_positional_encoding(self):
        """创建位置编码"""
        pe_config = self.config['model']['transformer']['positional_encoding']
        
        if pe_config['type'] == 'learned':
            # 学习的位置编码
            max_len = pe_config['max_len']
            return nn.Parameter(torch.randn(1, max_len, self.d_model))
        elif pe_config['type'] == 'sinusoidal':
            # 正弦位置编码
            return self._create_sinusoidal_encoding(pe_config['max_len'])
        else:
            return None
    
    def _create_sinusoidal_encoding(self, max_len: int):
        """创建正弦位置编码"""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _initialize_weights(self):
        """初始化模型权重"""
        init_config = self.config['model']['initialization']
        init_type = init_config['type']
        gain = init_config['gain']
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight, gain=gain)
                elif init_type == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                elif init_type == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, input_dim]
        
        Returns:
            输出张量，形状为 [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 添加位置编码
        if self.positional_encoding is not None:
            if self.positional_encoding.size(1) >= x.size(1):
                pe = self.positional_encoding[:, :x.size(1), :]
                x = x + pe
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        
        # 输出投影
        x = x.squeeze(1)  # [batch_size, d_model]
        output = self.output_projection(x)  # [batch_size, output_dim]
        
        return output


class PressureFieldLoss(nn.Module):
    """
    压力场重建损失函数
    
    结合多种损失函数以提高重建质量
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.loss_config = config['loss']
        
        # 主损失函数
        primary_type = self.loss_config['primary']['type']
        if primary_type == 'mse':
            self.primary_loss = nn.MSELoss()
        elif primary_type == 'mae':
            self.primary_loss = nn.L1Loss()
        elif primary_type == 'huber':
            self.primary_loss = nn.HuberLoss()
        elif primary_type == 'smooth_l1':
            self.primary_loss = nn.SmoothL1Loss()
        
        self.primary_weight = self.loss_config['primary']['weight']
        
        # 辅助损失函数权重
        aux_config = self.loss_config['auxiliary']
        self.gradient_weight = aux_config['gradient']['weight'] if aux_config['gradient']['enabled'] else 0
        self.frequency_weight = aux_config['frequency']['weight'] if aux_config['frequency']['enabled'] else 0
        self.physics_weight = aux_config['physics']['weight'] if aux_config['physics']['enabled'] else 0
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失
        
        Args:
            pred: 预测张量，形状为 [batch_size, output_dim]
            target: 目标张量，形状为 [batch_size, output_dim]
        
        Returns:
            总损失和各项损失的字典
        """
        losses = {}
        
        # 主损失
        primary_loss = self.primary_loss(pred, target)
        losses['primary'] = primary_loss
        total_loss = self.primary_weight * primary_loss
        
        # 梯度损失
        if self.gradient_weight > 0:
            gradient_loss = self._compute_gradient_loss(pred, target)
            losses['gradient'] = gradient_loss
            total_loss += self.gradient_weight * gradient_loss
        
        # 频域损失
        if self.frequency_weight > 0:
            frequency_loss = self._compute_frequency_loss(pred, target)
            losses['frequency'] = frequency_loss
            total_loss += self.frequency_weight * frequency_loss
        
        # 物理约束损失
        if self.physics_weight > 0:
            physics_loss = self._compute_physics_loss(pred, target)
            losses['physics'] = physics_loss
            total_loss += self.physics_weight * physics_loss
        
        losses['total'] = total_loss
        return total_loss, losses
    
    def _compute_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算梯度损失"""
        # 重塑为2D
        pred_2d = pred.view(-1, 200, 200)
        target_2d = target.view(-1, 200, 200)
        
        # 计算梯度
        pred_grad_x = torch.diff(pred_2d, dim=2)
        pred_grad_y = torch.diff(pred_2d, dim=1)
        
        target_grad_x = torch.diff(target_2d, dim=2)
        target_grad_y = torch.diff(target_2d, dim=1)
        
        # 梯度损失
        grad_loss_x = nn.functional.mse_loss(pred_grad_x, target_grad_x)
        grad_loss_y = nn.functional.mse_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y
    
    def _compute_frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算频域损失"""
        # 重塑为2D
        pred_2d = pred.view(-1, 200, 200)
        target_2d = target.view(-1, 200, 200)
        
        # 2D FFT
        pred_fft = torch.fft.fft2(pred_2d)
        target_fft = torch.fft.fft2(target_2d)
        
        # 频域损失（幅度谱）
        pred_magnitude = torch.abs(pred_fft)
        target_magnitude = torch.abs(target_fft)
        
        return nn.functional.mse_loss(pred_magnitude, target_magnitude)
    
    def _compute_physics_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算物理约束损失"""
        # 简单的连续性约束（拉普拉斯算子）
        pred_2d = pred.view(-1, 200, 200)
        target_2d = target.view(-1, 200, 200)
        
        # 计算拉普拉斯算子
        def laplacian(x):
            # 简单的5点差分格式
            laplace = -4 * x[:, 1:-1, 1:-1]
            laplace += x[:, :-2, 1:-1]  # 上
            laplace += x[:, 2:, 1:-1]   # 下
            laplace += x[:, 1:-1, :-2]  # 左
            laplace += x[:, 1:-1, 2:]   # 右
            return laplace
        
        pred_laplace = laplacian(pred_2d)
        target_laplace = laplacian(target_2d)
        
        return nn.functional.mse_loss(pred_laplace, target_laplace)


class PressureFieldTrainer:
    """
    压力场重建训练器
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['global']['device'])
        
        # 设置随机种子
        torch.manual_seed(config['global']['random_seed'])
        np.random.seed(config['global']['random_seed'])
        
        # 创建模型
        self.model = PressureFieldTransformer(config).to(self.device)
        
        # 创建损失函数
        self.criterion = PressureFieldLoss(config)
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # 日志记录
        self.writer = None
        if config['logging']['tensorboard']['enabled']:
            log_dir = config['logging']['tensorboard']['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        # 检查点目录
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"训练器初始化完成，模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_optimizer(self):
        """创建优化器"""
        opt_config = self.config['training']['optimizer']
        opt_type = opt_config['type']
        
        if opt_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=opt_config['betas'],
                eps=opt_config['eps']
            )
        elif opt_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                betas=opt_config['betas'],
                eps=opt_config['eps']
            )
        elif opt_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                momentum=0.9
            )
        else:
            raise ValueError(f"不支持的优化器类型: {opt_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        sched_config = self.config['training']['scheduler']
        sched_type = sched_config['type']
        
        if sched_type == 'cosine_annealing':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config['eta_min']
            )
        elif sched_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config['mode'],
                factor=sched_config['factor'],
                patience=sched_config['patience'],
                threshold=sched_config['threshold']
            )
        else:
            return None
    
    def _create_data_loaders(self):
        """创建数据加载器"""
        data_config = self.config['data']
        
        return create_pressure_field_datasets(
            data_path=data_config['path'],
            batch_size=data_config['batch_size'],
            normalize=data_config['normalize'],
            normalize_method=data_config['normalize_method'],
            augmentation=data_config['augmentation']['enabled'],
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (inputs, targets, _) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            loss, losses = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config['training']['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录日志
            if self.writer and batch_idx % self.config['logging']['log_frequency'] == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/LearningRate', 
                                     self.optimizer.param_groups[0]['lr'], global_step)
        
        # 计算平均损失
        avg_losses = {key: value / len(self.train_loader) for key, value in total_losses.items()}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_losses = {}
        
        with torch.no_grad():
            for inputs, targets, _ in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss, losses = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()
        
        # 计算平均损失
        avg_losses = {key: value / len(self.val_loader) for key, value in total_losses.items()}
        
        return avg_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        if self.config['checkpoint']['save_last']:
            torch.save(checkpoint, self.checkpoint_dir / 'last.pth')
        
        # 保存最佳检查点
        if is_best and self.config['checkpoint']['save_best']:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
        
        # 保存定期检查点
        if self.current_epoch % self.config['checkpoint']['save_frequency'] == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.current_epoch}.pth')
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch()
            
            # 验证
            val_losses = self.validate()
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # 记录日志
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}:")
            logger.info(f"  Train Loss: {train_losses['total']:.6f}")
            logger.info(f"  Val Loss: {val_losses['total']:.6f}")
            
            if self.writer:
                self.writer.add_scalar('Train/Loss', train_losses['total'], epoch)
                self.writer.add_scalar('Val/Loss', val_losses['total'], epoch)
                
                for key, value in train_losses.items():
                    if key != 'total':
                        self.writer.add_scalar(f'Train/{key}', value, epoch)
                
                for key, value in val_losses.items():
                    if key != 'total':
                        self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # 检查是否为最佳模型
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.early_stopping_counter = 0
                logger.info(f"  新的最佳验证损失: {self.best_val_loss:.6f}")
            else:
                self.early_stopping_counter += 1
            
            # 保存检查点
            self.save_checkpoint(is_best)
            
            # 早停检查
            if (self.config['training']['early_stopping']['enabled'] and 
                self.early_stopping_counter >= self.config['training']['early_stopping']['patience']):
                logger.info(f"早停触发，在epoch {epoch + 1}停止训练")
                break
        
        logger.info("训练完成!")
        
        if self.writer:
            self.writer.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修复数值类型的配置项
    if 'training' in config:
        # 确保学习率是浮点数
        if 'learning_rate' in config['training']:
            config['training']['learning_rate'] = float(config['training']['learning_rate'])
        # 确保权重衰减是浮点数
        if 'weight_decay' in config['training']:
            config['training']['weight_decay'] = float(config['training']['weight_decay'])
        # 确保梯度裁剪是浮点数
        if 'gradient_clip_norm' in config['training']:
            config['training']['gradient_clip_norm'] = float(config['training']['gradient_clip_norm'])
        
        # 修复优化器配置中的数值参数
        if 'optimizer' in config['training']:
            opt_config = config['training']['optimizer']
            if 'eps' in opt_config:
                opt_config['eps'] = float(opt_config['eps'])
            if 'betas' in opt_config and isinstance(opt_config['betas'], list):
                opt_config['betas'] = [float(x) for x in opt_config['betas']]
        
        # 修复调度器配置中的数值参数
        if 'scheduler' in config['training']:
            sched_config = config['training']['scheduler']
            if 'eta_min' in sched_config:
                sched_config['eta_min'] = float(sched_config['eta_min'])
            if 'gamma' in sched_config:
                sched_config['gamma'] = float(sched_config['gamma'])
            if 'factor' in sched_config:
                sched_config['factor'] = float(sched_config['factor'])
            if 'threshold' in sched_config:
                sched_config['threshold'] = float(sched_config['threshold'])
    
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='压力场重建训练')
    parser.add_argument('--config', type=str, 
                       default='configs/pressure_field_training.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_path', type=str,
                       help='数据文件路径（覆盖配置文件中的路径）')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help='计算设备（覆盖配置文件中的设备）')
    parser.add_argument('--batch_size', type=int,
                       help='批次大小（覆盖配置文件中的批次大小）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 命令行参数覆盖
    if args.data_path:
        config['data']['path'] = args.data_path
    if args.device:
        config['global']['device'] = args.device
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # 检查数据文件
    data_path = config['data']['path']
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        logger.info("请检查配置文件中的数据路径或使用--data_path参数指定正确的路径")
        return
    
    # 分析数据
    logger.info("分析数据...")
    try:
        data_analysis = analyze_pressure_field_data(data_path)
        logger.info(f"数据分析完成:")
        logger.info(f"  总样本数: {data_analysis['dataset_info']['total_samples']}")
        logger.info(f"  输入维度: {data_analysis['dataset_info']['input_dim']}")
        logger.info(f"  输出维度: {data_analysis['dataset_info']['output_dim']}")
    except Exception as e:
        logger.error(f"数据分析失败: {e}")
        return
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = PressureFieldTrainer(config)
    
    # 开始训练
    trainer.train()
    
    logger.info("训练脚本执行完成!")


if __name__ == "__main__":
    main()