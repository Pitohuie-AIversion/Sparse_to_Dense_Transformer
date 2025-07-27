#!/usr/bin/env python3
"""
使用自适应裁剪数据进行训练的示例

训练配置:
- 输入: 自适应裁剪后的数据 (64x64)
- 输出: 原始数据 (128x128)
- 任务: 从裁剪数据重建原始数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入自定义模块
from data.adaptive_transforms import create_darcy_flow_transform

class AdaptiveCropDataset(Dataset):
    """
    自适应裁剪数据集
    输入: 裁剪后的数据
    输出: 原始数据
    """
    
    def __init__(self, data_path, crop_multiplier=0.5, split='train', max_samples=1000):
        self.data_path = data_path
        self.crop_multiplier = crop_multiplier
        self.split = split
        self.max_samples = max_samples
        
        # 创建裁剪变换
        self.crop_transform = create_darcy_flow_transform(crop_multiplier=crop_multiplier)
        
        # 加载数据信息
        with h5py.File(data_path, 'r') as f:
            self.total_samples = min(f['tensor'].shape[0], max_samples)
            print(f"数据集大小: {self.total_samples} 样本")
            print(f"原始数据形状: {f['tensor'].shape}")
            print(f"裁剪倍数: {crop_multiplier}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        with h5py.File(self.data_path, 'r') as f:
            # 加载原始数据
            original_tensor = torch.from_numpy(f['tensor'][idx]).float()  # (1, 128, 128)
            original_nu = torch.from_numpy(f['nu'][idx]).float()  # (128, 128)
            
            # 应用裁剪变换
            cropped_tensor = self.crop_transform(original_tensor)  # (1, 64, 64)
            cropped_nu = self.crop_transform(original_nu)  # (64, 64)
            
            # 组合输入和输出
            # 输入: 裁剪后的数据
            input_data = torch.cat([
                cropped_tensor,  # (1, 64, 64)
                cropped_nu.unsqueeze(0)  # (1, 64, 64)
            ], dim=0)  # (2, 64, 64)
            
            # 输出: 原始数据
            target_data = torch.cat([
                original_tensor,  # (1, 128, 128)
                original_nu.unsqueeze(0)  # (1, 128, 128)
            ], dim=0)  # (2, 128, 128)
            
            return input_data, target_data

class SimpleUpscaleModel(nn.Module):
    """
    简单的上采样重建模型
    从64x64重建到128x128
    """
    
    def __init__(self, input_channels=2, output_channels=2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 64x64 -> 128x128
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 3, padding=1)
        )
    
    def forward(self, x):
        # x: (batch, 2, 64, 64)
        encoded = self.encoder(x)  # (batch, 256, 64, 64)
        decoded = self.decoder(encoded)  # (batch, 2, 128, 128)
        return decoded

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)  # (batch, 2, 64, 64)
        targets = targets.to(device)  # (batch, 2, 128, 128)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)  # (batch, 2, 128, 128)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device):
    """
    验证一个epoch
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def visualize_results(model, dataset, device, save_path='training_results.png'):
    """
    可视化训练结果
    """
    model.eval()
    
    # 获取一个样本
    inputs, targets = dataset[0]
    inputs = inputs.unsqueeze(0).to(device)  # (1, 2, 64, 64)
    targets = targets.unsqueeze(0).to(device)  # (1, 2, 128, 128)
    
    with torch.no_grad():
        outputs = model(inputs)  # (1, 2, 128, 128)
    
    # 转换为numpy
    inputs_np = inputs.cpu().numpy()[0]  # (2, 64, 64)
    targets_np = targets.cpu().numpy()[0]  # (2, 128, 128)
    outputs_np = outputs.cpu().numpy()[0]  # (2, 128, 128)
    
    # 创建可视化
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # 输入数据 (裁剪后)
    im1 = axes[0, 0].imshow(inputs_np[0], cmap='viridis')
    axes[0, 0].set_title('输入 Tensor (64x64)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(inputs_np[1], cmap='plasma')
    axes[0, 1].set_title('输入 Nu (64x64)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 目标数据 (原始)
    im3 = axes[1, 0].imshow(targets_np[0], cmap='viridis')
    axes[1, 0].set_title('目标 Tensor (128x128)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(targets_np[1], cmap='plasma')
    axes[1, 1].set_title('目标 Nu (128x128)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 预测数据 (重建)
    im5 = axes[2, 0].imshow(outputs_np[0], cmap='viridis')
    axes[2, 0].set_title('预测 Tensor (128x128)')
    plt.colorbar(im5, ax=axes[2, 0])
    
    im6 = axes[2, 1].imshow(outputs_np[1], cmap='plasma')
    axes[2, 1].set_title('预测 Nu (128x128)')
    plt.colorbar(im6, ax=axes[2, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练结果可视化已保存到: {save_path}")
    
    # 计算重建误差
    mse_tensor = np.mean((targets_np[0] - outputs_np[0]) ** 2)
    mse_nu = np.mean((targets_np[1] - outputs_np[1]) ** 2)
    
    print(f"重建误差:")
    print(f"  Tensor MSE: {mse_tensor:.6f}")
    print(f"  Nu MSE: {mse_nu:.6f}")

def main():
    # 配置参数
    config = {
        'data_path': 'PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5',
        'crop_multiplier': 0.5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'max_samples': 1000,  # 限制样本数量以加快训练
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=== 自适应裁剪数据训练示例 ===")
    print(f"设备: {config['device']}")
    print(f"数据路径: {config['data_path']}")
    print(f"裁剪倍数: {config['crop_multiplier']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"训练轮数: {config['num_epochs']}")
    
    # 检查数据文件
    if not Path(config['data_path']).exists():
        print(f"❌ 错误: 数据文件不存在 {config['data_path']}")
        return
    
    # 创建数据集
    print("\n📊 创建数据集...")
    train_dataset = AdaptiveCropDataset(
        data_path=config['data_path'],
        crop_multiplier=config['crop_multiplier'],
        split='train',
        max_samples=config['max_samples']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0  # Windows下设置为0
    )
    
    # 创建模型
    print("\n🏗️ 创建模型...")
    model = SimpleUpscaleModel(input_channels=2, output_channels=2)
    model = model.to(config['device'])
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 训练循环
    print("\n🚀 开始训练...")
    train_losses = []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        train_losses.append(train_loss)
        
        print(f"训练损失: {train_loss:.6f}")
        
        # 每5个epoch可视化一次结果
        if (epoch + 1) % 5 == 0:
            visualize_results(
                model, train_dataset, config['device'], 
                f'training_results_epoch_{epoch+1}.png'
            )
    
    # 保存模型
    model_path = 'adaptive_crop_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses
    }, model_path)
    print(f"\n💾 模型已保存到: {model_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 最终可视化
    visualize_results(model, train_dataset, config['device'], 'final_training_results.png')
    
    print("\n✅ 训练完成!")
    print("\n📁 生成的文件:")
    print("  - adaptive_crop_model.pth: 训练好的模型")
    print("  - training_loss_curve.png: 训练损失曲线")
    print("  - final_training_results.png: 最终训练结果")
    print("  - training_results_epoch_*.png: 各epoch的训练结果")
    
    print("\n💡 训练说明:")
    print("  - 输入: 自适应裁剪后的数据 (64x64)")
    print("  - 输出: 原始数据 (128x128)")
    print("  - 任务: 从裁剪数据重建原始数据")
    print("  - 这种配置可以用于数据重建、超分辨率等任务")

if __name__ == '__main__':
    main()