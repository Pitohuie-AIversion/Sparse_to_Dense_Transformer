#!/usr/bin/env python3
"""
DarcyFlow 数据集训练示例

这个脚本演示了如何使用 DarcyFlow 数据集训练 VIVTransformer 模型。
"""

import sys
import yaml
import torch
import logging
from pathlib import Path
from torch.optim import Adam
from torch.nn import MSELoss

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from data.dataloader import get_adaptive_loaders
from models.vivtransformer import VIVTransformer
from models.embedding import EmbeddingAndEncoding

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_model(config: dict) -> torch.nn.Module:
    """创建模型"""
    model_config = config['model']
    
    # 创建嵌入层
    embedding = EmbeddingAndEncoding(
        input_dim=model_config['input_dim'],
        d_model=model_config['d_model'],
        max_time_steps=model_config['max_time_steps']
    )
    
    # 创建主模型
    model = VIVTransformer(
        attention_type=model_config['attention_type'],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        input_dim=model_config['input_dim'],
        output_dim=model_config['output_dim'],
        seq_len=model_config['seq_len'],
        embedding_layer=embedding
    )
    
    return model


def update_model_config_from_data(config: dict, data_loader) -> dict:
    """根据数据加载器信息更新模型配置"""
    # 获取数据信息
    data_info = data_loader.get_data_info()
    
    # 更新模型配置
    config['model']['input_dim'] = data_info['input_dim']
    config['model']['output_dim'] = data_info['output_dim']
    config['model']['seq_len'] = data_info['sequence_length']
    
    logger.info(f"模型配置已更新:")
    logger.info(f"  输入维度: {data_info['input_dim']}")
    logger.info(f"  输出维度: {data_info['output_dim']}")
    logger.info(f"  序列长度: {data_info['sequence_length']}")
    
    return config


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets, time_steps) in enumerate(train_loader):
        # 移动数据到设备
        inputs = inputs.to(device)
        targets = targets.to(device)
        if time_steps is not None:
            time_steps = time_steps.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs, time_steps)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 每50个批次打印一次进度
        if batch_idx % 50 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, valid_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, time_steps in valid_loader:
            # 移动数据到设备
            inputs = inputs.to(device)
            targets = targets.to(device)
            if time_steps is not None:
                time_steps = time_steps.to(device)
            
            # 前向传播
            outputs = model(inputs, time_steps)
            
            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """主函数"""
    logger.info("🚀 开始 DarcyFlow 数据集训练示例")
    
    # 加载配置
    config_path = "configs/darcy_flow_config.yaml"
    if not Path(config_path).exists():
        logger.error(f"配置文件不存在: {config_path}")
        return False
    
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device(config['global']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    try:
        # 创建数据加载器
        logger.info("📊 创建数据加载器...")
        train_loader, valid_loader, test_loader = get_adaptive_loaders(config)
        
        logger.info(f"数据加载器创建成功:")
        logger.info(f"  训练集批次数: {len(train_loader)}")
        logger.info(f"  验证集批次数: {len(valid_loader)}")
        logger.info(f"  测试集批次数: {len(test_loader)}")
        
        # 根据数据更新模型配置
        config = update_model_config_from_data(config, train_loader)
        
        # 创建模型
        logger.info("🏗️ 创建模型...")
        model = create_model(config)
        model = model.to(device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
        # 创建优化器和损失函数
        optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
        criterion = MSELoss()
        
        # 训练循环
        logger.info("🎯 开始训练...")
        num_epochs = config['training']['epochs']
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n=== Epoch {epoch}/{num_epochs} ===")
            
            # 训练
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
            logger.info(f"训练损失: {train_loss:.6f}")
            
            # 验证
            val_loss = validate(model, valid_loader, criterion, device)
            logger.info(f"验证损失: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': config
                }, 'best_darcy_flow_model.pth')
                logger.info(f"💾 保存最佳模型 (验证损失: {val_loss:.6f})")
        
        # 测试最佳模型
        logger.info("\n🧪 测试最佳模型...")
        checkpoint = torch.load('best_darcy_flow_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss = validate(model, test_loader, criterion, device)
        logger.info(f"测试损失: {test_loss:.6f}")
        
        logger.info("\n🎉 训练完成！")
        logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"最终测试损失: {test_loss:.6f}")
        logger.info(f"模型已保存为: best_darcy_flow_model.pth")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("\n✅ 示例运行成功！")
    else:
        logger.error("\n❌ 示例运行失败！")
        sys.exit(1)