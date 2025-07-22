#!/usr/bin/env python3
"""
压力场数据处理示例

展示如何使用新的压力场适配器处理20x20→200x200压力场重建任务
"""

import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from data.pressure_field_adapter import (
    PressureFieldDataset,
    PressureFieldDataLoader,
    create_pressure_field_datasets,
    analyze_pressure_field_data
)
from data.unified_adapter import UnifiedDataAdapter, analyze_unified_data

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_pressure_field_adapter():
    """
    演示压力场适配器的基本功能
    """
    logger.info("=== 压力场适配器演示 ===")
    
    # 示例数据路径（请根据实际情况修改）
    data_path = "x:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codex/modify_multi_attention/data/pressure_data.pt"
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        logger.warning(f"数据文件不存在: {data_path}")
        logger.info("请将数据路径修改为您的实际数据文件路径")
        return
    
    try:
        # 1. 数据分析
        logger.info("1. 分析压力场数据...")
        analysis_result = analyze_pressure_field_data(data_path)
        
        logger.info("数据分析结果:")
        logger.info(f"  数据类型: {analysis_result['dataset_info']['data_type']}")
        logger.info(f"  输入维度: {analysis_result['dataset_info']['input_dim']}")
        logger.info(f"  输出维度: {analysis_result['dataset_info']['output_dim']}")
        logger.info(f"  输入形状: {analysis_result['dataset_info']['input_shape']}")
        logger.info(f"  输出形状: {analysis_result['dataset_info']['output_shape']}")
        logger.info(f"  总样本数: {analysis_result['dataset_info']['total_samples']}")
        logger.info(f"  上采样因子: {analysis_result['dataset_info']['upsampling_factor']}")
        
        # 输入数据统计
        input_stats = analysis_result['input_statistics']
        logger.info(f"  输入数据统计: 均值={input_stats['mean']:.4f}, 标准差={input_stats['std']:.4f}")
        logger.info(f"                范围=[{input_stats['min']:.4f}, {input_stats['max']:.4f}]")
        
        # 输出数据统计
        output_stats = analysis_result['output_statistics']
        logger.info(f"  输出数据统计: 均值={output_stats['mean']:.4f}, 标准差={output_stats['std']:.4f}")
        logger.info(f"                范围=[{output_stats['min']:.4f}, {output_stats['max']:.4f}]")
        
        # 2. 创建数据集
        logger.info("\n2. 创建压力场数据集...")
        train_loader, val_loader, test_loader = create_pressure_field_datasets(
            data_path=data_path,
            batch_size=16,
            normalize=True,
            normalize_method='minmax',
            augmentation=True,
            num_workers=2
        )
        
        logger.info(f"训练集批次数: {len(train_loader)}")
        logger.info(f"验证集批次数: {len(val_loader)}")
        logger.info(f"测试集批次数: {len(test_loader)}")
        
        # 3. 检查数据加载
        logger.info("\n3. 检查数据加载...")
        for batch_idx, (inputs, targets, time_steps) in enumerate(train_loader):
            logger.info(f"批次 {batch_idx + 1}:")
            logger.info(f"  输入形状: {inputs.shape}")
            logger.info(f"  目标形状: {targets.shape}")
            logger.info(f"  时间步形状: {time_steps.shape}")
            logger.info(f"  输入数据范围: [{inputs.min():.4f}, {inputs.max():.4f}]")
            logger.info(f"  目标数据范围: [{targets.min():.4f}, {targets.max():.4f}]")
            
            if batch_idx >= 2:  # 只检查前3个批次
                break
        
        # 4. 可视化样本
        logger.info("\n4. 可视化数据样本...")
        visualize_pressure_field_samples(train_loader.dataset.dataset, num_samples=3)
        
        logger.info("\n压力场适配器演示完成!")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def demo_unified_adapter():
    """
    演示统一数据适配器的自动检测功能
    """
    logger.info("\n=== 统一数据适配器演示 ===")
    
    # 示例配置
    config = {
        'data': {
            'path': "x:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codex/modify_multi_attention/data/pressure_data.pt",
            'batch_size': 16,
            'normalize': True,
            'num_workers': 2,
            'pin_memory': True
        }
    }
    
    try:
        # 创建统一适配器
        adapter = UnifiedDataAdapter(config)
        
        # 自动检测数据类型
        data_type = adapter.detect_data_type(config['data']['path'])
        logger.info(f"自动检测的数据类型: {data_type}")
        
        # 获取数据信息
        data_info = adapter.get_data_info()
        logger.info(f"数据信息: {data_info}")
        
        # 分析数据
        if os.path.exists(config['data']['path']):
            analysis = analyze_unified_data(config)
            logger.info(f"统一数据分析完成")
        
        logger.info("统一数据适配器演示完成!")
        
    except Exception as e:
        logger.error(f"统一适配器演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def visualize_pressure_field_samples(dataset, num_samples=3):
    """
    可视化压力场样本
    
    Args:
        dataset: 压力场数据集
        num_samples: 可视化样本数量
    """
    logger.info(f"可视化 {num_samples} 个压力场样本...")
    
    # 创建图形
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        try:
            # 获取样本
            input_tensor, output_tensor, time_step = dataset[i]
            
            # 重塑为2D
            input_2d = input_tensor.numpy().reshape(20, 20)
            output_2d = output_tensor.numpy().reshape(200, 200)
            
            # 绘制输入
            im1 = axes[i, 0].imshow(input_2d, cmap='coolwarm', interpolation='nearest')
            axes[i, 0].set_title(f'样本 {i+1} - 输入 (20x20)\nt={time_step:.3f}')
            axes[i, 0].set_xlabel('X')
            axes[i, 0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # 绘制输出
            im2 = axes[i, 1].imshow(output_2d, cmap='coolwarm', interpolation='nearest')
            axes[i, 1].set_title(f'样本 {i+1} - 输出 (200x200)\nt={time_step:.3f}')
            axes[i, 1].set_xlabel('X')
            axes[i, 1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[i, 1])
            
        except Exception as e:
            logger.error(f"可视化样本 {i+1} 时出现错误: {e}")
    
    plt.tight_layout()
    
    # 保存图像
    save_path = Path(__file__).parent / "pressure_field_samples.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"样本可视化已保存到: {save_path}")
    
    # 显示图像（如果在交互环境中）
    try:
        plt.show()
    except:
        pass
    
    plt.close()


def demo_data_augmentation():
    """
    演示数据增强功能
    """
    logger.info("\n=== 数据增强演示 ===")
    
    data_path = "x:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codex/modify_multi_attention/data/pressure_data.pt"
    
    if not os.path.exists(data_path):
        logger.warning(f"数据文件不存在: {data_path}")
        return
    
    try:
        # 创建带数据增强的数据集
        train_dataset = PressureFieldDataset(
            data_path=data_path,
            split='train',
            normalize=True,
            augmentation=True  # 启用数据增强
        )
        
        # 创建不带数据增强的数据集
        train_dataset_no_aug = PressureFieldDataset(
            data_path=data_path,
            split='train',
            normalize=True,
            augmentation=False  # 不启用数据增强
        )
        
        logger.info("比较数据增强效果...")
        
        # 获取同一个样本的不同版本
        sample_idx = 0
        
        # 原始样本
        input_orig, output_orig, time_step = train_dataset_no_aug[sample_idx]
        
        # 增强样本（多次获取同一索引会得到不同的增强结果）
        augmented_samples = []
        for _ in range(3):
            input_aug, output_aug, _ = train_dataset[sample_idx]
            augmented_samples.append((input_aug, output_aug))
        
        # 可视化比较
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # 原始样本
        input_2d = input_orig.numpy().reshape(20, 20)
        output_2d = output_orig.numpy().reshape(200, 200)
        
        axes[0, 0].imshow(input_2d, cmap='coolwarm')
        axes[0, 0].set_title('原始输入 (20x20)')
        axes[1, 0].imshow(output_2d, cmap='coolwarm')
        axes[1, 0].set_title('原始输出 (200x200)')
        
        # 增强样本
        for i, (input_aug, output_aug) in enumerate(augmented_samples):
            input_2d_aug = input_aug.numpy().reshape(20, 20)
            output_2d_aug = output_aug.numpy().reshape(200, 200)
            
            axes[0, i+1].imshow(input_2d_aug, cmap='coolwarm')
            axes[0, i+1].set_title(f'增强输入 {i+1}')
            axes[1, i+1].imshow(output_2d_aug, cmap='coolwarm')
            axes[1, i+1].set_title(f'增强输出 {i+1}')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = Path(__file__).parent / "data_augmentation_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"数据增强比较图已保存到: {save_path}")
        
        plt.close()
        
        logger.info("数据增强演示完成!")
        
    except Exception as e:
        logger.error(f"数据增强演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def demo_normalization_methods():
    """
    演示不同归一化方法的效果
    """
    logger.info("\n=== 归一化方法演示 ===")
    
    data_path = "x:/2025/Graduation_project/report/VIVTransformer-4sh2r1-codex/modify_multi_attention/data/pressure_data.pt"
    
    if not os.path.exists(data_path):
        logger.warning(f"数据文件不存在: {data_path}")
        return
    
    try:
        normalization_methods = ['minmax', 'zscore']
        
        for method in normalization_methods:
            logger.info(f"\n测试归一化方法: {method}")
            
            # 创建数据集
            dataset = PressureFieldDataset(
                data_path=data_path,
                split='train',
                normalize=True,
                normalize_method=method
            )
            
            # 收集一些样本的统计信息
            inputs = []
            outputs = []
            
            for i in range(min(100, len(dataset))):
                input_tensor, output_tensor, _ = dataset[i]
                inputs.append(input_tensor.numpy())
                outputs.append(output_tensor.numpy())
            
            inputs = np.array(inputs)
            outputs = np.array(outputs)
            
            logger.info(f"  输入数据统计 ({method}):")
            logger.info(f"    均值: {np.mean(inputs):.6f}")
            logger.info(f"    标准差: {np.std(inputs):.6f}")
            logger.info(f"    范围: [{np.min(inputs):.6f}, {np.max(inputs):.6f}]")
            
            logger.info(f"  输出数据统计 ({method}):")
            logger.info(f"    均值: {np.mean(outputs):.6f}")
            logger.info(f"    标准差: {np.std(outputs):.6f}")
            logger.info(f"    范围: [{np.min(outputs):.6f}, {np.max(outputs):.6f}]")
        
        logger.info("\n归一化方法演示完成!")
        
    except Exception as e:
        logger.error(f"归一化演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    主函数 - 运行所有演示
    """
    logger.info("开始压力场数据处理演示...")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        logger.info(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
        logger.info(f"当前设备: {torch.cuda.get_device_name()}")
    else:
        logger.info("CUDA不可用，使用CPU")
    
    # 运行各种演示
    demo_pressure_field_adapter()
    demo_unified_adapter()
    demo_data_augmentation()
    demo_normalization_methods()
    
    logger.info("\n所有演示完成!")
    logger.info("\n使用说明:")
    logger.info("1. 请确保数据文件路径正确")
    logger.info("2. 可以根据需要调整批次大小、归一化方法等参数")
    logger.info("3. 生成的可视化图像保存在examples目录下")
    logger.info("4. 可以将此适配器集成到您的训练流程中")


if __name__ == "__main__":
    main()