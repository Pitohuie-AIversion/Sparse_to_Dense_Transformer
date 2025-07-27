#!/usr/bin/env python3
"""测试DarcyFlow数据集加载的脚本

这个脚本专门用于测试PDEBench中的DarcyFlow数据集是否可以正确加载和处理。
"""

import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from data.custom_dataset_adapter import CustomDataset, CustomDataLoader
from data.dataloader import get_custom_loaders, get_adaptive_loaders

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_darcy_dataset_direct():
    """直接测试DarcyFlow数据集加载"""
    logger.info("=== 直接测试DarcyFlow数据集 ===")
    
    try:
        # 配置参数
        config = {
            'data_root': 'PDEBench/pdebench/data_download/data/2D/DarcyFlow',
            'dataset_type': 'darcy_flow',
            'file_paths': ['2D_DarcyFlow_beta0.1_Train.hdf5'],
            'spatial_resolution': [128, 128],
            'sequence_length': 2,
            'data_keys': {
                'input': 'nu',
                'target': 'tensor'
            },
            'normalization_method': 'minmax',
            'data_split_ratios': [0.7, 0.2, 0.1],
            'input_dim': 16384,
            'output_dim': 16384
        }
        
        # 构建数据文件路径
        data_path = str(Path(config['data_root']) / config['file_paths'][0])
        
        # 创建数据集
        dataset = CustomDataset(
            data_path=data_path,
            dataset_type=config['dataset_type'],
            split='train',
            spatial_resolution=config['spatial_resolution'],
            sequence_length=config['sequence_length'],
            data_key=config['data_keys']['input'],
            target_key=config['data_keys']['target'],
            normalization_method=config['normalization_method'],
            custom_split_ratios=config['data_split_ratios']
        )
        
        logger.info(f"✅ 数据集创建成功，样本数: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        input_data, target_data, time_steps = sample
        
        logger.info(f"✅ 数据加载成功")
        logger.info(f"   输入形状: {input_data.shape}")
        logger.info(f"   目标形状: {target_data.shape}")
        logger.info(f"   时间步形状: {time_steps.shape if time_steps is not None else None}")
        
        # 检查数据范围
        logger.info(f"   输入数据范围: [{input_data.min():.6f}, {input_data.max():.6f}]")
        logger.info(f"   目标数据范围: [{target_data.min():.6f}, {target_data.max():.6f}]")
        
        return True, dataset
        
    except Exception as e:
        logger.error(f"❌ 直接测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_darcy_dataloader():
    """测试DarcyFlow数据加载器"""
    logger.info("=== 测试DarcyFlow数据加载器 ===")
    
    try:
        # 加载配置
        config_path = "configs/darcy_flow_config.yaml"
        if not Path(config_path).exists():
            logger.error(f"配置文件不存在: {config_path}")
            return False
        
        config = load_config(config_path)
        
        # 创建数据加载器
        train_loader, valid_loader, test_loader = get_custom_loaders(config)
        
        logger.info(f"✅ 数据加载器创建成功")
        logger.info(f"   训练集批次数: {len(train_loader)}")
        logger.info(f"   验证集批次数: {len(valid_loader)}")
        logger.info(f"   测试集批次数: {len(test_loader)}")
        
        # 测试批次加载
        for batch_idx, (inputs, targets, time_steps) in enumerate(train_loader):
            logger.info(f"✅ 批次 {batch_idx} 加载成功")
            logger.info(f"   批次输入形状: {inputs.shape}")
            logger.info(f"   批次目标形状: {targets.shape}")
            logger.info(f"   批次时间步形状: {time_steps.shape if time_steps is not None else None}")
            
            if batch_idx >= 2:  # 只测试前3个批次
                break
        
        return True, (train_loader, valid_loader, test_loader)
        
    except Exception as e:
        logger.error(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_adaptive_loader():
    """测试自适应加载器"""
    logger.info("=== 测试自适应加载器 ===")
    
    try:
        # 加载配置
        config_path = "configs/darcy_flow_config.yaml"
        config = load_config(config_path)
        
        # 使用自适应加载器
        train_loader, valid_loader, test_loader = get_adaptive_loaders(config)
        
        logger.info(f"✅ 自适应加载器创建成功")
        logger.info(f"   训练集批次数: {len(train_loader)}")
        logger.info(f"   验证集批次数: {len(valid_loader)}")
        logger.info(f"   测试集批次数: {len(test_loader)}")
        
        # 获取数据信息
        if hasattr(train_loader, 'get_data_info'):
            data_info = train_loader.get_data_info()
            logger.info(f"✅ 数据信息获取成功:")
            logger.info(f"   输入维度: {data_info['input_dim']}")
            logger.info(f"   输出维度: {data_info['output_dim']}")
            logger.info(f"   序列长度: {data_info['sequence_length']}")
        
        return True, (train_loader, valid_loader, test_loader)
        
    except Exception as e:
        logger.error(f"❌ 自适应加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def visualize_darcy_data(dataset, num_samples=3):
    """可视化DarcyFlow数据"""
    logger.info("=== 可视化DarcyFlow数据 ===")
    
    try:
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
        fig.suptitle('DarcyFlow数据可视化：渗透率场(上) vs 压力场(下)', fontsize=14)
        
        for i in range(num_samples):
            # 获取样本
            input_data, target_data, _ = dataset[i]
            
            # 重塑为2D
            input_2d = input_data[0].reshape(128, 128)  # 第一个时间步
            target_2d = target_data[0].reshape(128, 128)  # 第一个时间步
            
            # 绘制输入（渗透率场）
            im1 = axes[0, i].imshow(input_2d, cmap='viridis', aspect='equal')
            axes[0, i].set_title(f'样本 {i+1}: 渗透率场 (nu)')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # 绘制目标（压力场）
            im2 = axes[1, i].imshow(target_2d, cmap='plasma', aspect='equal')
            axes[1, i].set_title(f'样本 {i+1}: 压力场 (tensor)')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = "darcy_flow_visualization.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ 可视化图像已保存: {save_path}")
        
        # 显示图像（如果在交互环境中）
        try:
            plt.show()
        except:
            pass
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_data_statistics(dataset):
    """分析数据统计信息"""
    logger.info("=== 分析数据统计信息 ===")
    
    try:
        # 收集统计信息
        input_stats = []
        target_stats = []
        
        num_samples = min(100, len(dataset))  # 分析前100个样本
        
        for i in range(num_samples):
            input_data, target_data, _ = dataset[i]
            
            input_stats.append({
                'min': input_data.min().item(),
                'max': input_data.max().item(),
                'mean': input_data.mean().item(),
                'std': input_data.std().item()
            })
            
            target_stats.append({
                'min': target_data.min().item(),
                'max': target_data.max().item(),
                'mean': target_data.mean().item(),
                'std': target_data.std().item()
            })
        
        # 计算总体统计
        input_mins = [s['min'] for s in input_stats]
        input_maxs = [s['max'] for s in input_stats]
        input_means = [s['mean'] for s in input_stats]
        input_stds = [s['std'] for s in input_stats]
        
        target_mins = [s['min'] for s in target_stats]
        target_maxs = [s['max'] for s in target_stats]
        target_means = [s['mean'] for s in target_stats]
        target_stds = [s['std'] for s in target_stats]
        
        logger.info(f"✅ 输入数据统计 (渗透率场):")
        logger.info(f"   全局范围: [{min(input_mins):.6f}, {max(input_maxs):.6f}]")
        logger.info(f"   平均值范围: [{min(input_means):.6f}, {max(input_means):.6f}]")
        logger.info(f"   标准差范围: [{min(input_stds):.6f}, {max(input_stds):.6f}]")
        
        logger.info(f"✅ 目标数据统计 (压力场):")
        logger.info(f"   全局范围: [{min(target_mins):.6f}, {max(target_maxs):.6f}]")
        logger.info(f"   平均值范围: [{min(target_means):.6f}, {max(target_means):.6f}]")
        logger.info(f"   标准差范围: [{min(target_stds):.6f}, {max(target_stds):.6f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 统计分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    logger.info("🔍 DarcyFlow数据集测试开始")
    logger.info("=" * 60)
    
    # 检查文件是否存在
    data_file = "PDEBench/pdebench/data_download/data/2D/DarcyFlow/2D_DarcyFlow_beta0.1_Train.hdf5"
    if not Path(data_file).exists():
        logger.error(f"❌ 数据文件不存在: {data_file}")
        return False
    
    # 检查配置文件是否存在
    config_file = "configs/darcy_flow_config.yaml"
    if not Path(config_file).exists():
        logger.error(f"❌ 配置文件不存在: {config_file}")
        return False
    
    results = []
    
    # 1. 直接测试数据集
    success, dataset = test_darcy_dataset_direct()
    results.append(("直接测试CustomDataset", success))
    
    if success and dataset:
        # 2. 分析数据统计
        success_stats = analyze_data_statistics(dataset)
        results.append(("数据统计分析", success_stats))
        
        # 3. 可视化数据
        success_viz = visualize_darcy_data(dataset)
        results.append(("数据可视化", success_viz))
    
    # 4. 测试数据加载器
    success_loader, loaders = test_darcy_dataloader()
    results.append(("测试数据加载器", success_loader))
    
    # 5. 测试自适应加载器
    success_adaptive, adaptive_loaders = test_adaptive_loader()
    results.append(("测试自适应加载器", success_adaptive))
    
    # 输出测试结果
    logger.info("\n=== 测试结果汇总 ===")
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！DarcyFlow数据集可以正常使用。")
        logger.info("\n📝 下一步:")
        logger.info("1. 使用 configs/darcy_flow_config.yaml 配置文件")
        logger.info("2. 运行 python examples/custom_dataset_example.py 开始训练")
        logger.info("3. 或者使用 python main.py --config configs/darcy_flow_config.yaml")
        return True
    else:
        logger.error("❌ 部分测试失败，请检查错误信息。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)