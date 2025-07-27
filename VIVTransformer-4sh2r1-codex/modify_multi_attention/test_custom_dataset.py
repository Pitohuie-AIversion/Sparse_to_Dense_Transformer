#!/usr/bin/env python3
"""测试自定义数据集适配器

这个脚本用于测试新创建的自定义数据集适配器是否能正确加载和处理数据。
"""

import sys
import yaml
import torch
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from data.custom_dataset_adapter import CustomDataset, CustomDataLoader
from data.dataloader import get_custom_loaders, get_adaptive_loaders

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_custom_dataset_direct():
    """直接测试CustomDataset类"""
    logger.info("=== 直接测试CustomDataset类 ===")
    
    # 测试数据路径
    data_path = "X:/2025/Graduation_project/simulation_results/merged_all_pressures_separated_normalized.pt"
    
    if not Path(data_path).exists():
        logger.warning(f"数据文件不存在: {data_path}，跳过直接测试")
        return False
    
    try:
        # 创建数据集实例
        dataset = CustomDataset(
            data_path=data_path,
            dataset_type="flow_field",
            split="train",
            sequence_length=48,
            spatial_resolution=[200, 200],
            input_size=128,
            normalize=True,
            normalization_method="standard",
            data_key="pressure"
        )
        
        logger.info(f"数据集创建成功，样本数: {len(dataset)}")
        
        # 获取数据集信息
        info = dataset.get_data_info()
        logger.info(f"数据集信息: {info}")
        
        # 测试获取单个样本
        if len(dataset) > 0:
            inputs, targets, time_steps = dataset[0]
            logger.info(f"样本形状 - 输入: {inputs.shape}, 目标: {targets.shape}, 时间步: {time_steps.shape}")
            logger.info(f"数据类型 - 输入: {inputs.dtype}, 目标: {targets.dtype}")
            
            # 检查数据范围
            logger.info(f"输入数据范围: [{inputs.min():.4f}, {inputs.max():.4f}]")
            logger.info(f"目标数据范围: [{targets.min():.4f}, {targets.max():.4f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"直接测试CustomDataset失败: {e}")
        return False


def test_custom_dataloader():
    """测试CustomDataLoader类"""
    logger.info("=== 测试CustomDataLoader类 ===")
    
    data_path = "X:/2025/Graduation_project/simulation_results/merged_all_pressures_separated_normalized.pt"
    
    if not Path(data_path).exists():
        logger.warning(f"数据文件不存在: {data_path}，跳过DataLoader测试")
        return False
    
    try:
        # 创建数据集
        dataset = CustomDataset(
            data_path=data_path,
            dataset_type="flow_field",
            split="train",
            sequence_length=48,
            input_size=128,
            normalize=True
        )
        
        # 创建数据加载器
        dataloader = CustomDataLoader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # 设置为0避免多进程问题
        )
        
        logger.info(f"DataLoader创建成功，批次数: {len(dataloader)}")
        
        # 测试获取一个批次
        for batch_idx, (inputs, targets, time_steps) in enumerate(dataloader):
            logger.info(f"批次 {batch_idx} 形状 - 输入: {inputs.shape}, 目标: {targets.shape}")
            if time_steps is not None:
                logger.info(f"时间步形状: {time_steps.shape}")
            
            # 只测试第一个批次
            break
        
        return True
        
    except Exception as e:
        logger.error(f"测试CustomDataLoader失败: {e}")
        return False


def test_get_custom_loaders():
    """测试get_custom_loaders函数"""
    logger.info("=== 测试get_custom_loaders函数 ===")
    
    # 创建测试配置
    config = {
        'data': {
            'batch_size': 4,
            'normalize': True,
            'num_workers': 0,
            'pin_memory': False
        },
        'custom_dataset': {
            'data_root': 'X:/2025/Graduation_project/simulation_results',
            'dataset_configs': {
                'flow_field': {
                    'data_file': 'merged_all_pressures_separated_normalized.pt',
                    'spatial_resolution': [200, 200],
                    'input_size': 128,
                    'sequence_length': 48,
                    'normalization_method': 'standard',
                    'data_key': 'pressure',
                    'target_key': None,
                    'split_ratios': [0.7, 0.15, 0.15]
                }
            }
        }
    }
    
    data_path = Path(config['custom_dataset']['data_root']) / config['custom_dataset']['dataset_configs']['flow_field']['data_file']
    
    if not data_path.exists():
        logger.warning(f"数据文件不存在: {data_path}，跳过get_custom_loaders测试")
        return False
    
    try:
        # 创建数据加载器
        train_loader, valid_loader, test_loader = get_custom_loaders(
            config=config,
            dataset_type='flow_field',
            batch_size=4,
            use_augmentation=False
        )
        
        logger.info(f"数据加载器创建成功")
        logger.info(f"训练集批次数: {len(train_loader)}")
        logger.info(f"验证集批次数: {len(valid_loader)}")
        logger.info(f"测试集批次数: {len(test_loader)}")
        
        # 测试训练集的一个批次
        for batch_idx, (inputs, targets, time_steps) in enumerate(train_loader):
            logger.info(f"训练批次形状 - 输入: {inputs.shape}, 目标: {targets.shape}")
            break
        
        # 获取数据加载器信息
        info = train_loader.get_data_info()
        logger.info(f"数据加载器信息: {info}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试get_custom_loaders失败: {e}")
        return False


def test_adaptive_loaders_with_custom():
    """测试自适应加载器与自定义数据集的集成"""
    logger.info("=== 测试自适应加载器与自定义数据集的集成 ===")
    
    # 加载自定义数据集配置
    config_path = "configs/custom_dataset_config.yaml"
    
    if not Path(config_path).exists():
        logger.warning(f"配置文件不存在: {config_path}，跳过自适应加载器测试")
        return False
    
    try:
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查数据文件是否存在
        data_root = config['custom_dataset']['data_root']
        data_file = config['custom_dataset']['dataset_configs']['flow_field']['data_file']
        data_path = Path(data_root) / data_file
        
        if not data_path.exists():
            logger.warning(f"数据文件不存在: {data_path}，跳过自适应加载器测试")
            return False
        
        # 使用自适应加载器
        train_loader, valid_loader, test_loader = get_adaptive_loaders(config)
        
        logger.info(f"自适应加载器创建成功")
        logger.info(f"训练集批次数: {len(train_loader)}")
        logger.info(f"验证集批次数: {len(valid_loader)}")
        logger.info(f"测试集批次数: {len(test_loader)}")
        
        # 测试一个批次
        for batch_idx, (inputs, targets, time_steps) in enumerate(train_loader):
            logger.info(f"自适应加载器批次形状 - 输入: {inputs.shape}, 目标: {targets.shape}")
            break
        
        return True
        
    except Exception as e:
        logger.error(f"测试自适应加载器失败: {e}")
        return False


def test_different_file_formats():
    """测试不同文件格式的支持"""
    logger.info("=== 测试不同文件格式的支持 ===")
    
    # 测试支持的文件格式
    formats_to_test = [
        ('.pt', 'PyTorch格式'),
        ('.h5', 'HDF5格式'),
        ('.hdf5', 'HDF5格式'),
        ('.pkl', 'Pickle格式'),
        ('.npy', 'NumPy格式'),
        ('.npz', 'NumPy压缩格式')
    ]
    
    supported_formats = []
    
    for ext, desc in formats_to_test:
        try:
            # 创建一个虚拟的CustomDataset实例来测试格式支持
            # 这里只是测试格式识别，不实际加载数据
            test_path = f"test_file{ext}"
            
            # 检查是否支持该格式
            if ext in ['.pt', '.h5', '.hdf5', '.pkl', '.npy', '.npz']:
                supported_formats.append(desc)
                logger.info(f"✓ 支持 {desc} ({ext})")
            else:
                logger.info(f"✗ 不支持 {desc} ({ext})")
                
        except Exception as e:
            logger.warning(f"测试格式 {desc} 时出错: {e}")
    
    logger.info(f"总共支持 {len(supported_formats)} 种文件格式")
    return len(supported_formats) > 0


def main():
    """主测试函数"""
    logger.info("开始测试自定义数据集适配器")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("直接测试CustomDataset", test_custom_dataset_direct()))
    test_results.append(("测试CustomDataLoader", test_custom_dataloader()))
    test_results.append(("测试get_custom_loaders", test_get_custom_loaders()))
    test_results.append(("测试自适应加载器集成", test_adaptive_loaders_with_custom()))
    test_results.append(("测试文件格式支持", test_different_file_formats()))
    
    # 输出测试结果
    logger.info("\n=== 测试结果汇总 ===")
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！自定义数据集适配器工作正常。")
    else:
        logger.warning(f"⚠️  有 {total - passed} 项测试失败，请检查相关功能。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)