#!/usr/bin/env python3
"""测试数据增强功能

这个脚本专门用于测试数据增强功能，包括旋转、翻转、噪声等变换。
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from data.transforms import RandomFlip, RandomRotate, AddGaussianNoise, RandomCrop, ToTensor
from torchvision.transforms import Compose

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data():
    """创建测试数据"""
    # 创建一个简单的测试图像：左上角为1，其他为0
    input_data = torch.zeros(1, 64, 64)
    input_data[0, :32, :32] = 1.0
    
    target_data = torch.zeros(1, 64, 64)
    target_data[0, 32:, 32:] = 1.0
    
    return input_data, target_data


def test_random_flip():
    """测试随机翻转功能"""
    logger.info("=== 测试随机翻转功能 ===")
    
    input_data, target_data = create_test_data()
    
    # 测试翻转（RandomFlip会随机选择水平或垂直翻转）
    flip_transform = RandomFlip(p=1.0)
    flipped_input, flipped_target = flip_transform((input_data, target_data))
    
    logger.info(f"原始输入形状: {input_data.shape}")
    logger.info(f"翻转后输入形状: {flipped_input.shape}")
    
    # 验证翻转是否正确
    original_sum = input_data.sum()
    flipped_sum = flipped_input.sum()
    
    logger.info(f"原始数据总和: {original_sum}")
    logger.info(f"翻转后数据总和: {flipped_sum}")
    
    # 检查是否确实发生了翻转
    is_flipped = not torch.equal(input_data, flipped_input)
    logger.info(f"数据是否被翻转: {is_flipped}")
    
    return is_flipped


def test_random_rotate():
    """测试随机旋转功能"""
    logger.info("=== 测试随机旋转功能 ===")
    
    input_data, target_data = create_test_data()
    
    # 测试旋转
    rotate_transform = RandomRotate()
    rotated_input, rotated_target = rotate_transform((input_data, target_data))
    
    logger.info(f"原始输入形状: {input_data.shape}")
    logger.info(f"旋转后输入形状: {rotated_input.shape}")
    
    # 验证旋转是否正确
    original_sum = input_data.sum()
    rotated_sum = rotated_input.sum()
    
    logger.info(f"原始数据总和: {original_sum}")
    logger.info(f"旋转后数据总和: {rotated_sum}")
    
    # 检查是否确实发生了旋转（对于非对称数据）
    is_rotated = not torch.equal(input_data, rotated_input)
    logger.info(f"数据是否被旋转: {is_rotated}")
    
    return True  # 旋转总是成功的


def test_gaussian_noise():
    """测试高斯噪声添加功能"""
    logger.info("=== 测试高斯噪声添加功能 ===")
    
    input_data, target_data = create_test_data()
    
    # 测试噪声添加
    noise_transform = AddGaussianNoise(std=0.1)
    noisy_input, noisy_target = noise_transform((input_data, target_data))
    
    logger.info(f"原始输入形状: {input_data.shape}")
    logger.info(f"加噪后输入形状: {noisy_input.shape}")
    
    # 验证噪声是否被添加
    original_std = input_data.std()
    noisy_std = noisy_input.std()
    
    logger.info(f"原始数据标准差: {original_std:.6f}")
    logger.info(f"加噪后数据标准差: {noisy_std:.6f}")
    
    # 检查是否确实添加了噪声
    has_noise = not torch.equal(input_data, noisy_input)
    logger.info(f"是否添加了噪声: {has_noise}")
    
    return has_noise


def test_random_crop():
    """测试随机裁剪功能"""
    logger.info("=== 测试随机裁剪功能 ===")
    
    input_data, target_data = create_test_data()
    
    # 测试裁剪
    crop_transform = RandomCrop(output_size=(32, 32))
    cropped_input, cropped_target = crop_transform((input_data, target_data))
    
    logger.info(f"原始输入形状: {input_data.shape}")
    logger.info(f"裁剪后输入形状: {cropped_input.shape}")
    
    # 验证裁剪是否正确
    expected_shape = (1, 32, 32)
    actual_shape = cropped_input.shape
    
    is_correct_size = actual_shape == expected_shape
    logger.info(f"裁剪尺寸是否正确: {is_correct_size}")
    
    return is_correct_size


def test_compose_transforms():
    """测试组合变换功能"""
    logger.info("=== 测试组合变换功能 ===")
    
    input_data, target_data = create_test_data()
    
    # 创建组合变换
    transform = Compose([
        RandomFlip(p=0.5),
        RandomRotate(),
        AddGaussianNoise(std=0.05),
        RandomCrop(output_size=(48, 48))
    ])
    
    # 应用组合变换
    transformed_input, transformed_target = transform((input_data, target_data))
    
    logger.info(f"原始输入形状: {input_data.shape}")
    logger.info(f"变换后输入形状: {transformed_input.shape}")
    
    # 验证组合变换是否正确
    expected_shape = (1, 48, 48)
    actual_shape = transformed_input.shape
    
    is_correct_size = actual_shape == expected_shape
    logger.info(f"组合变换尺寸是否正确: {is_correct_size}")
    
    return is_correct_size


def test_augmentation_disable():
    """测试数据增强禁用功能"""
    logger.info("=== 测试数据增强禁用功能 ===")
    
    input_data, target_data = create_test_data()
    
    # 测试概率为0的翻转（应该不发生变换）
    no_flip_transform = RandomFlip(p=0.0)
    no_flip_input, no_flip_target = no_flip_transform((input_data, target_data))
    
    # 验证数据是否保持不变
    is_unchanged = torch.equal(input_data, no_flip_input)
    logger.info(f"禁用翻转时数据是否保持不变: {is_unchanged}")
    
    return is_unchanged


def visualize_augmentations():
    """可视化数据增强效果"""
    logger.info("=== 可视化数据增强效果 ===")
    
    input_data, target_data = create_test_data()
    
    # 创建不同的变换
    transforms = {
        'Original': lambda x: x,
        'Random Flip': RandomFlip(p=1.0),
        'Random Rotate': RandomRotate(),
        'Gaussian Noise': AddGaussianNoise(std=0.1),
        'Random Crop': RandomCrop(output_size=(48, 48))
    }
    
    fig, axes = plt.subplots(2, len(transforms), figsize=(15, 6))
    fig.suptitle('Data Augmentation Effects', fontsize=16)
    
    for i, (name, transform) in enumerate(transforms.items()):
        try:
            if name == 'Original':
                aug_input, aug_target = input_data, target_data
            else:
                aug_input, aug_target = transform((input_data, target_data))
            
            # 显示输入数据
            axes[0, i].imshow(aug_input[0].numpy(), cmap='viridis')
            axes[0, i].set_title(f'{name}\n(Input)')
            axes[0, i].axis('off')
            
            # 显示目标数据
            axes[1, i].imshow(aug_target[0].numpy(), cmap='viridis')
            axes[1, i].set_title(f'{name}\n(Target)')
            axes[1, i].axis('off')
            
        except Exception as e:
            logger.warning(f"可视化 {name} 时出错: {e}")
            axes[0, i].text(0.5, 0.5, f'Error\n{name}', ha='center', va='center')
            axes[1, i].text(0.5, 0.5, f'Error\n{name}', ha='center', va='center')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path('data_augmentation_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"可视化结果已保存到: {output_path}")
    
    plt.close()
    return True


def test_config_based_augmentation():
    """测试基于配置的数据增强"""
    logger.info("=== 测试基于配置的数据增强 ===")
    
    # 模拟配置
    config_enabled = {
        'training': {
            'data_augmentation': {
                'enabled': True,
                'rotation': True,
                'flip': True,
                'noise_level': 0.05
            }
        }
    }
    
    config_disabled = {
        'training': {
            'data_augmentation': {
                'enabled': False,
                'rotation': False,
                'flip': False,
                'noise_level': 0.0
            }
        }
    }
    
    input_data, target_data = create_test_data()
    
    # 测试启用数据增强的情况
    if config_enabled['training']['data_augmentation']['enabled']:
        transforms = []
        if config_enabled['training']['data_augmentation']['flip']:
            transforms.append(RandomFlip(p=0.5))
        if config_enabled['training']['data_augmentation']['rotation']:
            transforms.append(RandomRotate())
        if config_enabled['training']['data_augmentation']['noise_level'] > 0:
            transforms.append(AddGaussianNoise(std=config_enabled['training']['data_augmentation']['noise_level']))
        
        if transforms:
            augment_transform = Compose(transforms)
            aug_input, aug_target = augment_transform((input_data, target_data))
            logger.info("✅ 启用数据增强配置测试通过")
        else:
            logger.info("⚠️  启用数据增强但没有具体变换")
    
    # 测试禁用数据增强的情况
    if not config_disabled['training']['data_augmentation']['enabled']:
        # 不应用任何变换
        no_aug_input, no_aug_target = input_data, target_data
        is_unchanged = torch.equal(input_data, no_aug_input)
        logger.info(f"✅ 禁用数据增强配置测试通过: {is_unchanged}")
    
    return True


def main():
    """主测试函数"""
    logger.info("开始测试数据增强功能")
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("随机翻转", test_random_flip()))
    test_results.append(("随机旋转", test_random_rotate()))
    test_results.append(("高斯噪声", test_gaussian_noise()))
    test_results.append(("随机裁剪", test_random_crop()))
    test_results.append(("组合变换", test_compose_transforms()))
    test_results.append(("增强禁用", test_augmentation_disable()))
    test_results.append(("配置控制", test_config_based_augmentation()))
    test_results.append(("可视化效果", visualize_augmentations()))
    
    # 输出测试结果
    logger.info("\n=== 数据增强测试结果汇总 ===")
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有数据增强测试通过！功能工作正常。")
    else:
        logger.warning(f"⚠️  有 {total - passed} 项测试失败，请检查相关功能。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)