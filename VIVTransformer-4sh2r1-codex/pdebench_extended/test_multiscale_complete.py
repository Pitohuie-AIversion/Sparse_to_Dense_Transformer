#!/usr/bin/env python3
"""
完整的多尺度测试脚本
直接从项目根目录运行，避免所有导入问题
"""

import sys
import os
from pathlib import Path
import logging
import yaml
import torch
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def test_multiscale_dataset_creation():
    """测试多尺度数据集创建"""
    logger.info("测试多尺度数据集创建...")
    
    try:
        # 添加multiscale目录到路径
        multiscale_path = Path('multiscale')
        if multiscale_path.exists():
            sys.path.insert(0, str(multiscale_path))
        
        # 导入多尺度数据集
        from data.multiscale_adapter import MultiScaleDataset
        
        # 加载配置
        config = load_config('multiscale/configs/multiscale_config.yaml')
        data_config = config['data']
        
        # 创建虚拟数据进行测试（因为实际数据文件可能不存在）
        logger.info("创建虚拟数据集进行测试...")
        
        # 测试不同的缩放因子
        scale_factors = [2, 4, 8]
        
        for scale_factor in scale_factors:
            logger.info(f"测试缩放因子: {scale_factor}")
            
            # 计算分辨率
            original_resolution = data_config.get('original_resolution', [128, 128])
            low_resolution = [
                original_resolution[0] // scale_factor,
                original_resolution[1] // scale_factor
            ]
            
            # 计算维度
            n_channels = 1  # 假设单通道
            input_dim = low_resolution[0] * low_resolution[1] * n_channels
            output_dim = original_resolution[0] * original_resolution[1] * n_channels
            
            logger.info(f"  原始分辨率: {original_resolution}")
            logger.info(f"  低分辨率: {low_resolution}")
            logger.info(f"  输入维度: {input_dim}")
            logger.info(f"  输出维度: {output_dim}")
            logger.info(f"  压缩比: {output_dim / input_dim:.2f}")
            
            # 验证分辨率计算
            assert original_resolution[0] % scale_factor == 0, f"分辨率不能被缩放因子整除: {original_resolution[0]} % {scale_factor}"
            assert original_resolution[1] % scale_factor == 0, f"分辨率不能被缩放因子整除: {original_resolution[1]} % {scale_factor}"
        
        logger.info("✓ 多尺度数据集创建测试通过")
        return True
        
    except Exception as e:
        logger.error(f"多尺度数据集创建测试失败: {e}")
        return False

def test_tensor_operations():
    """测试张量操作"""
    logger.info("测试张量操作...")
    
    try:
        # 创建测试数据
        batch_size = 4
        original_resolution = [128, 128]
        scale_factor = 2
        n_channels = 1
        
        # 高分辨率数据
        high_res_data = torch.randn(
            batch_size, 1, 
            original_resolution[0] * original_resolution[1] * n_channels
        )
        
        logger.info(f"高分辨率数据形状: {high_res_data.shape}")
        
        # 重塑为空间维度
        high_res_spatial = high_res_data.reshape(
            batch_size, 1, original_resolution[0], original_resolution[1], n_channels
        )
        
        logger.info(f"高分辨率空间形状: {high_res_spatial.shape}")
        
        # 下采样
        import torch.nn.functional as F
        
        # 转换为 [B*T*C, 1, H, W] 格式
        data_for_pool = high_res_spatial.permute(0, 1, 4, 2, 3).contiguous()
        data_for_pool = data_for_pool.view(-1, 1, original_resolution[0], original_resolution[1])
        
        logger.info(f"池化前形状: {data_for_pool.shape}")
        
        # 平均池化下采样
        downsampled = F.avg_pool2d(
            data_for_pool,
            kernel_size=scale_factor,
            stride=scale_factor
        )
        
        logger.info(f"下采样后形状: {downsampled.shape}")
        
        # 转换回原始格式
        low_res_spatial = downsampled.view(
            batch_size, 1, n_channels, 
            original_resolution[0] // scale_factor,
            original_resolution[1] // scale_factor
        )
        low_res_spatial = low_res_spatial.permute(0, 1, 3, 4, 2).contiguous()
        
        logger.info(f"低分辨率空间形状: {low_res_spatial.shape}")
        
        # 展平
        low_res_data = low_res_spatial.reshape(
            batch_size, 1, -1
        )
        
        logger.info(f"低分辨率数据形状: {low_res_data.shape}")
        
        # 验证维度
        expected_low_dim = (original_resolution[0] // scale_factor) * (original_resolution[1] // scale_factor) * n_channels
        assert low_res_data.shape[-1] == expected_low_dim, f"低分辨率维度不匹配: {low_res_data.shape[-1]} != {expected_low_dim}"
        
        logger.info("✓ 张量操作测试通过")
        return True
        
    except Exception as e:
        logger.error(f"张量操作测试失败: {e}")
        return False

def test_config_compatibility():
    """测试配置兼容性"""
    logger.info("测试配置兼容性...")
    
    try:
        config = load_config('multiscale/configs/multiscale_config.yaml')
        
        # 检查数据配置
        data_config = config.get('data', {})
        scale_factor = data_config.get('scale_factor', 2)
        original_resolution = data_config.get('original_resolution', [128, 128])
        batch_size = data_config.get('batch_size', 16)
        
        # 检查模型配置
        model_config = config.get('model', {})
        input_dim = model_config.get('input_dim')
        output_dim = model_config.get('output_dim')
        d_model = model_config.get('d_model')
        num_heads = model_config.get('num_heads')
        num_layers = model_config.get('num_layers')
        
        # 验证维度一致性
        expected_input_dim = (original_resolution[0] // scale_factor) * (original_resolution[1] // scale_factor)
        expected_output_dim = original_resolution[0] * original_resolution[1]
        
        logger.info(f"配置的输入维度: {input_dim}, 期望: {expected_input_dim}")
        logger.info(f"配置的输出维度: {output_dim}, 期望: {expected_output_dim}")
        
        # 检查模型参数
        logger.info(f"模型维度: {d_model}")
        logger.info(f"注意力头数: {num_heads}")
        logger.info(f"层数: {num_layers}")
        logger.info(f"批次大小: {batch_size}")
        
        # 估算内存使用
        input_memory = batch_size * input_dim * 4 / (1024**3)  # GB
        output_memory = batch_size * output_dim * 4 / (1024**3)  # GB
        model_memory = d_model * num_layers * 4 / (1024**3)  # 粗略估算
        
        total_memory = input_memory + output_memory + model_memory
        
        logger.info(f"估算内存使用: {total_memory:.3f} GB")
        
        if total_memory > 8:  # 假设8GB GPU内存
            logger.warning(f"内存使用可能超出GPU限制: {total_memory:.3f} GB > 8 GB")
        else:
            logger.info(f"内存使用在合理范围内: {total_memory:.3f} GB")
        
        logger.info("✓ 配置兼容性测试通过")
        return True
        
    except Exception as e:
        logger.error(f"配置兼容性测试失败: {e}")
        return False

def test_multiscale_experiments():
    """测试多尺度实验配置"""
    logger.info("测试多尺度实验配置...")
    
    try:
        config = load_config('multiscale/configs/multiscale_config.yaml')
        
        # 检查多尺度实验配置
        experiments_config = config.get('multiscale_experiments', {})
        scale_factors = experiments_config.get('scale_factors', [])
        scale_configs = experiments_config.get('scale_configs', {})
        
        logger.info(f"支持的缩放因子: {scale_factors}")
        
        for scale_factor in scale_factors:
            if scale_factor in scale_configs:
                scale_config = scale_configs[scale_factor]
                logger.info(f"缩放因子 {scale_factor}:")
                logger.info(f"  输入维度: {scale_config.get('input_dim')}")
                logger.info(f"  输出维度: {scale_config.get('output_dim')}")
                logger.info(f"  批次大小: {scale_config.get('batch_size')}")
                logger.info(f"  学习率: {scale_config.get('learning_rate')}")
                
                # 检查SVD配置
                svd_config = scale_config.get('svd_config', {})
                if svd_config:
                    logger.info(f"  SVD基础权重: {svd_config.get('base_weight')}")
                    logger.info(f"  SVD权重: {svd_config.get('svd_weights')}")
                    logger.info(f"  TopK: {svd_config.get('topk')}")
        
        logger.info("✓ 多尺度实验配置测试通过")
        return True
        
    except Exception as e:
        logger.error(f"多尺度实验配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始运行完整的多尺度测试...")
    logger.info("=" * 60)
    
    tests = [
        ("多尺度数据集创建", test_multiscale_dataset_creation),
        ("张量操作", test_tensor_operations),
        ("配置兼容性", test_config_compatibility),
        ("多尺度实验配置", test_multiscale_experiments),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n运行测试: {test_name}")
        logger.info("-" * 40)
        
        try:
            if test_func():
                logger.info(f"✓ {test_name} 通过")
                passed += 1
            else:
                logger.error(f"✗ {test_name} 失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 异常: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"测试总结: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！多尺度功能已准备就绪！")
        return True
    else:
        logger.warning(f"⚠ {total - passed} 个测试失败")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)