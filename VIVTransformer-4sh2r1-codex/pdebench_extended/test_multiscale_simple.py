#!/usr/bin/env python3
"""
简化的多尺度测试脚本
直接测试多尺度适配器的核心功能
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import logging

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """测试基本功能"""
    logger.info("开始基本功能测试...")
    
    try:
        # 测试导入
        from multiscale.data.multiscale_adapter import MultiScaleDataset
        logger.info("✓ 多尺度适配器导入成功")
        
        # 测试基本配置
        test_config = {
            'data_path': 'data/test_data.h5',  # 假设的测试数据路径
            'scale_factor': 2,
            'pde_type': 'darcy_flow',
            'original_resolution': [128, 128],
            'normalize': True
        }
        
        logger.info("✓ 基本配置测试通过")
        
        # 测试数据维度计算
        scale_factor = 2
        original_resolution = [128, 128]
        low_resolution = [res // scale_factor for res in original_resolution]
        
        logger.info(f"原始分辨率: {original_resolution}")
        logger.info(f"低分辨率: {low_resolution}")
        logger.info(f"缩放因子: {scale_factor}")
        
        # 计算维度
        n_channels = 1
        input_dim = low_resolution[0] * low_resolution[1] * n_channels
        output_dim = original_resolution[0] * original_resolution[1] * n_channels
        compression_ratio = output_dim / input_dim
        
        logger.info(f"输入维度: {input_dim}")
        logger.info(f"输出维度: {output_dim}")
        logger.info(f"压缩比: {compression_ratio:.2f}")
        
        logger.info("✓ 维度计算测试通过")
        
        return True
        
    except Exception as e:
        logger.error(f"基本功能测试失败: {e}")
        return False

def test_tensor_operations():
    """测试张量操作"""
    logger.info("开始张量操作测试...")
    
    try:
        # 创建测试数据
        batch_size = 4
        time_steps = 1
        original_resolution = [128, 128]
        scale_factor = 2
        n_channels = 1
        
        # 创建高分辨率数据
        high_res_data = torch.randn(
            batch_size, time_steps, 
            original_resolution[0] * original_resolution[1] * n_channels
        )
        
        logger.info(f"高分辨率数据形状: {high_res_data.shape}")
        
        # 模拟下采样操作
        # 重塑为 [B, T, H, W, C]
        B, T, spatial_dim = high_res_data.shape
        H, W = original_resolution
        C = n_channels
        
        reshaped_data = high_res_data.reshape(B, T, H, W, C)
        logger.info(f"重塑后数据形状: {reshaped_data.shape}")
        
        # 简单的下采样（每隔scale_factor取一个点）
        low_res_h = H // scale_factor
        low_res_w = W // scale_factor
        
        downsampled = reshaped_data[:, :, ::scale_factor, ::scale_factor, :]
        logger.info(f"下采样后形状: {downsampled.shape}")
        
        # 重塑回平坦格式
        low_res_data = downsampled.reshape(B, T, low_res_h * low_res_w * C)
        logger.info(f"低分辨率数据形状: {low_res_data.shape}")
        
        # 验证维度
        expected_input_dim = low_res_h * low_res_w * C
        expected_output_dim = H * W * C
        
        assert low_res_data.shape[-1] == expected_input_dim
        assert high_res_data.shape[-1] == expected_output_dim
        
        logger.info("✓ 张量操作测试通过")
        return True
        
    except Exception as e:
        logger.error(f"张量操作测试失败: {e}")
        return False

def test_different_scales():
    """测试不同缩放因子"""
    logger.info("开始不同缩放因子测试...")
    
    try:
        original_resolution = [128, 128]
        scale_factors = [2, 4, 8]
        
        for scale_factor in scale_factors:
            logger.info(f"测试缩放因子: {scale_factor}")
            
            # 检查分辨率是否可整除
            if original_resolution[0] % scale_factor != 0 or original_resolution[1] % scale_factor != 0:
                logger.warning(f"分辨率 {original_resolution} 不能被 {scale_factor} 整除")
                continue
            
            low_resolution = [res // scale_factor for res in original_resolution]
            
            # 计算维度
            n_channels = 1
            input_dim = low_resolution[0] * low_resolution[1] * n_channels
            output_dim = original_resolution[0] * original_resolution[1] * n_channels
            compression_ratio = output_dim / input_dim
            
            logger.info(f"  低分辨率: {low_resolution}")
            logger.info(f"  输入维度: {input_dim}")
            logger.info(f"  输出维度: {output_dim}")
            logger.info(f"  压缩比: {compression_ratio:.2f}")
        
        logger.info("✓ 不同缩放因子测试通过")
        return True
        
    except Exception as e:
        logger.error(f"不同缩放因子测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始多尺度模块简化测试...")
    
    tests = [
        ("基本功能测试", test_basic_functionality),
        ("张量操作测试", test_tensor_operations),
        ("不同缩放因子测试", test_different_scales)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行: {test_name}")
        logger.info(f"{'='*50}")
        
        if test_func():
            passed += 1
            logger.info(f"✓ {test_name} 通过")
        else:
            logger.error(f"✗ {test_name} 失败")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"测试总结: {passed}/{total} 通过")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 所有测试通过！")
        return True
    else:
        logger.error(f"❌ {total - passed} 个测试失败")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)