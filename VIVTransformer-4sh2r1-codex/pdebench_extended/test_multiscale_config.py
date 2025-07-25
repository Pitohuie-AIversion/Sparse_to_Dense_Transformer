#!/usr/bin/env python3
"""
多尺度配置文件测试脚本
测试multiscale_config.yaml的加载和验证
"""

import sys
import os
from pathlib import Path
import yaml
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_loading():
    """测试配置文件加载"""
    logger.info("开始配置文件加载测试...")
    
    try:
        config_path = 'multiscale/configs/multiscale_config.yaml'
        
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info("✓ 配置文件加载成功")
        logger.info(f"配置文件路径: {config_path}")
        
        return config
        
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        return False

def test_config_structure(config):
    """测试配置文件结构"""
    logger.info("开始配置文件结构测试...")
    
    try:
        # 检查必要的顶级键
        required_keys = ['global', 'model', 'data', 'training', 'loss', 'output']
        
        for key in required_keys:
            if key not in config:
                logger.error(f"缺少必要的配置键: {key}")
                return False
            logger.info(f"✓ 找到配置键: {key}")
        
        # 检查数据配置
        data_config = config.get('data', {})
        data_required = ['scale_factor', 'pde_type', 'batch_size', 'original_resolution']
        
        for key in data_required:
            if key not in data_config:
                logger.error(f"数据配置缺少必要键: {key}")
                return False
            logger.info(f"✓ 数据配置包含: {key} = {data_config[key]}")
        
        # 检查模型配置
        model_config = config.get('model', {})
        model_required = ['d_model', 'num_heads', 'num_layers']
        
        for key in model_required:
            if key not in model_config:
                print(f"❌ 模型配置缺少字段: {key}")
                return False
        
        print("✅ 模型配置字段完整")
        
        # 检查损失配置
        loss_config = config.get('loss', {})
        if 'svd_loss' in loss_config:
            svd_config = loss_config['svd_loss']
            logger.info(f"✓ SVD损失配置: enabled = {svd_config.get('enabled', False)}")
            if svd_config.get('enabled', False):
                logger.info(f"  SVD权重: {svd_config.get('weight', 0.1)}")
                logger.info(f"  SVD组件数: {svd_config.get('n_components', 10)}")
        
        logger.info("✓ 配置文件结构验证通过")
        return True
        
    except Exception as e:
        logger.error(f"配置文件结构测试失败: {e}")
        return False

def test_multiscale_specific_config(config):
    """测试多尺度特定配置"""
    logger.info("开始多尺度特定配置测试...")
    
    try:
        data_config = config.get('data', {})
        
        # 检查缩放因子
        scale_factor = data_config.get('scale_factor', 2)
        if scale_factor not in [2, 4, 8, 16]:
            logger.warning(f"缩放因子 {scale_factor} 可能不被支持")
        else:
            logger.info(f"✓ 缩放因子: {scale_factor}")
        
        # 检查分辨率
        original_resolution = data_config.get('original_resolution', [128, 128])
        logger.info(f"✓ 原始分辨率: {original_resolution}")
        
        # 计算低分辨率
        low_resolution = [res // scale_factor for res in original_resolution]
        logger.info(f"✓ 低分辨率: {low_resolution}")
        
        # 检查分辨率是否可整除
        for i, res in enumerate(original_resolution):
            if res % scale_factor != 0:
                logger.error(f"分辨率维度 {i} ({res}) 不能被缩放因子 {scale_factor} 整除")
                return False
        
        # 计算维度
        n_channels = data_config.get('n_channels', 1)
        input_dim = low_resolution[0] * low_resolution[1] * n_channels
        output_dim = original_resolution[0] * original_resolution[1] * n_channels
        compression_ratio = output_dim / input_dim
        
        logger.info(f"✓ 输入维度: {input_dim}")
        logger.info(f"✓ 输出维度: {output_dim}")
        logger.info(f"✓ 压缩比: {compression_ratio:.2f}")
        
        # 检查下采样方法
        downsampling_method = data_config.get('downsampling_method', 'average')
        supported_methods = ['average', 'bilinear', 'nearest']
        if downsampling_method not in supported_methods:
            logger.warning(f"下采样方法 {downsampling_method} 可能不被支持")
        else:
            logger.info(f"✓ 下采样方法: {downsampling_method}")
        
        logger.info("✓ 多尺度特定配置验证通过")
        return True
        
    except Exception as e:
        logger.error(f"多尺度特定配置测试失败: {e}")
        return False

def test_config_compatibility(config):
    """测试配置兼容性"""
    logger.info("开始配置兼容性测试...")
    
    try:
        # 检查模型维度与数据维度的兼容性
        data_config = config.get('data', {})
        model_config = config.get('model', {})
        
        scale_factor = data_config.get('scale_factor', 2)
        original_resolution = data_config.get('original_resolution', [128, 128])
        n_channels = data_config.get('n_channels', 1)
        
        low_resolution = [res // scale_factor for res in original_resolution]
        input_dim = low_resolution[0] * low_resolution[1] * n_channels
        output_dim = original_resolution[0] * original_resolution[1] * n_channels
        
        # 检查模型输入输出维度
        model_input_dim = model_config.get('input_dim')
        model_output_dim = model_config.get('output_dim')
        
        if model_input_dim and model_input_dim != input_dim:
            logger.warning(f"模型输入维度 {model_input_dim} 与计算的输入维度 {input_dim} 不匹配")
        else:
            logger.info(f"✓ 输入维度兼容: {input_dim}")
        
        if model_output_dim and model_output_dim != output_dim:
            logger.warning(f"模型输出维度 {model_output_dim} 与计算的输出维度 {output_dim} 不匹配")
        else:
            logger.info(f"✓ 输出维度兼容: {output_dim}")
        
        # 检查批次大小与硬件配置的兼容性
        batch_size = data_config.get('batch_size', 32)
        hardware_config = config.get('hardware', {})
        gpu_memory_gb = hardware_config.get('gpu_memory_gb', 8)
        
        # 估算内存使用（简化计算）
        estimated_memory_per_sample = (input_dim + output_dim) * 4 / (1024**3)  # 假设float32，转换为GB
        estimated_batch_memory = estimated_memory_per_sample * batch_size
        
        logger.info(f"✓ 批次大小: {batch_size}")
        logger.info(f"✓ 估算批次内存使用: {estimated_batch_memory:.2f} GB")
        logger.info(f"✓ GPU内存: {gpu_memory_gb} GB")
        
        if estimated_batch_memory > gpu_memory_gb * 0.8:  # 使用80%作为安全阈值
            logger.warning(f"批次内存使用可能超过GPU内存限制")
        
        logger.info("✓ 配置兼容性验证通过")
        return True
        
    except Exception as e:
        logger.error(f"配置兼容性测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始多尺度配置文件测试...")
    
    tests = [
        ("配置文件加载测试", test_config_loading),
        ("配置文件结构测试", lambda: test_config_structure(config) if 'config' in locals() else False),
        ("多尺度特定配置测试", lambda: test_multiscale_specific_config(config) if 'config' in locals() else False),
        ("配置兼容性测试", lambda: test_config_compatibility(config) if 'config' in locals() else False)
    ]
    
    passed = 0
    total = len(tests)
    config = None
    
    for i, (test_name, test_func) in enumerate(tests):
        logger.info(f"\n{'='*50}")
        logger.info(f"运行: {test_name}")
        logger.info(f"{'='*50}")
        
        if i == 0:  # 第一个测试返回配置
            result = test_func()
            if result:
                config = result
                passed += 1
                logger.info(f"✓ {test_name} 通过")
            else:
                logger.error(f"✗ {test_name} 失败")
        else:
            if config and test_func():
                passed += 1
                logger.info(f"✓ {test_name} 通过")
            else:
                logger.error(f"✗ {test_name} 失败")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"测试总结: {passed}/{total} 通过")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 所有配置测试通过！")
        return True
    else:
        logger.error(f"❌ {total - passed} 个测试失败")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)