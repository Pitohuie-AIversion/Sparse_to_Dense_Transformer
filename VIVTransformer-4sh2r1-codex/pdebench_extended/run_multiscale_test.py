#!/usr/bin/env python3
"""
简化的多尺度测试脚本
直接从项目根目录运行，避免复杂的导入问题
"""

import sys
import os
from pathlib import Path
import logging
import yaml

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

def test_config_loading():
    """测试配置文件加载"""
    logger.info("测试配置文件加载...")
    
    config_path = 'multiscale/configs/multiscale_config.yaml'
    
    try:
        config = load_config(config_path)
        logger.info(f"✓ 配置文件加载成功: {config_path}")
        
        # 检查关键配置
        assert 'data' in config, "缺少data配置"
        assert 'model' in config, "缺少model配置"
        assert 'training' in config, "缺少training配置"
        
        data_config = config['data']
        model_config = config['model']
        
        logger.info(f"缩放因子: {data_config.get('scale_factor')}")
        logger.info(f"原始分辨率: {data_config.get('original_resolution')}")
        logger.info(f"模型维度: d_model={model_config.get('d_model')}, num_heads={model_config.get('num_heads')}")
        
        return True
        
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        return False

def test_basic_imports():
    """测试基本导入"""
    logger.info("测试基本导入...")
    
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        
        logger.info(f"✓ PyTorch版本: {torch.__version__}")
        logger.info(f"✓ NumPy版本: {np.__version__}")
        logger.info(f"✓ CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA设备数量: {torch.cuda.device_count()}")
            logger.info(f"✓ 当前CUDA设备: {torch.cuda.current_device()}")
        
        return True
        
    except Exception as e:
        logger.error(f"基本导入失败: {e}")
        return False

def test_multiscale_imports():
    """测试多尺度模块导入"""
    logger.info("测试多尺度模块导入...")
    
    try:
        # 添加multiscale目录到路径
        multiscale_path = Path('multiscale')
        if multiscale_path.exists():
            sys.path.insert(0, str(multiscale_path))
        
        # 尝试导入多尺度模块
        from data.multiscale_adapter import MultiScaleDataset
        logger.info("✓ MultiScaleDataset导入成功")
        
        return True
        
    except Exception as e:
        logger.error(f"多尺度模块导入失败: {e}")
        return False

def test_data_path():
    """测试数据路径"""
    logger.info("测试数据路径...")
    
    try:
        config = load_config('multiscale/configs/multiscale_config.yaml')
        data_path = config['data']['data_path']
        
        if os.path.exists(data_path):
            logger.info(f"✓ 数据文件存在: {data_path}")
            file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
            logger.info(f"✓ 文件大小: {file_size:.2f} MB")
            return True
        else:
            logger.warning(f"⚠ 数据文件不存在: {data_path}")
            logger.info("这是正常的，如果您还没有下载数据文件")
            return True
            
    except Exception as e:
        logger.error(f"数据路径测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始运行多尺度测试...")
    logger.info("=" * 50)
    
    tests = [
        ("配置文件加载", test_config_loading),
        ("基本导入", test_basic_imports),
        ("多尺度模块导入", test_multiscale_imports),
        ("数据路径", test_data_path),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n运行测试: {test_name}")
        logger.info("-" * 30)
        
        try:
            if test_func():
                logger.info(f"✓ {test_name} 通过")
                passed += 1
            else:
                logger.error(f"✗ {test_name} 失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 异常: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"测试总结: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！")
        return True
    else:
        logger.warning(f"⚠ {total - passed} 个测试失败")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)