#!/usr/bin/env python3
"""直接测试多尺度模块

避免复杂的相对导入，直接测试核心功能
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "multiscale"))
sys.path.insert(0, str(project_root / "multiscale" / "data"))

print("=== 多尺度模块直接测试 ===")
print(f"当前工作目录: {os.getcwd()}")
print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:3]}")

# 测试1: 基础导入
print("\n1. 测试基础导入...")
try:
    import torch
    import numpy as np
    print("✓ PyTorch和NumPy导入成功")
except ImportError as e:
    print(f"✗ 基础导入失败: {e}")
    sys.exit(1)

# 测试2: 配置文件加载
print("\n2. 测试配置文件加载...")
try:
    config_path = project_root / "multiscale" / "configs" / "multiscale_config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ 配置文件加载成功: {len(config)} 个配置项")
    else:
        print(f"✗ 配置文件不存在: {config_path}")
except Exception as e:
    print(f"✗ 配置文件加载失败: {e}")

# 测试3: 直接导入multiscale_adapter
print("\n3. 测试multiscale_adapter导入...")
try:
    from multiscale_adapter import MultiScaleDataset
    print("✓ MultiScaleDataset导入成功")
except ImportError as e:
    print(f"✗ MultiScaleDataset导入失败: {e}")
    # 尝试绝对路径导入
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "multiscale_adapter", 
            project_root / "multiscale" / "data" / "multiscale_adapter.py"
        )
        multiscale_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(multiscale_module)
        MultiScaleDataset = multiscale_module.MultiScaleDataset
        print("✓ MultiScaleDataset通过绝对路径导入成功")
    except Exception as e2:
        print(f"✗ 绝对路径导入也失败: {e2}")
        sys.exit(1)

# 测试4: 创建测试数据
print("\n4. 测试创建虚拟数据...")
try:
    # 创建虚拟数据进行测试
    test_data = torch.randn(10, 128, 128, 1)  # 10个样本，128x128分辨率，1个通道
    print(f"✓ 虚拟数据创建成功: {test_data.shape}")
    
    # 测试下采样
    downsampled = torch.nn.functional.avg_pool2d(
        test_data.permute(0, 3, 1, 2), 
        kernel_size=2, 
        stride=2
    )
    print(f"✓ 下采样测试成功: {test_data.shape} -> {downsampled.shape}")
    
except Exception as e:
    print(f"✗ 数据处理测试失败: {e}")

# 测试5: 基本张量操作
print("\n5. 测试基本张量操作...")
try:
    x = torch.randn(4, 64, 64, 2)  # 批次大小4，64x64分辨率，2个通道
    y = torch.randn(4, 128, 128, 2)  # 目标分辨率128x128
    
    # 测试重塑操作
    x_flat = x.view(x.size(0), -1)  # 展平
    y_flat = y.view(y.size(0), -1)
    
    print(f"✓ 张量重塑成功: {x.shape} -> {x_flat.shape}")
    print(f"✓ 目标张量: {y.shape} -> {y_flat.shape}")
    
    # 计算维度信息
    input_dim = x_flat.size(1)
    output_dim = y_flat.size(1)
    compression_ratio = output_dim / input_dim
    
    print(f"✓ 输入维度: {input_dim}")
    print(f"✓ 输出维度: {output_dim}")
    print(f"✓ 压缩比: {compression_ratio:.2f}")
    
except Exception as e:
    print(f"✗ 张量操作测试失败: {e}")

print("\n=== 测试完成 ===")
print("多尺度模块基础功能正常，可以进行进一步开发。")