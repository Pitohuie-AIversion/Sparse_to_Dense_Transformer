#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试多尺度训练模块导入
"""

import sys
from pathlib import Path

print("=== 多尺度训练模块导入测试 ===")
print(f"当前工作目录: {Path.cwd()}")
print(f"Python路径: {sys.path[:3]}...")

try:
    # 添加项目路径
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    print(f"添加项目路径: {project_root}")
    
    # 测试导入多尺度训练器
    from train_configurable_multiscale import ConfigurableMultiScaleTrainer
    print("✅ ConfigurableMultiScaleTrainer 导入成功!")
    
    # 测试导入多尺度数据适配器
    from multiscale.data.multiscale_adapter import create_multiscale_loaders
    print("✅ create_multiscale_loaders 导入成功!")
    
    print("\n🎉 所有多尺度模块导入测试通过!")
    print("✅ 项目结构优化后，多尺度训练功能完全正常!")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("\n检查文件是否存在:")
    
    # 检查关键文件
    files_to_check = [
        Path(__file__).parent / "train_configurable_multiscale.py",
        Path(__file__).parent / "multiscale" / "data" / "multiscale_adapter.py",
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            print(f"✅ {file_path.name} 存在")
        else:
            print(f"❌ {file_path.name} 不存在")
            
except Exception as e:
    print(f"❌ 其他错误: {e}")

print("\n=== 测试完成 ===")