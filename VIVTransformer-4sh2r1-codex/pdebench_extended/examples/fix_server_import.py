#!/usr/bin/env python3
"""
服务器端导入问题修复脚本

这个脚本用于修复在Linux服务器上运行时可能遇到的模块导入问题。
主要解决multiscale.data.multiscale_adapter模块导入失败的问题。

使用方法:
1. 在服务器上运行此脚本进行环境检查和修复
2. 然后再运行训练脚本

作者: VIVTransformer Team
"""

import os
import sys
from pathlib import Path
import importlib.util

def check_python_path():
    """检查Python路径设置"""
    print("=== Python路径检查 ===")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    print()

def check_file_structure():
    """检查文件结构"""
    print("=== 文件结构检查 ===")
    
    current_dir = Path.cwd()
    print(f"当前目录: {current_dir}")
    
    # 检查关键文件和目录
    key_paths = [
        "multiscale",
        "multiscale/data", 
        "multiscale/data/multiscale_adapter.py",
        "multiscale/data/__init__.py",
        "multiscale/__init__.py",
        "mymodels",
        "mymodels/transformer.py"
    ]
    
    for path_str in key_paths:
        path = current_dir / path_str
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {path_str}")
    
    print()

def fix_import_paths():
    """修复导入路径"""
    print("=== 修复导入路径 ===")
    
    current_dir = Path.cwd()
    
    # 添加必要的路径到sys.path
    paths_to_add = [
        current_dir,
        current_dir / "multiscale",
        current_dir / "multiscale" / "data"
    ]
    
    for path in paths_to_add:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            print(f"  添加路径: {path}")
    
    print()

def test_imports():
    """测试关键模块导入"""
    print("=== 测试模块导入 ===")
    
    # 测试multiscale_adapter导入
    try:
        from multiscale.data.multiscale_adapter import create_multiscale_loaders
        print("  ✓ multiscale.data.multiscale_adapter 导入成功")
    except ImportError as e:
        print(f"  ✗ multiscale.data.multiscale_adapter 导入失败: {e}")
        
        # 尝试备用导入方式
        try:
            import multiscale_adapter
            print("  ✓ 直接导入 multiscale_adapter 成功")
        except ImportError as e2:
            print(f"  ✗ 直接导入 multiscale_adapter 失败: {e2}")
    
    # 测试transformer模型导入
    try:
        from mymodels.transformer import TransformerFlowReconstructionModel
        print("  ✓ mymodels.transformer 导入成功")
    except ImportError as e:
        print(f"  ✗ mymodels.transformer 导入失败: {e}")
    
    print()

def create_init_files():
    """创建缺失的__init__.py文件"""
    print("=== 创建缺失的__init__.py文件 ===")
    
    current_dir = Path.cwd()
    
    # 需要__init__.py的目录
    init_dirs = [
        current_dir / "multiscale",
        current_dir / "multiscale" / "data",
        current_dir / "mymodels"
    ]
    
    for dir_path in init_dirs:
        if dir_path.exists():
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"  创建: {init_file}")
            else:
                print(f"  存在: {init_file}")
    
    print()

def main():
    """主函数"""
    print("服务器端导入问题修复脚本")
    print("=" * 50)
    
    check_python_path()
    check_file_structure()
    create_init_files()
    fix_import_paths()
    test_imports()
    
    print("=== 修复完成 ===")
    print("如果所有测试都通过，现在可以运行训练脚本了。")
    print("")
    print("推荐的运行命令:")
    print("python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 4")
    print("")
    print("如果仍有问题，请检查:")
    print("1. 确保在正确的目录下运行脚本")
    print("2. 确保所有必要的文件都存在")
    print("3. 检查Python环境和依赖包")

if __name__ == "__main__":
    main()