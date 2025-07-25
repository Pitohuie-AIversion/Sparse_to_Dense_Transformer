#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer 服务器部署工具包
Server Deployment Toolkit for VIVTransformer

这个脚本提供了在Linux服务器上部署和运行VIVTransformer的完整工具集。
This script provides a complete toolkit for deploying and running VIVTransformer on Linux servers.

作者: VIVTransformer Team
版本: 1.0
日期: 2025-01-26
"""

import os
import sys
import subprocess
import platform
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional

class ServerDeploymentManager:
    """服务器部署管理器"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / "viv_env"
        self.python_executable = "python3"
        
    def check_system_requirements(self) -> bool:
        """检查系统要求"""
        print("🔍 检查系统要求...")
        
        # 检查操作系统
        if platform.system() != "Linux":
            print("❌ 错误: 此工具仅支持Linux系统")
            return False
            
        # 检查Python3
        if not shutil.which("python3"):
            print("❌ 错误: 未找到Python3")
            print("请安装Python3:")
            print("  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv")
            print("  CentOS/RHEL: sudo yum install python3 python3-pip")
            return False
            
        # 检查Python版本
        try:
            result = subprocess.run(["python3", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], 
                                  capture_output=True, text=True)
            version = float(result.stdout.strip())
            if version < 3.7:
                print(f"❌ 错误: Python版本过低 ({version})，需要3.7+")
                return False
            print(f"✅ Python版本检查通过: {version}")
        except Exception as e:
            print(f"❌ 错误: 无法检查Python版本: {e}")
            return False
            
        return True
    
    def setup_virtual_environment(self) -> bool:
        """设置虚拟环境"""
        print("🐍 设置Python虚拟环境...")
        
        try:
            if self.venv_path.exists():
                print("⚠️  虚拟环境已存在，跳过创建")
            else:
                subprocess.run(["python3", "-m", "venv", str(self.venv_path)], check=True)
                print("✅ 虚拟环境创建完成")
                
            # 升级pip
            pip_path = self.venv_path / "bin" / "pip"
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            print("✅ pip已升级")
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 错误: 虚拟环境设置失败: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """安装依赖包"""
        print("📦 安装Python依赖包...")
        
        dependencies = [
            "torch",
            "torchvision", 
            "matplotlib",
            "numpy",
            "seaborn",
            "pandas",
            "psutil",
            "tqdm",
            "tensorboard"
        ]
        
        pip_path = self.venv_path / "bin" / "pip"
        
        for dep in dependencies:
            try:
                print(f"  安装 {dep}...")
                subprocess.run([str(pip_path), "install", dep], 
                             check=True, capture_output=True)
                print(f"  ✅ {dep} 安装完成")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ {dep} 安装失败: {e}")
                return False
                
        print("✅ 所有依赖包安装完成")
        return True
    
    def configure_environment(self) -> bool:
        """配置环境变量"""
        print("⚙️  配置环境变量...")
        
        try:
            activate_script = self.venv_path / "bin" / "activate"
            
            env_vars = """
# VIVTransformer 环境变量
export MPLBACKEND=Agg
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1
"""
            
            with open(activate_script, "a") as f:
                f.write(env_vars)
                
            print("✅ 环境变量配置完成")
            return True
        except Exception as e:
            print(f"❌ 错误: 环境变量配置失败: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """验证安装"""
        print("🔬 验证安装...")
        
        python_path = self.venv_path / "bin" / "python"
        
        test_script = """
import torch
import matplotlib
import numpy
import seaborn
import pandas
import psutil
print('所有依赖包导入成功')
"""
        
        try:
            result = subprocess.run([str(python_path), "-c", test_script], 
                                  check=True, capture_output=True, text=True)
            print("✅ 依赖包验证通过")
            
            # 检查CUDA
            cuda_script = "import torch; print(torch.cuda.is_available())"
            cuda_result = subprocess.run([str(python_path), "-c", cuda_script], 
                                       capture_output=True, text=True)
            if cuda_result.stdout.strip() == "True":
                print("✅ CUDA可用，支持GPU加速")
            else:
                print("⚠️  CUDA不可用，将使用CPU训练")
                
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 错误: 验证失败: {e}")
            return False
    
    def create_launcher_scripts(self) -> bool:
        """创建启动脚本"""
        print("🚀 创建启动脚本...")
        
        try:
            # 创建主启动脚本
            launcher_content = f"""#!/bin/bash

# VIVTransformer 快速启动脚本
echo "激活虚拟环境..."
source {self.venv_path}/bin/activate

echo "选择运行模式:"
echo "1) 多尺度训练 (9x9 -> 128x128)"
echo "2) 增强版可视化训练"
echo "3) 独立预测可视化演示"
echo "4) 自定义参数训练"
read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "启动多尺度训练..."
        python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 4
        ;;
    2)
        echo "启动增强版可视化训练..."
        python enhanced_visual_training.py --epochs 20 --output-dir enhanced_training_output
        ;;
    3)
        echo "启动预测可视化演示..."
        python create_prediction_visualization.py
        ;;
    4)
        read -p "输入训练轮数 (默认20): " epochs
        epochs=${{epochs:-20}}
        read -p "输入批次大小 (默认4): " batch_size
        batch_size=${{batch_size:-4}}
        read -p "输入缩放因子 (默认4): " scale_factor
        scale_factor=${{scale_factor:-4}}
        echo "启动自定义训练..."
        python train_configurable_multiscale.py --scale_factor $scale_factor --num_epochs $epochs --batch_size $batch_size
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
"""
            
            launcher_path = self.project_root / "run_training.sh"
            with open(launcher_path, "w") as f:
                f.write(launcher_content)
            os.chmod(launcher_path, 0o755)
            
            # 创建TensorBoard启动脚本
            tensorboard_content = f"""#!/bin/bash

# TensorBoard 启动脚本
source {self.venv_path}/bin/activate

echo "选择TensorBoard日志目录:"
echo "1) 最新的多尺度训练结果"
echo "2) 指定目录"
read -p "请选择 (1-2): " choice

case $choice in
    1)
        latest_dir=$(ls -td configurable_multiscale_results_* 2>/dev/null | head -1)
        if [ -z "$latest_dir" ]; then
            echo "未找到训练结果目录"
            exit 1
        fi
        tensorboard --logdir "$latest_dir/tensorboard" --port 6006 --host 0.0.0.0
        ;;
    2)
        read -p "输入日志目录路径: " log_dir
        tensorboard --logdir "$log_dir" --port 6006 --host 0.0.0.0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
"""
            
            tensorboard_path = self.project_root / "start_tensorboard.sh"
            with open(tensorboard_path, "w") as f:
                f.write(tensorboard_content)
            os.chmod(tensorboard_path, 0o755)
            
            print("✅ 启动脚本创建完成")
            return True
        except Exception as e:
            print(f"❌ 错误: 启动脚本创建失败: {e}")
            return False
    
    def deploy(self) -> bool:
        """执行完整部署"""
        print("🚀 开始VIVTransformer服务器部署...")
        print("=" * 50)
        
        steps = [
            ("检查系统要求", self.check_system_requirements),
            ("设置虚拟环境", self.setup_virtual_environment),
            ("安装依赖包", self.install_dependencies),
            ("配置环境变量", self.configure_environment),
            ("验证安装", self.verify_installation),
            ("创建启动脚本", self.create_launcher_scripts)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ 部署失败: {step_name}")
                return False
                
        print("\n" + "=" * 50)
        print("🎉 部署完成！")
        self.show_usage_instructions()
        return True
    
    def show_usage_instructions(self):
        """显示使用说明"""
        print("\n📖 使用说明:")
        print("=" * 30)
        print("\n🚀 快速启动:")
        print("   ./run_training.sh")
        print("\n📊 启动TensorBoard:")
        print("   ./start_tensorboard.sh")
        print("\n⚙️  手动激活环境:")
        print(f"   source {self.venv_path}/bin/activate")
        print("\n📁 输出文件位置:")
        print("   - 多尺度训练: configurable_multiscale_results_*/")
        print("   - 增强版训练: enhanced_training_output/")
        print("   - 预测可视化: demo_prediction_visualizations/")
        print("\n📖 详细文档: LINUX_SERVER_GUIDE.md")
        print("\n🌐 远程访问TensorBoard: http://your_server_ip:6006")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VIVTransformer服务器部署工具")
    parser.add_argument("--deploy", action="store_true", help="执行完整部署")
    parser.add_argument("--check", action="store_true", help="仅检查系统要求")
    parser.add_argument("--verify", action="store_true", help="验证现有安装")
    
    args = parser.parse_args()
    
    manager = ServerDeploymentManager()
    
    if args.check:
        manager.check_system_requirements()
    elif args.verify:
        manager.verify_installation()
    elif args.deploy:
        manager.deploy()
    else:
        print("VIVTransformer 服务器部署工具")
        print("使用 --deploy 执行完整部署")
        print("使用 --check 检查系统要求")
        print("使用 --verify 验证现有安装")

if __name__ == "__main__":
    main()