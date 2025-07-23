#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCharm 快速启动脚本
PyCharm Quick Start Script for VIVTransformer Visualization Training

使用方法 / Usage:
1. 在PyCharm中直接运行此脚本
2. 根据提示选择运行模式
3. 自动配置环境并启动训练

作者: VIVTransformer Team
日期: 2025-07-23
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 设置环境变量
os.environ['MPLBACKEND'] = 'Agg'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'

def print_banner():
    """打印启动横幅"""
    print("="*60)
    print("🚀 VIVTransformer 可视化训练 - PyCharm 快速启动")
    print("🚀 VIVTransformer Visualization Training - PyCharm Quick Start")
    print("="*60)
    print()

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"   Python版本: {python_version}")
    
    if sys.version_info < (3, 7):
        print("❌ Python版本过低，需要3.7+")
        return False
    
    # 检查必要的包
    required_packages = [
        'torch', 'matplotlib', 'numpy', 'seaborn', 'pandas', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请在PyCharm Terminal中运行: pip install -r requirements.txt")
        return False
    
    print("✅ 环境检查通过")
    return True

def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.absolute()

def run_enhanced_training(epochs=20, output_dir="enhanced_training_output", batch_size=None, config_file=None):
    """运行增强版可视化训练"""
    if config_file:
        print(f"🎯 启动增强版可视化训练 (轮数: {epochs}, 配置文件: {config_file})")
    else:
        print(f"🎯 启动增强版可视化训练 (轮数: {epochs})")
    
    cmd = [sys.executable, "enhanced_visual_training.py", 
           "--epochs", str(epochs), 
           "--output-dir", output_dir]
    
    if config_file:
        cmd.extend(["--config", config_file])
    else:
        cmd.append("--demo")
    
    if batch_size:
        cmd.extend(["--batch-size", str(batch_size)])
    
    try:
        result = subprocess.run(cmd, cwd=get_project_root(), 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ 训练完成！结果保存在: {output_dir}/")
            return True
        else:
            print(f"❌ 训练失败，退出码: {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False

def run_prediction_demo():
    """运行预测可视化演示"""
    print("🎯 启动预测可视化演示")
    
    cmd = [sys.executable, "create_prediction_visualization.py"]
    
    try:
        result = subprocess.run(cmd, cwd=get_project_root(), 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print("✅ 预测可视化演示完成！结果保存在: demo_prediction_visualizations/")
            return True
        else:
            print(f"❌ 演示失败，退出码: {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False

def interactive_mode():
    """交互式模式"""
    print("📋 请选择运行模式:")
    print("1. 增强版可视化训练 (默认参数)")
    print("2. 增强版可视化训练 (自定义参数)")
    print("3. 预测可视化演示")
    print("4. 快速测试 (5轮训练)")
    print("5. 使用配置文件训练 (1000轮)")
    print("6. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-6): ").strip()
            
            if choice == '1':
                return run_enhanced_training()
            
            elif choice == '2':
                try:
                    epochs = int(input("输入训练轮数 (默认20): ") or "20")
                    output_dir = input("输入输出目录 (默认enhanced_training_output): ") or "enhanced_training_output"
                    batch_size_input = input("输入批次大小 (默认自动): ")
                    batch_size = int(batch_size_input) if batch_size_input else None
                    config_file_input = input("输入配置文件路径 (可选): ").strip()
                    config_file = config_file_input if config_file_input else None
                    return run_enhanced_training(epochs, output_dir, batch_size, config_file)
                except ValueError:
                    print("❌ 输入无效，请输入数字")
                    continue
            
            elif choice == '3':
                return run_prediction_demo()
            
            elif choice == '4':
                return run_enhanced_training(epochs=5, output_dir="quick_test_output")
            
            elif choice == '5':
                # 使用指定的配置文件运行1000轮训练
                config_file = "x:\\2025\\Graduation_project\\report\\VIVTransformer-4sh2r1-codex\\modify_multi_attention\\configs\\loss_configs\\loss_config_36.yaml"
                output_dir = "config_training_1000_epochs"
                print(f"🎯 使用配置文件运行1000轮训练")
                print(f"📁 配置文件: {config_file}")
                print(f"📁 输出目录: {output_dir}")
                return run_enhanced_training(epochs=1000, output_dir=output_dir, config_file=config_file)
            
            elif choice == '6':
                print("👋 退出程序")
                return True
            
            else:
                print("❌ 无效选择，请输入1-6")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序")
            return True
        except Exception as e:
            print(f"❌ 输入错误: {e}")
            continue

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyCharm快速启动脚本')
    parser.add_argument('--mode', choices=['enhanced', 'prediction', 'test', 'config'], 
                       help='运行模式')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--output-dir', default='enhanced_training_output', 
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--interactive', action='store_true', 
                       help='交互式模式')
    
    args = parser.parse_args()
    
    print_banner()
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请先安装依赖包")
        print("在PyCharm Terminal中运行: pip install -r requirements.txt")
        return 1
    
    print()
    
    # 根据参数运行
    if args.interactive or not args.mode:
        success = interactive_mode()
    elif args.mode == 'enhanced':
        success = run_enhanced_training(args.epochs, args.output_dir, args.batch_size, args.config)
    elif args.mode == 'prediction':
        success = run_prediction_demo()
    elif args.mode == 'test':
        success = run_enhanced_training(5, 'quick_test_output')
    elif args.mode == 'config':
        # 使用指定配置文件运行1000轮训练
        config_file = args.config or "x:\\2025\\Graduation_project\\report\\VIVTransformer-4sh2r1-codex\\modify_multi_attention\\configs\\loss_configs\\loss_config_36.yaml"
        success = run_enhanced_training(1000, 'config_training_1000_epochs', None, config_file)
    else:
        print("❌ 未知模式")
        return 1
    
    if success:
        print("\n🎉 任务完成！")
        print("\n📁 查看结果:")
        print("   - 在PyCharm项目文件树中刷新")
        print("   - 展开输出目录查看生成的文件")
        print("   - 双击图片文件可在PyCharm中预览")
        print("   - 双击HTML文件可在浏览器中打开")
        return 0
    else:
        print("\n❌ 任务失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)