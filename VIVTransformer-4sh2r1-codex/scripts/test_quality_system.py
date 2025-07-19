#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量监控系统测试脚本

功能:
- 测试所有质量监控组件
- 验证脚本可以正常运行
- 检查依赖和配置

作者: VIVTransformer Team
日期: 2024
"""

import os
import sys
import subprocess
from pathlib import Path


def test_script_syntax(script_path: str) -> bool:
    """测试脚本语法"""
    try:
        result = subprocess.run([
            sys.executable, '-m', 'py_compile', script_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {Path(script_path).name} - 语法正确")
            return True
        else:
            print(f"❌ {Path(script_path).name} - 语法错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {Path(script_path).name} - 测试失败: {e}")
        return False


def test_script_help(script_path: str) -> bool:
    """测试脚本帮助信息"""
    try:
        result = subprocess.run([
            sys.executable, script_path, '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ {Path(script_path).name} - 帮助信息正常")
            return True
        else:
            print(f"❌ {Path(script_path).name} - 帮助信息错误: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⚠️ {Path(script_path).name} - 帮助信息超时")
        return False
    except Exception as e:
        print(f"❌ {Path(script_path).name} - 测试失败: {e}")
        return False


def test_import_dependencies() -> bool:
    """测试依赖导入"""
    print("\n🔍 测试依赖导入...")
    
    required_modules = [
        'yaml', 'pathlib', 'argparse', 'logging', 'json',
        'datetime', 'subprocess', 'threading'
    ]
    
    optional_modules = [
        'matplotlib', 'seaborn', 'pandas', 'schedule', 
        'requests', 'smtplib', 'sqlite3'
    ]
    
    success = True
    
    # 测试必需模块
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - 必需模块缺失")
            success = False
    
    # 测试可选模块
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module} (可选)")
        except ImportError:
            print(f"⚠️ {module} (可选) - 未安装")
    
    return success


def main():
    """主函数"""
    print("🚀 VIVTransformer 质量监控系统测试")
    print("=" * 50)
    
    scripts_dir = Path(__file__).parent
    
    # 要测试的脚本列表
    scripts_to_test = [
        'start_quality_monitor.py',
        'quality_monitor_system.py',
        'quality_dashboard.py',
        'quality_trend_analyzer.py',
        'integrated_quality_system.py',
        'example_usage.py'
    ]
    
    # 测试依赖导入
    deps_ok = test_import_dependencies()
    
    print("\n🔍 测试脚本语法...")
    syntax_results = []
    for script in scripts_to_test:
        script_path = scripts_dir / script
        if script_path.exists():
            result = test_script_syntax(str(script_path))
            syntax_results.append(result)
        else:
            print(f"⚠️ {script} - 文件不存在")
            syntax_results.append(False)
    
    print("\n🔍 测试脚本帮助信息...")
    help_results = []
    for script in scripts_to_test:
        script_path = scripts_dir / script
        if script_path.exists():
            result = test_script_help(str(script_path))
            help_results.append(result)
        else:
            help_results.append(False)
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    
    print(f"依赖导入: {'✅ 通过' if deps_ok else '❌ 失败'}")
    print(f"语法检查: {sum(syntax_results)}/{len(syntax_results)} 通过")
    print(f"帮助信息: {sum(help_results)}/{len(help_results)} 通过")
    
    total_tests = len(syntax_results) + len(help_results) + 1
    passed_tests = sum(syntax_results) + sum(help_results) + (1 if deps_ok else 0)
    
    print(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！质量监控系统可以正常使用。")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查上述错误信息。")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)