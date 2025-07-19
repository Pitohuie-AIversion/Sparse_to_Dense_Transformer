#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量监控系统演示脚本

这个脚本演示了VIVTransformer质量监控系统的主要功能：
1. 质量检查和分析
2. 质量趋势分析
3. 质量报告生成
4. 质量监控配置
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        if result.stdout:
            print("输出:")
            print(result.stdout)
        
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 命令执行成功")
        else:
            print(f"❌ 命令执行失败 (退出码: {result.returncode})")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ 命令执行超时")
        return False
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False

def main():
    """主演示函数"""
    print("🚀 VIVTransformer 质量监控系统演示")
    print("=" * 60)
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 检查脚本是否存在
    scripts_dir = Path("scripts")
    if not scripts_dir.exists():
        print("❌ scripts目录不存在，请在项目根目录运行此脚本")
        return
    
    # 1. 显示帮助信息
    commands = [
        ("python scripts\\start_quality_monitor.py --help", "质量监控系统帮助"),
        ("python scripts\\integrated_quality_system.py --help", "集成质量分析帮助"),
        ("python scripts\\quality_trend_analyzer.py --help", "质量趋势分析帮助"),
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)
        time.sleep(1)
    
    # 2. 生成配置文件
    run_command("python scripts\\start_quality_monitor.py --generate-config", "生成质量监控配置")
    time.sleep(1)
    
    # 3. 执行质量检查
    run_command("python scripts\\start_quality_monitor.py --check", "执行质量检查")
    time.sleep(2)
    
    # 4. 质量趋势分析
    run_command("python scripts\\quality_trend_analyzer.py --collect . --analyze --report --days 7", "质量趋势分析")
    time.sleep(2)
    
    # 5. 显示生成的文件
    print("\n" + "="*60)
    print("📁 生成的文件和报告")
    print("="*60)
    
    files_to_check = [
        "quality_monitoring_config.yaml",
        "quality_monitoring.db",
        "reports/quality_trend_report.md",
        "temp_metrics.json"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} (不存在)")
    
    # 6. 显示配置文件内容
    config_file = "quality_monitoring_config.yaml"
    if os.path.exists(config_file):
        print(f"\n📋 配置文件内容 ({config_file}):")
        print("-" * 40)
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content[:500] + "..." if len(content) > 500 else content)
        except Exception as e:
            print(f"读取配置文件失败: {e}")
    
    # 7. 显示趋势报告内容
    trend_report = "reports/quality_trend_report.md"
    if os.path.exists(trend_report):
        print(f"\n📈 趋势报告内容 ({trend_report}):")
        print("-" * 40)
        try:
            with open(trend_report, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content[:800] + "..." if len(content) > 800 else content)
        except Exception as e:
            print(f"读取趋势报告失败: {e}")
    
    print("\n" + "="*60)
    print("🎉 演示完成！")
    print("="*60)
    print("\n💡 使用建议:")
    print("1. 定期运行质量检查: python scripts\\start_quality_monitor.py --check")
    print("2. 查看质量趋势: python scripts\\quality_trend_analyzer.py --analyze --report")
    print("3. 配置监控规则: 编辑 quality_monitoring_config.yaml")
    print("4. 设置持续监控: python scripts\\start_quality_monitor.py --monitor")

if __name__ == "__main__":
    main()