#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer 服务器快速命令工具
Quick Commands Tool for VIVTransformer Server Operations

提供常用的服务器操作命令和快捷功能。
Provides common server operation commands and shortcuts.

作者: VIVTransformer Team
版本: 1.0
日期: 2025-01-26
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import glob

class ServerQuickCommands:
    """服务器快速命令工具类"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / "viv_env"
        
    def activate_env_command(self):
        """返回激活虚拟环境的命令"""
        return f"source {self.venv_path}/bin/activate"
    
    def list_training_results(self):
        """列出所有训练结果目录"""
        print("📁 训练结果目录:")
        print("=" * 40)
        
        # 多尺度训练结果
        multiscale_dirs = glob.glob("configurable_multiscale_results_*")
        if multiscale_dirs:
            print("\n🔬 多尺度训练结果:")
            for i, dir_name in enumerate(sorted(multiscale_dirs, reverse=True), 1):
                size = self._get_dir_size(dir_name)
                print(f"  {i}. {dir_name} ({size})")
        
        # 增强版训练结果
        enhanced_dirs = glob.glob("enhanced_training_output*")
        if enhanced_dirs:
            print("\n🎨 增强版训练结果:")
            for i, dir_name in enumerate(sorted(enhanced_dirs, reverse=True), 1):
                size = self._get_dir_size(dir_name)
                print(f"  {i}. {dir_name} ({size})")
        
        # 预测可视化结果
        viz_dirs = glob.glob("demo_prediction_visualizations*")
        if viz_dirs:
            print("\n📊 预测可视化结果:")
            for i, dir_name in enumerate(sorted(viz_dirs, reverse=True), 1):
                size = self._get_dir_size(dir_name)
                print(f"  {i}. {dir_name} ({size})")
        
        if not (multiscale_dirs or enhanced_dirs or viz_dirs):
            print("❌ 未找到训练结果目录")
    
    def _get_dir_size(self, dir_path):
        """获取目录大小"""
        try:
            result = subprocess.run(["du", "-sh", dir_path], 
                                  capture_output=True, text=True)
            return result.stdout.split()[0]
        except:
            return "未知"
    
    def show_latest_config(self):
        """显示最新训练的配置"""
        print("⚙️  最新训练配置:")
        print("=" * 30)
        
        # 查找最新的配置文件
        config_files = glob.glob("configurable_multiscale_results_*/config.json")
        if not config_files:
            print("❌ 未找到配置文件")
            return
        
        latest_config = max(config_files, key=os.path.getctime)
        print(f"📄 配置文件: {latest_config}")
        
        try:
            with open(latest_config, 'r') as f:
                config = json.load(f)
            
            print("\n📋 关键配置:")
            print(f"  缩放因子: {config.get('scale_factor', 'N/A')}")
            print(f"  训练轮数: {config.get('num_epochs', 'N/A')}")
            print(f"  批次大小: {config.get('batch_size', 'N/A')}")
            print(f"  输入分辨率: {config.get('input_resolution', 'N/A')}")
            print(f"  输出分辨率: {config.get('output_resolution', 'N/A')}")
            print(f"  中心裁剪: {config.get('enable_center_crop', 'N/A')}")
            print(f"  学习率: {config.get('learning_rate', 'N/A')}")
            
        except Exception as e:
            print(f"❌ 读取配置文件失败: {e}")
    
    def generate_training_commands(self):
        """生成常用训练命令"""
        print("🚀 常用训练命令:")
        print("=" * 40)
        
        commands = [
            {
                "name": "标准多尺度训练",
                "cmd": "python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 4"
            },
            {
                "name": "快速测试训练 (3轮)",
                "cmd": "python train_configurable_multiscale.py --scale_factor 4 --num_epochs 3 --batch_size 4"
            },
            {
                "name": "长时间训练 (100轮)",
                "cmd": "python train_configurable_multiscale.py --scale_factor 4 --num_epochs 100 --batch_size 4"
            },
            {
                "name": "大批次训练",
                "cmd": "python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 8"
            },
            {
                "name": "小批次训练 (内存受限)",
                "cmd": "python train_configurable_multiscale.py --scale_factor 4 --num_epochs 20 --batch_size 2"
            },
            {
                "name": "增强版可视化训练",
                "cmd": "python enhanced_visual_training.py --epochs 20 --output-dir enhanced_training_output"
            },
            {
                "name": "预测可视化演示",
                "cmd": "python create_prediction_visualization.py"
            }
        ]
        
        for i, cmd_info in enumerate(commands, 1):
            print(f"\n{i}. {cmd_info['name']}:")
            print(f"   {self.activate_env_command()}")
            print(f"   {cmd_info['cmd']}")
    
    def generate_monitoring_commands(self):
        """生成监控命令"""
        print("📊 监控命令:")
        print("=" * 30)
        
        commands = [
            {
                "name": "启动TensorBoard (最新结果)",
                "cmd": "./start_tensorboard.sh"
            },
            {
                "name": "手动启动TensorBoard",
                "cmd": "tensorboard --logdir configurable_multiscale_results_*/tensorboard --port 6006 --host 0.0.0.0"
            },
            {
                "name": "查看训练日志",
                "cmd": "tail -f configurable_multiscale_results_*/logs/training.log"
            },
            {
                "name": "查看系统资源",
                "cmd": "htop"
            },
            {
                "name": "查看GPU状态",
                "cmd": "nvidia-smi"
            },
            {
                "name": "查看磁盘使用",
                "cmd": "df -h"
            },
            {
                "name": "查看内存使用",
                "cmd": "free -h"
            }
        ]
        
        for i, cmd_info in enumerate(commands, 1):
            print(f"\n{i}. {cmd_info['name']}:")
            print(f"   {cmd_info['cmd']}")
    
    def generate_file_operations(self):
        """生成文件操作命令"""
        print("📁 文件操作命令:")
        print("=" * 35)
        
        commands = [
            {
                "name": "打包最新训练结果",
                "cmd": "tar -czf latest_results.tar.gz $(ls -td configurable_multiscale_results_* | head -1)"
            },
            {
                "name": "下载结果到本地 (SCP)",
                "cmd": "scp -r username@server_ip:/path/to/results ./local_results/"
            },
            {
                "name": "启动HTTP文件服务器",
                "cmd": "python3 -m http.server 8000"
            },
            {
                "name": "清理旧的训练结果 (保留最新5个)",
                "cmd": "ls -td configurable_multiscale_results_* | tail -n +6 | xargs rm -rf"
            },
            {
                "name": "查看结果目录大小",
                "cmd": "du -sh configurable_multiscale_results_*"
            },
            {
                "name": "备份重要结果",
                "cmd": "cp -r configurable_multiscale_results_* /backup/location/"
            }
        ]
        
        for i, cmd_info in enumerate(commands, 1):
            print(f"\n{i}. {cmd_info['name']}:")
            print(f"   {cmd_info['cmd']}")
    
    def generate_troubleshooting_commands(self):
        """生成故障排除命令"""
        print("🔧 故障排除命令:")
        print("=" * 35)
        
        commands = [
            {
                "name": "检查Python环境",
                "cmd": "python3 -c \"import torch, matplotlib, numpy; print('Environment OK')\""
            },
            {
                "name": "检查CUDA可用性",
                "cmd": "python3 -c \"import torch; print(f'CUDA: {torch.cuda.is_available()}')\""
            },
            {
                "name": "重新安装依赖",
                "cmd": "pip install --force-reinstall torch torchvision matplotlib numpy seaborn pandas"
            },
            {
                "name": "清理pip缓存",
                "cmd": "pip cache purge"
            },
            {
                "name": "检查端口占用",
                "cmd": "netstat -tulpn | grep :6006"
            },
            {
                "name": "杀死TensorBoard进程",
                "cmd": "pkill -f tensorboard"
            },
            {
                "name": "设置权限",
                "cmd": "chmod +x *.sh && chmod +x *.py"
            },
            {
                "name": "查看错误日志",
                "cmd": "journalctl -f | grep python"
            }
        ]
        
        for i, cmd_info in enumerate(commands, 1):
            print(f"\n{i}. {cmd_info['name']}:")
            print(f"   {cmd_info['cmd']}")
    
    def create_quick_scripts(self):
        """创建快速操作脚本"""
        print("📝 创建快速操作脚本...")
        
        # 创建状态检查脚本
        status_script = f"""#!/bin/bash
# 系统状态检查脚本

echo "=== VIVTransformer 系统状态 ==="
echo "时间: $(date)"
echo ""

echo "📊 系统资源:"
echo "内存使用: $(free -h | grep Mem | awk '{{print $3"/"$2}}')"
echo "磁盘使用: $(df -h . | tail -1 | awk '{{print $3"/"$2" ("$5")}}')"
echo ""

echo "🐍 Python环境:"
{self.activate_env_command()}
python3 -c "import torch; print(f'PyTorch: {{torch.__version__}}')"
python3 -c "import torch; print(f'CUDA可用: {{torch.cuda.is_available()}}')"
echo ""

echo "📁 训练结果:"
ls -la configurable_multiscale_results_* 2>/dev/null | wc -l | xargs echo "多尺度结果数量:"
ls -la enhanced_training_output* 2>/dev/null | wc -l | xargs echo "增强版结果数量:"
echo ""

echo "🔄 运行中的进程:"
ps aux | grep -E "(python.*train|tensorboard)" | grep -v grep
"""
        
        with open("check_status.sh", "w") as f:
            f.write(status_script)
        os.chmod("check_status.sh", 0o755)
        
        # 创建清理脚本
        cleanup_script = """#!/bin/bash
# 清理脚本

echo "🧹 开始清理..."

# 清理旧的训练结果 (保留最新3个)
echo "清理旧的训练结果..."
ls -td configurable_multiscale_results_* 2>/dev/null | tail -n +4 | xargs rm -rf

# 清理临时文件
echo "清理临时文件..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 清理日志文件 (保留最新的)
echo "清理旧日志文件..."
find . -name "*.log" -mtime +7 -delete 2>/dev/null

echo "✅ 清理完成"
"""
        
        with open("cleanup.sh", "w") as f:
            f.write(cleanup_script)
        os.chmod("cleanup.sh", 0o755)
        
        # 创建备份脚本
        backup_script = """#!/bin/bash
# 备份脚本

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo "📦 创建备份: $BACKUP_DIR"

mkdir -p "$BACKUP_DIR"

# 备份最新的训练结果
latest_result=$(ls -td configurable_multiscale_results_* 2>/dev/null | head -1)
if [ -n "$latest_result" ]; then
    echo "备份训练结果: $latest_result"
    cp -r "$latest_result" "$BACKUP_DIR/"
fi

# 备份配置文件
echo "备份配置文件..."
cp *.py "$BACKUP_DIR/" 2>/dev/null
cp *.sh "$BACKUP_DIR/" 2>/dev/null
cp *.md "$BACKUP_DIR/" 2>/dev/null

# 创建压缩包
tar -czf "${BACKUP_DIR}.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "✅ 备份完成: ${BACKUP_DIR}.tar.gz"
"""
        
        with open("backup.sh", "w") as f:
            f.write(backup_script)
        os.chmod("backup.sh", 0o755)
        
        print("✅ 快速操作脚本创建完成:")
        print("   - check_status.sh: 系统状态检查")
        print("   - cleanup.sh: 清理旧文件")
        print("   - backup.sh: 备份重要文件")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VIVTransformer服务器快速命令工具")
    parser.add_argument("--list-results", action="store_true", help="列出训练结果")
    parser.add_argument("--show-config", action="store_true", help="显示最新配置")
    parser.add_argument("--training-commands", action="store_true", help="显示训练命令")
    parser.add_argument("--monitoring-commands", action="store_true", help="显示监控命令")
    parser.add_argument("--file-operations", action="store_true", help="显示文件操作命令")
    parser.add_argument("--troubleshooting", action="store_true", help="显示故障排除命令")
    parser.add_argument("--create-scripts", action="store_true", help="创建快速操作脚本")
    parser.add_argument("--all", action="store_true", help="显示所有命令")
    
    args = parser.parse_args()
    
    tool = ServerQuickCommands()
    
    if args.list_results:
        tool.list_training_results()
    elif args.show_config:
        tool.show_latest_config()
    elif args.training_commands:
        tool.generate_training_commands()
    elif args.monitoring_commands:
        tool.generate_monitoring_commands()
    elif args.file_operations:
        tool.generate_file_operations()
    elif args.troubleshooting:
        tool.generate_troubleshooting_commands()
    elif args.create_scripts:
        tool.create_quick_scripts()
    elif args.all:
        tool.list_training_results()
        print("\n")
        tool.show_latest_config()
        print("\n")
        tool.generate_training_commands()
        print("\n")
        tool.generate_monitoring_commands()
        print("\n")
        tool.generate_file_operations()
        print("\n")
        tool.generate_troubleshooting_commands()
    else:
        print("VIVTransformer 服务器快速命令工具")
        print("\n可用选项:")
        print("  --list-results      列出训练结果")
        print("  --show-config       显示最新配置")
        print("  --training-commands 显示训练命令")
        print("  --monitoring-commands 显示监控命令")
        print("  --file-operations   显示文件操作命令")
        print("  --troubleshooting   显示故障排除命令")
        print("  --create-scripts    创建快速操作脚本")
        print("  --all              显示所有命令")
        print("\n示例:")
        print("  python3 server_quick_commands.py --list-results")
        print("  python3 server_quick_commands.py --all")

if __name__ == "__main__":
    main()