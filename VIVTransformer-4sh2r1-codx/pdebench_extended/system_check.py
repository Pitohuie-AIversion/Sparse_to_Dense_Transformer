#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统检查脚本 - Windows/Linux兼容
检查训练环境的完整性和GPU状态
"""

import os
import sys
import platform
import subprocess
import psutil
from pathlib import Path
import time
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# 颜色定义
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'
    
    @classmethod
    def disable_on_windows(cls):
        """在Windows命令行中禁用颜色"""
        if platform.system() == 'Windows':
            for attr in dir(cls):
                if not attr.startswith('_') and attr != 'disable_on_windows':
                    setattr(cls, attr, '')

# 在Windows中禁用颜色（除非使用支持ANSI的终端）
if platform.system() == 'Windows' and 'ANSICON' not in os.environ:
    Colors.disable_on_windows()

def print_header(title):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== {title} ==={Colors.NC}")

def print_success(message):
    """打印成功信息"""
    print(f"{Colors.GREEN}✓{Colors.NC} {message}")

def print_warning(message):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠️{Colors.NC} {message}")

def print_error(message):
    """打印错误信息"""
    print(f"{Colors.RED}❌{Colors.NC} {message}")

def print_info(message):
    """打印信息"""
    print(f"{Colors.CYAN}ℹ️{Colors.NC} {message}")

def get_gpu_info():
    """获取GPU信息"""
    gpu_info = []
    
    if not TORCH_AVAILABLE:
        return gpu_info
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i)
            }
            gpu_info.append(gpu)
    
    return gpu_info

def get_nvidia_smi_info():
    """获取nvidia-smi信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpu_info.append({
                            'index': parts[0],
                            'name': parts[1],
                            'memory_used': int(parts[2]),
                            'memory_total': int(parts[3]),
                            'utilization': int(parts[4]),
                            'temperature': int(parts[5])
                        })
            return gpu_info
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return []

def check_python_environment():
    """检查Python环境"""
    print_header("Python环境检查")
    
    # Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_success(f"Python版本: {python_version}")
    
    # 平台信息
    print_info(f"操作系统: {platform.system()} {platform.release()}")
    print_info(f"架构: {platform.machine()}")
    
    # 检查必要的包
    packages = {
        'torch': TORCH_AVAILABLE,
        'numpy': NUMPY_AVAILABLE,
    }
    
    for package, available in packages.items():
        if available:
            if package == 'torch':
                print_success(f"{package}: {torch.__version__}")
            elif package == 'numpy':
                print_success(f"{package}: {np.__version__}")
        else:
            print_error(f"{package}: 未安装")
    
    # 检查其他包
    optional_packages = ['tensorboard', 'matplotlib', 'tqdm', 'pyyaml']
    for package in optional_packages:
        try:
            __import__(package)
            print_success(f"{package}: 已安装")
        except ImportError:
            print_warning(f"{package}: 未安装 (可选)")

def check_gpu_status():
    """检查GPU状态"""
    print_header("GPU状态检查")
    
    # 检查PyTorch CUDA支持
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            print_success(f"CUDA可用: {torch.version.cuda}")
            print_success(f"GPU数量: {torch.cuda.device_count()}")
            
            # PyTorch GPU信息
            gpu_info = get_gpu_info()
            for gpu in gpu_info:
                memory_used_gb = gpu['memory_allocated'] / (1024**3)
                memory_total_gb = gpu['memory_total'] / (1024**3)
                memory_usage_percent = (gpu['memory_allocated'] / gpu['memory_total']) * 100
                
                print_info(f"GPU {gpu['index']}: {gpu['name']}")
                print_info(f"  显存: {memory_used_gb:.1f}GB / {memory_total_gb:.1f}GB ({memory_usage_percent:.1f}%)")
        else:
            print_warning("CUDA不可用")
    else:
        print_error("PyTorch未安装，无法检查CUDA")
    
    # 检查nvidia-smi
    nvidia_info = get_nvidia_smi_info()
    if nvidia_info:
        print_success("nvidia-smi可用")
        for gpu in nvidia_info:
            memory_usage_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            
            # 状态判断
            if memory_usage_percent < 10 and gpu['utilization'] < 10:
                status = f"{Colors.GREEN}空闲{Colors.NC}"
            elif memory_usage_percent < 50 and gpu['utilization'] < 50:
                status = f"{Colors.YELLOW}轻载{Colors.NC}"
            else:
                status = f"{Colors.RED}重载{Colors.NC}"
            
            print_info(f"GPU {gpu['index']}: {gpu['name']}")
            print_info(f"  显存: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({memory_usage_percent:.1f}%)")
            print_info(f"  利用率: {gpu['utilization']}%")
            print_info(f"  温度: {gpu['temperature']}°C")
            print_info(f"  状态: {status}")
    else:
        print_warning("nvidia-smi不可用")

def check_system_resources():
    """检查系统资源"""
    print_header("系统资源检查")
    
    # CPU信息
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=1)
    
    print_info(f"CPU核心: {cpu_count} 物理核心, {cpu_count_logical} 逻辑核心")
    print_info(f"CPU使用率: {cpu_usage}%")
    
    # 内存信息
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024**3)
    memory_used_gb = memory.used / (1024**3)
    memory_usage_percent = memory.percent
    
    print_info(f"内存: {memory_used_gb:.1f}GB / {memory_total_gb:.1f}GB ({memory_usage_percent:.1f}%)")
    
    # 磁盘信息
    disk = psutil.disk_usage('.')
    disk_total_gb = disk.total / (1024**3)
    disk_used_gb = disk.used / (1024**3)
    disk_usage_percent = (disk.used / disk.total) * 100
    
    print_info(f"磁盘: {disk_used_gb:.1f}GB / {disk_total_gb:.1f}GB ({disk_usage_percent:.1f}%)")
    
    # 负载平均值（仅Linux）
    if hasattr(os, 'getloadavg'):
        load_avg = os.getloadavg()
        print_info(f"负载平均值: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")

def check_training_files():
    """检查训练相关文件"""
    print_header("训练文件检查")
    
    # 检查数据文件
    data_files = [
        'data/pressure_field_data.pt',
        'data/pressure_field_data.h5',
        'data/pressure_field_data.npz'
    ]
    
    data_found = False
    for data_file in data_files:
        if Path(data_file).exists():
            file_size = Path(data_file).stat().st_size / (1024**2)  # MB
            print_success(f"数据文件: {data_file} ({file_size:.1f}MB)")
            data_found = True
            break
    
    if not data_found:
        print_warning("未找到数据文件")
    
    # 检查配置文件
    config_files = [
        'configs/pressure_field_training.yaml',
        'config.yaml',
        'pressure_field_training.yaml'
    ]
    
    config_found = False
    for config_file in config_files:
        if Path(config_file).exists():
            print_success(f"配置文件: {config_file}")
            config_found = True
            break
    
    if not config_found:
        print_warning("未找到配置文件")
    
    # 检查训练脚本
    train_scripts = [
        'train_pressure_field.py',
        'train.py',
        'main.py'
    ]
    
    script_found = False
    for script in train_scripts:
        if Path(script).exists():
            print_success(f"训练脚本: {script}")
            script_found = True
            break
    
    if not script_found:
        print_warning("未找到训练脚本")
    
    # 检查输出目录
    output_dirs = ['outputs', 'results', 'checkpoints']
    for output_dir in output_dirs:
        if Path(output_dir).exists():
            print_success(f"输出目录: {output_dir}")
        else:
            print_info(f"输出目录不存在: {output_dir} (将自动创建)")

def check_running_processes():
    """检查运行中的训练进程"""
    print_header("运行进程检查")
    
    training_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if any(keyword in cmdline.lower() for keyword in ['train', 'python.*train', 'tensorboard']):
                training_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmdline': cmdline[:80] + '...' if len(cmdline) > 80 else cmdline,
                    'cpu_percent': proc.info['cpu_percent'],
                    'memory_percent': proc.info['memory_percent']
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if training_processes:
        print_info(f"发现 {len(training_processes)} 个相关进程:")
        for proc in training_processes:
            print_info(f"  PID {proc['pid']}: {proc['name']}")
            print_info(f"    命令: {proc['cmdline']}")
            print_info(f"    CPU: {proc['cpu_percent']:.1f}%, 内存: {proc['memory_percent']:.1f}%")
    else:
        print_info("未发现训练相关进程")

def generate_report():
    """生成系统报告"""
    print_header("生成系统报告")
    
    report_file = f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"系统检查报告\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"="*50 + "\n\n")
        
        # 重定向输出到文件
        import io
        import contextlib
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            # 重新运行所有检查（不带颜色）
            Colors.disable_on_windows()
            check_python_environment()
            check_gpu_status()
            check_system_resources()
            check_training_files()
            check_running_processes()
        finally:
            sys.stdout = old_stdout
        
        # 清理ANSI颜色代码
        import re
        content = buffer.getvalue()
        content = re.sub(r'\x1b\[[0-9;]*m', '', content)
        f.write(content)
    
    print_success(f"系统报告已生成: {report_file}")

def main():
    """主函数"""
    print(f"{Colors.BOLD}{Colors.PURPLE}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                          系统检查工具 v1.0                                  ║")
    print("║                     VIV Transformer 训练环境检查                            ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.NC}")
    
    print_info(f"检查时间: {datetime.now()}")
    print_info(f"当前目录: {os.getcwd()}")
    
    try:
        check_python_environment()
        check_gpu_status()
        check_system_resources()
        check_training_files()
        check_running_processes()
        
        print_header("检查完成")
        
        # 询问是否生成报告
        if len(sys.argv) > 1 and '--report' in sys.argv:
            generate_report()
        else:
            response = input(f"\n{Colors.YELLOW}是否生成详细报告? (y/N): {Colors.NC}")
            if response.lower() in ['y', 'yes']:
                generate_report()
        
        print_success("系统检查完成!")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}检查被用户中断{Colors.NC}")
    except Exception as e:
        print_error(f"检查过程中出现错误: {e}")

if __name__ == '__main__':
    main()