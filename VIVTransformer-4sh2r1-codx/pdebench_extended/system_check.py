#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Check Script - Windows/Linux Compatible
Check the integrity of the training environment and GPU status
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

# Color definitions
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
        """Disable ANSI colors in Windows command line"""
        if platform.system() == 'Windows':
            for attr in dir(cls):
                if not attr.startswith('_') and attr != 'disable_on_windows':
                    setattr(cls, attr, '')

# Disable colors on Windows unless ANSI is supported
if platform.system() == 'Windows' and 'ANSICON' not in os.environ:
    Colors.disable_on_windows()

def print_header(title):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== {title} ==={Colors.NC}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.NC} {message}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️{Colors.NC} {message}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌{Colors.NC} {message}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️{Colors.NC} {message}")

def get_gpu_info():
    """Get GPU info via PyTorch"""
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
    """Get GPU info via nvidia-smi"""
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
    """Check Python environment"""
    print_header("Python Environment Check")
    
    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print_success(f"Python version: {python_version}")
    
    # Platform info
    print_info(f"Operating System: {platform.system()} {platform.release()}")
    print_info(f"Architecture: {platform.machine()}")
    
    # Required packages
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
            print_error(f"{package}: not installed")
    
    # Optional packages
    optional_packages = ['tensorboard', 'matplotlib', 'tqdm', 'pyyaml']
    for package in optional_packages:
        try:
            __import__(package)
            print_success(f"{package}: installed")
        except ImportError:
            print_warning(f"{package}: not installed (optional)")

def check_gpu_status():
    """Check GPU status"""
    print_header("GPU Status Check")
    
    # Check PyTorch CUDA support
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.version.cuda}")
            print_success(f"GPU count: {torch.cuda.device_count()}")
            
            # PyTorch GPU info
            gpu_info = get_gpu_info()
            for gpu in gpu_info:
                memory_used_gb = gpu['memory_allocated'] / (1024**3)
                memory_total_gb = gpu['memory_total'] / (1024**3)
                memory_usage_percent = (gpu['memory_allocated'] / gpu['memory_total']) * 100
                
                print_info(f"GPU {gpu['index']}: {gpu['name']}")
                print_info(f"  Memory: {memory_used_gb:.1f}GB / {memory_total_gb:.1f}GB ({memory_usage_percent:.1f}%)")
        else:
            print_warning("CUDA not available")
    else:
        print_error("PyTorch not installed, unable to check CUDA")
    
    # Check nvidia-smi
    nvidia_info = get_nvidia_smi_info()
    if nvidia_info:
        print_success("nvidia-smi available")
        for gpu in nvidia_info:
            memory_usage_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            
            # Status assessment
            if memory_usage_percent < 10 and gpu['utilization'] < 10:
                status = f"{Colors.GREEN}Idle{Colors.NC}"
            elif memory_usage_percent < 50 and gpu['utilization'] < 50:
                status = f"{Colors.YELLOW}Light Load{Colors.NC}"
            else:
                status = f"{Colors.RED}Heavy Load{Colors.NC}"
            
            print_info(f"GPU {gpu['index']}: {gpu['name']}")
            print_info(f"  Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({memory_usage_percent:.1f}%)")
            print_info(f"  Utilization: {gpu['utilization']}%")
            print_info(f"  Temperature: {gpu['temperature']}°C")
            print_info(f"  Status: {status}")
    else:
        print_warning("nvidia-smi not available")

def check_system_resources():
    """Check system resources"""
    print_header("System Resources Check")
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=1)
    
    print_info(f"CPU Cores: {cpu_count} physical, {cpu_count_logical} logical")
    print_info(f"CPU Usage: {cpu_usage}%")
    
    # Memory info
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024**3)
    memory_used_gb = memory.used / (1024**3)
    memory_usage_percent = memory.percent
    
    print_info(f"Memory: {memory_used_gb:.1f}GB / {memory_total_gb:.1f}GB ({memory_usage_percent:.1f}%)")
    
    # Disk info
    disk = psutil.disk_usage('.')
    disk_total_gb = disk.total / (1024**3)
    disk_used_gb = disk.used / (1024**3)
    disk_usage_percent = (disk.used / disk.total) * 100
    
    print_info(f"Disk: {disk_used_gb:.1f}GB / {disk_total_gb:.1f}GB ({disk_usage_percent:.1f}%)")
    
    # Load average (Linux only)
    if hasattr(os, 'getloadavg'):
        load_avg = os.getloadavg()
        print_info(f"Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")

def check_training_files():
    """Check training-related files"""
    print_header("Training Files Check")
    
    # Check data files
    data_files = [
        'data/pressure_field_data.pt',
        'data/pressure_field_data.h5',
        'data/pressure_field_data.npz'
    ]
    
    data_found = False
    for data_file in data_files:
        if Path(data_file).exists():
            file_size = Path(data_file).stat().st_size / (1024**2)  # MB
            print_success(f"Data file: {data_file} ({file_size:.1f}MB)")
            data_found = True
            break
    
    if not data_found:
        print_warning("No data file found")
    
    # Check config files
    config_files = [
        'configs/pressure_field_training.yaml',
        'config.yaml',
        'pressure_field_training.yaml'
    ]
    
    config_found = False
    for config_file in config_files:
        if Path(config_file).exists():
            print_success(f"Config file: {config_file}")
            config_found = True
            break
    
    if not config_found:
        print_warning("No config file found")
    
    # Check training scripts
    train_scripts = [
        'train_pressure_field.py',
        'train.py',
        'main.py'
    ]
    
    script_found = False
    for script in train_scripts:
        if Path(script).exists():
            print_success(f"Training script: {script}")
            script_found = True
            break
    
    if not script_found:
        print_warning("No training script found")
    
    # Check output directories
    output_dirs = ['outputs', 'results', 'checkpoints']
    for output_dir in output_dirs:
        if Path(output_dir).exists():
            print_success(f"Output directory: {output_dir}")
        else:
            print_info(f"Output directory not found: {output_dir} (will be created automatically)")

def check_running_processes():
    """Check running training processes"""
    print_header("Running Processes Check")
    
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
        print_info(f"Found {len(training_processes)} related process(es):")
        for proc in training_processes:
            print_info(f"  PID {proc['pid']}: {proc['name']}")
            print_info(f"    Command: {proc['cmdline']}")
            print_info(f"    CPU: {proc['cpu_percent']:.1f}%, Memory: {proc['memory_percent']:.1f}%")
    else:
        print_info("No training-related process found")

def generate_report():
    """Generate system report"""
    print_header("Generate System Report")
    
    report_file = f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"System Check Report\n")
        f.write(f"Generated at: {datetime.now()}\n")
        f.write(f"="*50 + "\n\n")
        
        # Redirect output to buffer
        import io
        import contextlib
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            # Re-run all checks (without colors)
            Colors.disable_on_windows()
            check_python_environment()
            check_gpu_status()
            check_system_resources()
            check_training_files()
            check_running_processes()
        finally:
            sys.stdout = old_stdout
        
        # Strip ANSI color codes
        import re
        content = buffer.getvalue()
        content = re.sub(r'\x1b\[[0-9;]*m', '', content)
        f.write(content)
    
    print_success(f"System report generated: {report_file}")

def main():
    """Main function"""
    print(f"{Colors.BOLD}{Colors.PURPLE}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                          System Check Tool v1.0                             ║")
    print("║                 VIV Transformer Training Environment Check                  ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.NC}")
    
    print_info(f"Check time: {datetime.now()}")
    print_info(f"Current directory: {os.getcwd()}")
    
    try:
        check_python_environment()
        check_gpu_status()
        check_system_resources()
        check_training_files()
        check_running_processes()
        
        print_header("Check Completed")
        
        # Ask whether to generate report
        if len(sys.argv) > 1 and '--report' in sys.argv:
            generate_report()
        else:
            response = input(f"\n{Colors.YELLOW}Generate a detailed report? (y/N): {Colors.NC}")
            if response.lower() in ['y', 'yes']:
                generate_report()
        
        print_success("System check completed!")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Check interrupted by user{Colors.NC}")
    except Exception as e:
        print_error(f"Error occurred during check: {e}")

if __name__ == '__main__':
    main()