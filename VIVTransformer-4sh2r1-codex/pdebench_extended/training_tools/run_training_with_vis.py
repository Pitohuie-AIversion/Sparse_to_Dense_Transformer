#!/usr/bin/env python3
"""
快速启动带可视化功能的训练脚本

这个脚本提供了一个简单的方式来启动带有实时可视化的训练过程。
它会自动检查环境，设置必要的参数，并启动训练。
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """检查运行环境"""
    logger.info("检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    logger.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必要的包
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'seaborn', 
        'tqdm', 'tensorboard', 'yaml', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} 未安装")
    
    if missing_packages:
        logger.error(f"缺少必要的包: {missing_packages}")
        logger.info("请运行以下命令安装缺少的包:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA可用 - GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {gpu_name}")
        else:
            logger.info("⚠ CUDA不可用，将使用CPU训练")
    except Exception as e:
        logger.warning(f"检查CUDA时出错: {e}")
    
    return True


def find_config_file():
    """查找配置文件"""
    current_dir = Path.cwd()
    
    # 可能的配置文件位置
    config_paths = [
        current_dir / 'configs' / 'pressure_field_training.yaml',
        current_dir / 'pressure_field_training.yaml',
        current_dir / 'config.yaml',
        current_dir / 'training_config.yaml'
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            logger.info(f"找到配置文件: {config_path}")
            return str(config_path)
    
    logger.warning("未找到配置文件，将使用默认配置")
    return None


def create_default_config():
    """创建默认配置文件"""
    config_content = """
# 压力场重建训练配置
data:
  input_dim: 784  # 28x28 输入
  output_dim: 784  # 28x28 输出
  batch_size: 32

model:
  transformer:
    d_model: 512
    nhead: 8
    num_layers: 6
    dim_feedforward: 2048
    dropout: 0.1

training:
  num_epochs: 50
  learning_rate: 0.001
  batch_size: 32
  output_dir: './training_output'
  save_interval: 10
  validation_interval: 5

visualization:
  enabled: true
  save_interval: 5
  plot_types:
    - loss_curves
    - hardware_usage
    - prediction_comparison
  
logging:
  level: INFO
  save_logs: true

hardware_monitoring:
  enabled: true
  interval: 1.0
  log_gpu: true
  log_memory: true
  log_cpu: true
"""
    
    config_path = Path.cwd() / 'default_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    logger.info(f"创建默认配置文件: {config_path}")
    return str(config_path)


def start_tensorboard(log_dir: str, port: int = 6006):
    """启动TensorBoard"""
    try:
        logger.info(f"启动TensorBoard在端口 {port}...")
        cmd = f"tensorboard --logdir={log_dir} --port={port} --host=0.0.0.0"
        
        # 在Windows上使用start命令在新窗口中启动
        if os.name == 'nt':
            subprocess.Popen(f"start cmd /k {cmd}", shell=True)
        else:
            subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logger.info(f"TensorBoard已启动: http://localhost:{port}")
        return True
    except Exception as e:
        logger.warning(f"启动TensorBoard失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='快速启动带可视化功能的训练')
    parser.add_argument(
        '--config', 
        type=str, 
        help='配置文件路径（可选）'
    )
    parser.add_argument(
        '--no-realtime-vis', 
        action='store_true',
        help='禁用实时可视化窗口'
    )
    parser.add_argument(
        '--no-tensorboard', 
        action='store_true',
        help='禁用TensorBoard'
    )
    parser.add_argument(
        '--tensorboard-port', 
        type=int, 
        default=6006,
        help='TensorBoard端口号（默认6006）'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        help='训练轮数（覆盖配置文件设置）'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 压力场重建训练 - 带可视化功能")
    print("=" * 70)
    
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败，请安装必要的依赖包")
        return 1
    
    # 查找或创建配置文件
    config_path = args.config or find_config_file()
    if not config_path:
        config_path = create_default_config()
    
    # 检查训练脚本
    train_script = Path(__file__).parent / 'train_with_visualization.py'
    if not train_script.exists():
        logger.error(f"训练脚本不存在: {train_script}")
        return 1
    
    # 准备训练命令
    cmd = [sys.executable, str(train_script), '--config', config_path]
    
    if args.no_realtime_vis:
        cmd.append('--no-realtime-vis')
    
    if args.no_tensorboard:
        cmd.append('--no-tensorboard')
    
    # 启动TensorBoard（如果需要）
    if not args.no_tensorboard:
        # 从配置文件中读取输出目录
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            output_dir = config.get('training', {}).get('output_dir', './training_output')
            tb_dir = Path(output_dir) / 'tensorboard'
            tb_dir.mkdir(exist_ok=True, parents=True)
            start_tensorboard(str(tb_dir), args.tensorboard_port)
        except Exception as e:
            logger.warning(f"无法启动TensorBoard: {e}")
    
    # 显示启动信息
    logger.info("=" * 50)
    logger.info("🚀 启动训练")
    logger.info("=" * 50)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"实时可视化: {'禁用' if args.no_realtime_vis else '启用'}")
    logger.info(f"TensorBoard: {'禁用' if args.no_tensorboard else f'启用 (端口 {args.tensorboard_port})'}")
    
    if not args.no_tensorboard:
        logger.info(f"TensorBoard访问地址: http://localhost:{args.tensorboard_port}")
    
    logger.info("=" * 50)
    logger.info("训练过程中的提示:")
    logger.info("- 实时可视化窗口会显示训练进度")
    logger.info("- 按 Ctrl+C 可以安全停止训练")
    logger.info("- 训练结果会保存在输出目录中")
    logger.info("=" * 50)
    
    try:
        # 启动训练
        logger.info("开始训练...")
        result = subprocess.run(cmd, check=True)
        logger.info("训练成功完成！")
        return result.returncode
        
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"训练过程中出现错误: {e}")
        return e.returncode
    except Exception as e:
        logger.error(f"启动训练时出现错误: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)