#!/usr/bin/env python3
"""多尺度超分辨率重构运行脚本

这个脚本提供了一个简单的接口来运行多尺度超分辨率重构实验。
支持不同的缩放因子和配置选项。

使用方法:
    python run_multiscale.py --scale_factor 2 --data_path data/pdebench/darcy_flow_beta_0.01.h5
    python run_multiscale.py --config configs/multiscale_config.yaml
    python run_multiscale.py --test_only  # 仅运行测试
"""

import argparse
import sys
import yaml
from pathlib import Path
import logging
from typing import Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_default_config(scale_factor: int = 2, data_path: str = None) -> Dict[str, Any]:
    """创建默认配置
    
    Args:
        scale_factor: 缩放因子
        data_path: 数据文件路径
        
    Returns:
        默认配置字典
    """
    # 根据缩放因子计算维度
    original_res = [128, 128]
    low_res = [original_res[0] // scale_factor, original_res[1] // scale_factor]
    input_dim = low_res[0] * low_res[1] * 1  # 假设单通道
    output_dim = original_res[0] * original_res[1] * 1
    
    config = {
        'project_name': 'VIVTransformer_MultiScale',
        'experiment_name': f'darcy_flow_super_resolution_{scale_factor}x',
        'device': 'auto',
        'seed': 42,
        
        'data': {
            'use_multiscale': True,
            'data_path': data_path or 'data/pdebench/darcy_flow_beta_0.01.h5',
            'pde_type': 'darcy_flow',
            'scale_factor': scale_factor,
            'original_resolution': original_res,
            'downsampling_method': 'average',
            'sequence_length': 1,
            'normalize': True,
            'batch_size': 16 if scale_factor <= 2 else 32,
            'num_workers': 4,
            'pin_memory': True
        },
        
        'model': {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'activation': 'relu',
            'use_data_parallel': True
        },
        
        'training': {
            'epochs': 100,
            'learning_rate': 0.0001,
            'weight_decay': 0.0001,
            'scheduler': {
                'type': 'StepLR',
                'step_size': 30,
                'gamma': 0.5
            },
            'early_stopping': {
                'patience': 15,
                'min_delta': 0.0001
            },
            'save_checkpoint': True,
            'checkpoint_interval': 10,
            'save_best_only': True
        },
        
        'loss': {
            'primary': {
                'type': 'MSELoss',
                'weight': 1.0
            }
        },
        
        'attention_mechanisms': ['standard'],
        
        'evaluation': {
            'metrics': ['mse', 'mae'],
            'visualization': {
                'enabled': True,
                'save_predictions': True,
                'num_samples': 5,
                'save_path': f'results/multiscale_predictions_{scale_factor}x'
            }
        },
        
        'logging': {
            'level': 'INFO',
            'save_logs': True,
            'log_dir': f'logs/multiscale_{scale_factor}x',
            'tensorboard': False,
            'wandb': {
                'enabled': False
            }
        },
        
        'output': {
            'save_dir': f'results/multiscale_{scale_factor}x',
            'save_model': True,
            'save_predictions': True,
            'save_plots': True
        }
    }
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    logger.info(f"配置已保存到: {config_file}")


def run_training(config_path: str):
    """运行训练
    
    Args:
        config_path: 配置文件路径
    """
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from multiscale.training.train_multiscale import MultiScaleTrainer, load_config
        
        # 加载配置
        config = load_config(config_path)
        
        # 创建训练器并开始训练
        trainer = MultiScaleTrainer(config)
        model = trainer.train()
        
        logger.info("训练完成！")
        return True
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        return False


def run_test(config_path: str):
    """运行测试
    
    Args:
        config_path: 配置文件路径
    """
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from multiscale.testing.test_multiscale import run_comprehensive_test
        
        # 运行测试
        results = run_comprehensive_test(config_path)
        
        if results:
            logger.info("测试完成！")
            return True
        else:
            logger.error("测试失败")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='多尺度超分辨率重构实验运行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认配置运行2倍缩放实验
  python run_multiscale.py --scale_factor 2
  
  # 指定数据路径
  python run_multiscale.py --scale_factor 4 --data_path data/my_data.h5
  
  # 使用自定义配置文件
  python run_multiscale.py --config configs/my_config.yaml
  
  # 仅运行测试
  python run_multiscale.py --test_only --config configs/multiscale_config.yaml
  
  # 生成配置文件但不运行
  python run_multiscale.py --scale_factor 8 --generate_config_only
        """
    )
    
    parser.add_argument(
        '--scale_factor', 
        type=int, 
        default=2,
        choices=[2, 4, 8, 16],
        help='缩放因子 (默认: 2)'
    )
    
    parser.add_argument(
        '--data_path', 
        type=str,
        help='数据文件路径'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--test_only', 
        action='store_true',
        help='仅运行测试，不进行训练'
    )
    
    parser.add_argument(
        '--generate_config_only', 
        action='store_true',
        help='仅生成配置文件，不运行实验'
    )
    
    parser.add_argument(
        '--output_config', 
        type=str,
        help='输出配置文件路径 (默认: configs/multiscale_config_<scale_factor>x.yaml)'
    )
    
    args = parser.parse_args()
    
    # 确定配置文件路径
    if args.config:
        config_path = args.config
        if not Path(config_path).exists():
            logger.error(f"配置文件不存在: {config_path}")
            return 1
    else:
        # 生成默认配置
        config = create_default_config(args.scale_factor, args.data_path)
        
        # 确定输出配置文件路径
        if args.output_config:
            config_path = args.output_config
        else:
            config_path = f"configs/multiscale_config_{args.scale_factor}x.yaml"
        
        # 保存配置
        save_config(config, config_path)
        
        logger.info(f"使用缩放因子: {args.scale_factor}")
        logger.info(f"输入维度: {config['model']['input_dim']}")
        logger.info(f"输出维度: {config['model']['output_dim']}")
    
    # 如果只生成配置文件，则退出
    if args.generate_config_only:
        logger.info("配置文件生成完成")
        return 0
    
    # 检查数据文件是否存在
    if args.config:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        data_path = config.get('data', {}).get('data_path')
    else:
        data_path = args.data_path or 'data/pdebench/darcy_flow_beta_0.01.h5'
    
    if data_path and not Path(data_path).exists():
        logger.warning(f"数据文件不存在: {data_path}")
        logger.info("请确保数据文件存在，或者使用测试模式")
        
        # 询问是否继续
        response = input("是否继续运行测试？(y/N): ")
        if response.lower() != 'y':
            return 1
    
    # 运行实验
    success = True
    
    if args.test_only:
        logger.info("运行测试模式...")
        success = run_test(config_path)
    else:
        logger.info("运行训练模式...")
        success = run_training(config_path)
    
    if success:
        logger.info("实验完成！")
        return 0
    else:
        logger.error("实验失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())