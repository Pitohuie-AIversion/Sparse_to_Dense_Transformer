#!/usr/bin/env python3
"""
多尺度超分辨率重构快速启动脚本

这个脚本提供了一个简单的入口点来快速开始多尺度实验。
无需复杂的路径配置，直接运行即可。

使用方法:
    python quick_start.py                    # 使用默认配置运行2x超分辨率
    python quick_start.py --scale 4          # 运行4x超分辨率
    python quick_start.py --test             # 仅运行测试
    python quick_start.py --config custom.yaml  # 使用自定义配置
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='多尺度超分辨率重构快速启动',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quick_start.py                           # 默认2x超分辨率
  python quick_start.py --scale 4                 # 4x超分辨率
  python quick_start.py --scale 2 --epochs 50     # 2x超分辨率，50个epoch
  python quick_start.py --test                    # 仅运行测试
  python quick_start.py --config configs/multiscale_config.yaml  # 使用配置文件
        """
    )
    
    parser.add_argument(
        '--scale', 
        type=int, 
        default=2, 
        choices=[2, 4, 8, 16],
        help='缩放因子 (默认: 2)'
    )
    
    parser.add_argument(
        '--data_path', 
        type=str,
        default='data/pdebench/darcy_flow_beta_0.01.h5',
        help='数据文件路径'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='训练轮数 (默认: 100)'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int,
        help='批次大小 (默认: 根据缩放因子自动设置)'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='配置文件路径'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='仅运行测试，不进行训练'
    )
    
    parser.add_argument(
        '--generate_config', 
        action='store_true',
        help='仅生成配置文件，不运行训练'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str,
        help='输出目录 (默认: results/multiscale_<scale>x)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.test:
            # 运行测试
            logger.info("开始运行多尺度测试...")
            from testing.test_multiscale import run_comprehensive_test
            
            config_path = args.config or 'configs/multiscale_config.yaml'
            success = run_comprehensive_test(config_path)
            
            if success:
                logger.info("✓ 测试完成！")
            else:
                logger.error("✗ 测试失败！")
                sys.exit(1)
                
        elif args.generate_config:
            # 生成配置文件
            logger.info(f"生成 {args.scale}x 超分辨率配置文件...")
            from examples.run_multiscale import create_default_config, save_config
            
            config = create_default_config(
                scale_factor=args.scale,
                data_path=args.data_path
            )
            
            # 更新配置
            if args.epochs:
                config['training']['epochs'] = args.epochs
            if args.batch_size:
                config['data']['batch_size'] = args.batch_size
            if args.output_dir:
                config['output']['save_dir'] = args.output_dir
            
            config_path = f'configs/multiscale_{args.scale}x_config.yaml'
            save_config(config, config_path)
            
            logger.info(f"✓ 配置文件已生成: {config_path}")
            
        else:
            # 运行训练
            if args.config:
                # 使用指定配置文件
                logger.info(f"使用配置文件: {args.config}")
                config_path = args.config
            else:
                # 生成默认配置
                logger.info(f"生成 {args.scale}x 超分辨率默认配置...")
                from examples.run_multiscale import create_default_config, save_config
                
                config = create_default_config(
                    scale_factor=args.scale,
                    data_path=args.data_path
                )
                
                # 更新配置
                if args.epochs:
                    config['training']['epochs'] = args.epochs
                if args.batch_size:
                    config['data']['batch_size'] = args.batch_size
                if args.output_dir:
                    config['output']['save_dir'] = args.output_dir
                
                config_path = f'configs/temp_multiscale_{args.scale}x.yaml'
                save_config(config, config_path)
            
            # 开始训练
            logger.info(f"开始 {args.scale}x 超分辨率训练...")
            from examples.run_multiscale import run_training
            
            success = run_training(config_path)
            
            if success:
                logger.info("✓ 训练完成！")
            else:
                logger.error("✗ 训练失败！")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(0)
    except Exception as e:
        logger.error(f"运行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()