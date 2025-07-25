#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多尺度训练启动脚本
支持多种预设配置和自定义参数

使用示例:
1. 使用默认配置:
   python run_multiscale_training.py

2. 指定缩放因子:
   python run_multiscale_training.py --scale_factor 2

3. 使用预设配置:
   python run_multiscale_training.py --preset scale_factor_8

4. 服务器高性能配置:
   python run_multiscale_training.py --preset server_config_high_performance

5. 自定义配置文件:
   python run_multiscale_training.py --config my_config.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from train_configurable_multiscale import ConfigurableMultiScaleTrainer, create_config


def load_preset_config(preset_name: str) -> Dict[str, Any]:
    """加载预设配置"""
    config_file = Path(__file__).parent / 'configs' / 'multiscale_configs.json'
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)
    
    if preset_name not in all_configs:
        available_presets = list(all_configs.keys())
        raise ValueError(f"预设配置 '{preset_name}' 不存在。可用预设: {available_presets}")
    
    return all_configs[preset_name]


def print_available_presets():
    """打印可用的预设配置"""
    config_file = Path(__file__).parent / 'configs' / 'multiscale_configs.json'
    
    if not config_file.exists():
        print("配置文件不存在")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        all_configs = json.load(f)
    
    print("\n可用的预设配置:")
    print("=" * 50)
    
    for preset_name, config in all_configs.items():
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        
        print(f"\n📋 {preset_name}:")
        print(f"   缩放因子: {data_config.get('scale_factor', 'N/A')}")
        print(f"   输入分辨率: {data_config.get('center_crop_input_resolution', 'N/A')}")
        print(f"   输出分辨率: {data_config.get('center_crop_output_resolution', 'N/A')}")
        print(f"   批次大小: {data_config.get('batch_size', 'N/A')}")
        print(f"   训练轮数: {training_config.get('num_epochs', 'N/A')}")
        print(f"   学习率: {training_config.get('learning_rate', 'N/A')}")
    
    print("\n" + "=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='多尺度VIVTransformer训练启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_multiscale_training.py --scale_factor 4
  python run_multiscale_training.py --preset scale_factor_8
  python run_multiscale_training.py --config my_config.json
  python run_multiscale_training.py --list_presets
        """
    )
    
    # 配置选项
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument('--preset', type=str, help='使用预设配置')
    config_group.add_argument('--config', type=str, help='使用自定义配置文件')
    
    # 快速参数
    parser.add_argument('--scale_factor', type=int, help='缩放因子 (当不使用preset或config时)')
    parser.add_argument('--input_resolution', type=int, nargs=2, help='输入分辨率 [H, W]')
    parser.add_argument('--output_resolution', type=int, nargs=2, help='输出分辨率 [H, W]')
    parser.add_argument('--num_epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    
    # 其他选项
    parser.add_argument('--list_presets', action='store_true', help='列出所有可用的预设配置')
    parser.add_argument('--dry_run', action='store_true', help='只显示配置，不开始训练')
    parser.add_argument('--gpu', type=int, help='指定GPU设备ID')
    parser.add_argument('--yes', action='store_true', help='自动确认开始训练，跳过用户输入')
    
    args = parser.parse_args()
    
    # 列出预设配置
    if args.list_presets:
        print_available_presets()
        return
    
    # 设置GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"设置使用GPU: {args.gpu}")
    
    # 加载配置
    config = None
    
    if args.preset:
        print(f"使用预设配置: {args.preset}")
        config = load_preset_config(args.preset)
        
    elif args.config:
        print(f"使用自定义配置文件: {args.config}")
        if not os.path.exists(args.config):
            print(f"错误: 配置文件不存在: {args.config}")
            return
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
    else:
        # 使用命令行参数创建配置
        print("使用命令行参数创建配置")
        config = create_config(
            scale_factor=args.scale_factor or 4,
            input_resolution=args.input_resolution or [96, 96],
            output_resolution=args.output_resolution or [112, 112],
            num_epochs=args.num_epochs or 10,
            batch_size=args.batch_size or 8
        )
        
        # 应用命令行覆盖
        if args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
    
    # 显示配置信息
    print("\n" + "=" * 80)
    print("🚀 多尺度VIVTransformer训练配置")
    print("=" * 80)
    
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    print(f"📊 数据配置:")
    print(f"   数据路径: {data_config.get('data_path', 'N/A')}")
    print(f"   原始分辨率: {data_config.get('original_resolution', 'N/A')}")
    print(f"   缩放因子: {data_config.get('scale_factor', 'N/A')}")
    print(f"   输入分辨率: {data_config.get('center_crop_input_resolution', 'N/A')}")
    print(f"   输出分辨率: {data_config.get('center_crop_output_resolution', 'N/A')}")
    print(f"   批次大小: {data_config.get('batch_size', 'N/A')}")
    print(f"   归一化: {data_config.get('normalize', 'N/A')}")
    
    print(f"\n🧠 模型配置:")
    print(f"   隐藏维度: {model_config.get('hidden_dim', 'N/A')}")
    print(f"   层数: {model_config.get('num_layers', 'N/A')}")
    print(f"   注意力头数: {model_config.get('num_heads', 'N/A')}")
    print(f"   Dropout: {model_config.get('dropout', 'N/A')}")
    
    print(f"\n🎯 训练配置:")
    print(f"   训练轮数: {training_config.get('num_epochs', 'N/A')}")
    print(f"   学习率: {training_config.get('learning_rate', 'N/A')}")
    print(f"   权重衰减: {training_config.get('weight_decay', 'N/A')}")
    print(f"   梯度裁剪: {training_config.get('grad_clip_norm', 'N/A')}")
    
    print(f"\n💾 输出目录: {config.get('output_dir', 'N/A')}")
    print("=" * 80)
    
    # 干运行模式
    if args.dry_run:
        print("\n🔍 干运行模式 - 仅显示配置，不开始训练")
        
        # 保存配置到临时文件
        temp_config_file = Path('temp_config.json')
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {temp_config_file}")
        return
    
    # 确认开始训练
    if not args.yes:
        try:
            response = input("\n是否开始训练? (y/N): ").strip().lower()
            if response not in ['y', 'yes', '是']:
                print("训练已取消")
                return
        except KeyboardInterrupt:
            print("\n训练已取消")
            return
    else:
        print("\n✅ 自动确认开始训练")
    
    # 创建训练器并开始训练
    try:
        print("\n🎯 创建训练器...")
        trainer = ConfigurableMultiScaleTrainer(config)
        
        print("🚀 开始训练...")
        trainer.train()
        
        print("\n🎉 训练成功完成！")
        print(f"📁 结果保存在: {config['output_dir']}")
        print("📊 可以使用以下命令启动TensorBoard查看训练过程:")
        print(f"   tensorboard --logdir {config['output_dir']}/tensorboard")
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存错误配置以便调试
        error_config_file = Path('error_config.json')
        with open(error_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"错误时的配置已保存到: {error_config_file}")


if __name__ == "__main__":
    main()