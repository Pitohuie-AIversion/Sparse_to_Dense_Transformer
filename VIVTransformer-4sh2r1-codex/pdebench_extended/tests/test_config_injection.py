#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置参数注入测试脚本
用于验证配置文件参数是否正确加载和打印
"""

import argparse
import json
import yaml
import os
from datetime import datetime
from pathlib import Path

def test_config_loading():
    """测试配置加载功能"""
    parser = argparse.ArgumentParser(description='配置参数注入测试')
    parser.add_argument('--config_file', type=str, help='配置文件路径')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    
    args = parser.parse_args()
    
    # 加载配置文件
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r', encoding='utf-8') as f:
            if args.config_file.endswith('.yaml') or args.config_file.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        print(f"✅ 从配置文件加载: {args.config_file}")
    else:
        print(f"❌ 配置文件不存在: {args.config_file}")
        return
    
    # 打印配置参数
    print("\n" + "=" * 80)
    print("📋 配置参数注入测试")
    print("=" * 80)
    
    # 数据配置
    print("\n🔧 数据配置:")
    data_config = config.get('data', {})
    print(f"  - 数据路径: {data_config.get('path', 'N/A')}")
    print(f"  - 批次大小: {data_config.get('batch_size', 'N/A')}")
    print(f"  - 原始分辨率: {data_config.get('original_resolution', 'N/A')}")
    print(f"  - 缩放因子: {data_config.get('scale_factor', 'N/A')}")
    print(f"  - 输入分辨率: {data_config.get('center_crop_input_resolution', 'N/A')}")
    print(f"  - 输出分辨率: {data_config.get('center_crop_output_resolution', 'N/A')}")
    
    # 数据增强配置
    print("\n🎨 数据增强配置:")
    aug_config = config.get('data_augmentation', {})
    print(f"  - 启用数据增强: {aug_config.get('enabled', 'N/A')}")
    print(f"  - 随机裁剪: {aug_config.get('enable_random_crop', 'N/A')}")
    print(f"  - 翻转: {aug_config.get('enable_flip', 'N/A')}")
    print(f"  - 噪声水平: {aug_config.get('noise_level', 'N/A')}")
    
    # 模型配置
    print("\n🧠 模型配置:")
    model_config = config.get('model', {})
    print(f"  - 注意力类型: {model_config.get('attention_type', 'N/A')}")
    print(f"  - 隐藏维度: {model_config.get('hidden_dim', 'N/A')}")
    print(f"  - 层数: {model_config.get('num_layers', 'N/A')}")
    print(f"  - 注意力头数: {model_config.get('num_heads', 'N/A')}")
    print(f"  - Dropout: {model_config.get('dropout', 'N/A')}")
    print(f"  - 激活函数: {model_config.get('activation', 'N/A')}")
    
    # 损失函数配置
    print("\n📊 损失函数配置:")
    loss_config = config.get('loss', {})
    print(f"  - 损失类型: {loss_config.get('type', 'N/A')}")
    svd_config = loss_config.get('svd_config', {})
    if svd_config:
        print(f"  - SVD基础权重: {svd_config.get('base_weight', 'N/A')}")
        print(f"  - SVD权重: {svd_config.get('svd_weights', 'N/A')}")
        print(f"  - TopK: {svd_config.get('topk', 'N/A')}")
    
    # 训练配置
    print("\n🏃 训练配置:")
    train_config = config.get('training', {})
    print(f"  - 训练轮数: {train_config.get('epochs', 'N/A')}")
    print(f"  - 学习率: {train_config.get('learning_rate', 'N/A')}")
    print(f"  - 权重衰减: {train_config.get('weight_decay', 'N/A')}")
    print(f"  - 最小学习率: {train_config.get('min_learning_rate', 'N/A')}")
    print(f"  - 梯度裁剪: {train_config.get('grad_clip_norm', 'N/A')}")
    
    # 调度器配置
    print("\n📈 调度器配置:")
    scheduler_config = config.get('scheduler', {})
    print(f"  - 调度器类型: {scheduler_config.get('type', 'N/A')}")
    print(f"  - T_0: {scheduler_config.get('T_0', 'N/A')}")
    print(f"  - T_mult: {scheduler_config.get('T_mult', 'N/A')}")
    print(f"  - eta_min: {scheduler_config.get('eta_min', 'N/A')}")
    
    # 命令行参数覆盖测试
    print("\n🔄 命令行参数覆盖测试:")
    print(f"  - 命令行epochs参数: {args.num_epochs}")
    print(f"  - 命令行batch_size参数: {args.batch_size}")
    
    # 模拟参数覆盖
    if args.num_epochs != 10:  # 10是默认值
        train_config['epochs'] = args.num_epochs
        print(f"  ✅ epochs参数已覆盖为: {args.num_epochs}")
    
    if args.batch_size != 8:  # 8是默认值
        data_config['batch_size'] = args.batch_size
        print(f"  ✅ batch_size参数已覆盖为: {args.batch_size}")
    
    print("\n" + "=" * 80)
    print("✅ 配置参数注入测试完成！")
    print("=" * 80)
    
    return config

if __name__ == "__main__":
    test_config_loading()