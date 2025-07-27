#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查流体数据增强配置
"""

import yaml
import os

def check_augmentation_configs():
    """检查所有配置文件中的数据增强设置"""
    configs = [
        'configs/darcy_flow_config.yaml',
        'configs/custom_dataset_config.yaml', 
        'configs/config_test.yaml',
        'configs/config.yaml'
    ]
    
    print("=== 流体数据增强配置检查 ===")
    print()
    
    for cfg_path in configs:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                data_config = config.get('data', {})
                use_augmentation = data_config.get('use_augmentation', '未设置')
                
                # 检查training.data_augmentation.enabled（DarcyFlow配置风格）
                training_config = config.get('training', {})
                data_aug_config = training_config.get('data_augmentation', {})
                training_enabled = data_aug_config.get('enabled', '未设置')
                
                # 综合判断
                if use_augmentation is False or training_enabled is False:
                    status = "✅ 正确"
                    aug_status = f"data.use_augmentation={use_augmentation}, training.data_augmentation.enabled={training_enabled}"
                elif use_augmentation is True or training_enabled is True:
                    status = "❌ 需要修正"
                    aug_status = f"data.use_augmentation={use_augmentation}, training.data_augmentation.enabled={training_enabled}"
                else:
                    status = "⚠️ 未设置"
                    aug_status = f"data.use_augmentation={use_augmentation}, training.data_augmentation.enabled={training_enabled}"
                
                print(f"{cfg_path}: {aug_status} {status}")
                
            except Exception as e:
                print(f"{cfg_path}: 读取失败 - {e}")
        else:
            print(f"{cfg_path}: 文件不存在")
    
    print()
    print("=== PDEBench扩展配置检查 ===")
    
    pdebench_configs = [
        'pdebench_extended/configs/base/pdebench_extended_config.yaml',
        'pdebench_extended/configs/development/config_test.yaml',
        'pdebench_extended/configs/base/config.yaml'
    ]
    
    for cfg_path in pdebench_configs:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                data_config = config.get('data', {})
                use_augmentation = data_config.get('use_augmentation', '未设置')
                
                status = "✅ 正确" if use_augmentation is False else "❌ 需要修正" if use_augmentation is True else "⚠️ 未设置"
                print(f"{cfg_path}: use_augmentation = {use_augmentation} {status}")
                
            except Exception as e:
                print(f"{cfg_path}: 读取失败 - {e}")
        else:
            print(f"{cfg_path}: 文件不存在")
    
    print()
    print("=== 总结 ===")
    print("✅ 所有流体相关配置文件都应该设置 use_augmentation: false")
    print("❌ 如果发现 use_augmentation: true，需要立即修正")
    print("⚠️ 如果未设置，建议明确设置为 false")

if __name__ == "__main__":
    check_augmentation_configs()