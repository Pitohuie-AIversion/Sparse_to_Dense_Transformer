#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 验证数据处理功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from makedata200plus200origin_data_channel_aoming_mergedd_all_reynold4D_plot_normalize_afterextraction import (
    Config, DataProcessor, create_test_data, quick_process_data
)
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_create_test_data():
    """测试创建测试数据功能"""
    print("\n=== 测试创建测试数据 ===")
    try:
        config = Config()
        create_test_data(config)
        print("✓ 测试数据创建成功")
        return True
    except Exception as e:
        print(f"✗ 测试数据创建失败: {str(e)}")
        return False

def test_data_processing():
    """测试数据处理功能"""
    print("\n=== 测试数据处理 ===")
    try:
        config = Config()
        processor = DataProcessor(config, test_mode=True)
        
        # 验证文件格式
        print("验证文件格式...")
        if processor._validate_file_formats():
            print("✓ 文件格式验证通过")
        else:
            print("✗ 文件格式验证失败")
            return False
        
        # 处理数据
        print("处理数据...")
        success = processor.preprocess_and_save_data()
        if success:
            print("✓ 数据处理成功")
            return True
        else:
            print("✗ 数据处理失败")
            return False
            
    except Exception as e:
        print(f"✗ 数据处理过程中发生错误: {str(e)}")
        return False

def test_data_loading():
    """测试数据加载功能"""
    print("\n=== 测试数据加载 ===")
    try:
        from makedata200plus200origin_data_channel_aoming_mergedd_all_reynold4D_plot_normalize_afterextraction import DataVisualizer
        
        config = Config()
        visualizer = DataVisualizer(config)
        
        data = visualizer.load_data()
        if data is not None:
            print("✓ 数据加载成功")
            
            # 检查数据格式
            if 'reynolds_data' in data:
                print("  - 检测到新数据格式")
                print(f"  - 雷诺数: {data['reynolds_keys']}")
                print(f"  - 归一化参数: {list(data['normalization_params'].keys())}")
                
                # 显示每个雷诺数的数据形状
                for reynolds in data['reynolds_keys']:
                    reynolds_data = data['reynolds_data'][reynolds]
                    print(f"  - 雷诺数 {reynolds}: pressure {reynolds_data['pressure'].shape}, in_pressure {reynolds_data['in_pressure'].shape}")
            else:
                print("  - 检测到旧数据格式")
                print(f"  - 压力数据形状: {data['pressure'].shape}")
                print(f"  - 输入压力数据形状: {data['in_pressure'].shape}")
                print(f"  - 雷诺数: {data['reynolds_keys']}")
            
            return True
        else:
            print("✗ 数据加载失败")
            return False
            
    except Exception as e:
        print(f"✗ 数据加载过程中发生错误: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始测试CFD数据处理脚本...")
    
    test_results = []
    
    # 测试1: 创建测试数据
    test_results.append(test_create_test_data())
    
    # 测试2: 数据处理
    test_results.append(test_data_processing())
    
    # 测试3: 数据加载
    test_results.append(test_data_loading())
    
    # 总结测试结果
    print("\n=== 测试总结 ===")
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有测试通过！脚本功能正常。")
        return 0
    else:
        print("✗ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())