#!/usr/bin/env python3
"""
测试SGE注意力错误修复

这个脚本用于验证修复后的配置文件是否能正常工作
"""

import torch
import yaml
import sys
from pathlib import Path

# 添加项目路径
sys.path.append('.')
sys.path.append('./mymodels')

def test_config_loading(config_path):
    """
    测试配置文件加载
    """
    print(f"\n=== 测试配置文件: {config_path} ===")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✓ 配置文件加载成功")
        
        # 检查关键参数
        model_config = config['model']
        print(f"  - attention_type: {model_config['attention_type']}")
        print(f"  - seq_len: {model_config['seq_len']}")
        print(f"  - input_dim: {model_config['input_dim']}")
        print(f"  - output_dim: {model_config['output_dim']}")
        
        # 检查seq_len是否为完全平方数（仅对SGE注意力）
        if model_config['attention_type'] == 'sge':
            seq_len = model_config['seq_len']
            sqrt_seq_len = int(seq_len ** 0.5)
            if sqrt_seq_len * sqrt_seq_len == seq_len:
                print(f"  ✓ seq_len {seq_len} 是完全平方数 ({sqrt_seq_len}x{sqrt_seq_len})")
            else:
                print(f"  ✗ seq_len {seq_len} 不是完全平方数")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

def test_attention_module_creation(config_path):
    """
    测试注意力模块创建
    """
    print(f"\n=== 测试注意力模块创建 ===")
    
    try:
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        model_config = config['model']
        attention_type = model_config['attention_type']
        seq_len = model_config['seq_len']
        hidden_dim = model_config['hidden_dim']
        
        # 导入必要的模块
        from mymodels.components.attention_factory import get_attention_module
        from mymodels.components.attention_adapter import get_attention_adapter
        
        # 创建注意力模块
        attention_module, adapter_type = get_attention_module(attention_type, d_model=hidden_dim, spatial_dim=int(seq_len**0.5))
        print(f"✓ 注意力模块创建成功: {attention_type} -> {adapter_type}")
        
        # 创建注意力适配器
        attention_adapter = get_attention_adapter(attention_module, adapter_type)
        print(f"✓ 注意力适配器创建成功")
        
        # 测试前向传播
        batch_size = 2
        test_input = torch.randn(batch_size, seq_len, hidden_dim)
        
        with torch.no_grad():
            output = attention_adapter(test_input)
            print(f"✓ 前向传播测试成功")
            print(f"  - 输入形状: {test_input.shape}")
            print(f"  - 输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 注意力模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主测试函数
    """
    print("SGE注意力错误修复验证")
    print("=" * 50)
    
    # 测试配置文件
    configs_to_test = [
        "multiscale/improved_multiscale_config_fixed_seqlen.yaml",
        "multiscale/improved_multiscale_config_fixed_attention.yaml"
    ]
    
    results = []
    
    for config_path in configs_to_test:
        if Path(config_path).exists():
            # 测试配置加载
            config_ok = test_config_loading(config_path)
            
            # 测试注意力模块创建
            if config_ok:
                module_ok = test_attention_module_creation(config_path)
                results.append((config_path, config_ok and module_ok))
            else:
                results.append((config_path, False))
        else:
            print(f"\n✗ 配置文件不存在: {config_path}")
            results.append((config_path, False))
    
    # 总结结果
    print("\n" + "=" * 50)
    print("测试结果总结:")
    
    for config_path, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {config_path}: {status}")
    
    # 推荐使用方案
    print("\n推荐使用方案:")
    
    successful_configs = [config for config, success in results if success]
    
    if successful_configs:
        print(f"\n1. 优先推荐: {successful_configs[0]}")
        print(f"   python train_configurable_multiscale.py --config_file {successful_configs[0]}")
        
        if len(successful_configs) > 1:
            print(f"\n2. 备选方案: {successful_configs[1]}")
            print(f"   python train_configurable_multiscale.py --config_file {successful_configs[1]}")
    else:
        print("\n✗ 所有配置文件测试都失败了，请检查修复脚本")

if __name__ == "__main__":
    main()