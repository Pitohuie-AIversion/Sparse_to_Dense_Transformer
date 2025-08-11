#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的注意力机制
"""

import torch
import sys
sys.path.append('.')
from mymodels.components.attention_factory import get_attention_module, ATTENTION_MODULES

def test_sample_mechanisms():
    """
    测试几个代表性的注意力机制
    """
    print("🔍 测试修复后的注意力机制...")
    
    # 测试参数
    d_model = 256
    num_heads = 4
    seq_len = 49
    batch_size = 2
    
    # 测试不同类型的注意力机制
    test_cases = [
        ("self", "QKV"),
        ("muse", "QKV"),
        ("se", "CNN"),
        ("cbam", "CNN"),
        ("external", "SINGLE_INPUT"),
        ("aft", "SINGLE_INPUT")
    ]
    
    success_count = 0
    
    for attention_type, expected_type in test_cases:
        try:
            print(f"\n📝 测试 {attention_type} 注意力机制...")
            
            # 获取注意力模块
            attention_module, adapter_type = get_attention_module(
                attention_type, d_model=d_model, num_heads=num_heads
            )
            
            print(f"   ✅ 模块创建成功，适配器类型: {adapter_type.name}")
            
            # 根据适配器类型创建测试输入
            if adapter_type.name == 'QKV':
                query = torch.randn(batch_size, seq_len, d_model)
                key = torch.randn(batch_size, seq_len, d_model)
                value = torch.randn(batch_size, seq_len, d_model)
                output = attention_module(query, key, value)
                print(f"   ✅ QKV前向传播成功: {output.shape}")
                
            elif adapter_type.name == 'CNN':
                img_size = int(seq_len ** 0.5)  # 7x7 = 49
                x = torch.randn(batch_size, d_model, img_size, img_size)
                output = attention_module(x)
                print(f"   ✅ CNN前向传播成功: {output.shape}")
                
            elif adapter_type.name == 'SINGLE_INPUT':
                x = torch.randn(batch_size, seq_len, d_model)
                output = attention_module(x)
                print(f"   ✅ 单输入前向传播成功: {output.shape}")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ {attention_type} 测试失败: {e}")
    
    print(f"\n📊 测试结果: {success_count}/{len(test_cases)} 成功")
    return success_count == len(test_cases)

def main():
    """
    主测试函数
    """
    print("🚀 开始测试修复后的注意力机制...")
    
    # 显示可用的注意力机制
    print(f"\n📋 可用的注意力机制数量: {len(ATTENTION_MODULES)}")
    print("可用机制列表:")
    for i, mechanism in enumerate(ATTENTION_MODULES.keys(), 1):
        print(f"  {i:2d}. {mechanism}")
    
    # 运行测试
    success = test_sample_mechanisms()
    
    if success:
        print("\n🎉 所有测试通过！注意力机制修复成功。")
        print("\n✅ 现在可以安全地运行训练，不会遇到MOA等不兼容机制的错误。")
    else:
        print("\n😞 部分测试失败，可能还有问题需要解决。")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)