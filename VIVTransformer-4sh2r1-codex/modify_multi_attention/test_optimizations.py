#!/usr/bin/env python3
"""
测试所有代码优化的脚本
验证以下优化是否正常工作：
1. attention_adapter.py 的导入和内存处理
2. SparseSelfAttention 的向量化掩码创建
3. 解码器输入与内存的分离
4. EmbeddingAndEncoding 的时间步索引校验
"""

import torch
import torch.nn as nn
import warnings
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mymodels.components.attention_adapter import SingleInputAttentionAdapter, AdapterType
from mymodels.components.attention import SparseSelfAttention
from mymodels.embedding import EmbeddingAndEncoding
from mymodels.embedding_2d import EmbeddingAndEncoding2D
from mymodels.transformer import TransformerFlowReconstructionModel

def test_attention_adapter_memory_warning():
    """测试SingleInputAttentionAdapter对memory参数的警告处理"""
    print("\n=== 测试 1: SingleInputAttentionAdapter 内存参数警告 ===")
    
    # 创建一个简单的注意力模块用于测试
    class SimpleAttention(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model
            
        def forward(self, x, key=None, value=None, attn_mask=None):
            # 单输入注意力，忽略key和value参数
            return x, torch.ones(x.size(0), x.size(1), x.size(1))
    
    adapter = SingleInputAttentionAdapter(SimpleAttention(64))
    
    # 测试数据
    batch_size, seq_len, d_model = 2, 10, 64
    x = torch.randn(batch_size, seq_len, d_model)
    memory = torch.randn(batch_size, seq_len, d_model)
    
    # 捕获警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        output, attn_weights = adapter(x, memory=memory)
        
        # 检查是否有警告
        if len(w) > 0:
            warning_msg = str(w[0].message)
            print(f"捕获到警告: {warning_msg}")
            if "memory" in warning_msg and "ignored" in warning_msg:
                print("✓ 成功捕获内存参数警告")
            else:
                print("✗ 警告内容不匹配")
                return False
        else:
            print("✗ 未捕获到任何警告")
            return False
    
    print(f"输出形状: {output.shape}")
    return True

def test_sparse_attention_vectorized_mask():
    """测试SparseSelfAttention的向量化掩码创建"""
    print("\n=== 测试 2: SparseSelfAttention 向量化掩码创建 ===")
    
    d_model = 64
    num_heads = 8
    block_size = 4
    
    sparse_attn = SparseSelfAttention(d_model, num_heads, block_size)
    
    # 测试数据
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 测试前向传播
    try:
        # SparseSelfAttention需要query, key, value三个参数
        output, attn_weights = sparse_attn(x, x, x)
        print(f"✓ 稀疏注意力前向传播成功")
        print(f"输出形状: {output.shape}")
        print(f"注意力权重形状: {attn_weights.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 稀疏注意力测试失败: {e}")
        return False

def test_decoder_input_separation():
    """测试解码器输入与内存的分离"""
    print("\n=== 测试 3: 解码器输入与内存分离 ===")
    
    config = {
        'input_dim': 49,
        'output_dim': 49,
        'num_heads': 8,
        'num_layers': 2,
        'd_model': 64,
        'max_time_steps': 100,
        'attention_type': 'self',
        'seq_len': 49,
        'use_2d_embedding': False
    }
    
    model = TransformerFlowReconstructionModel(**config)
    
    # 测试数据
    batch_size = 2
    x_in_pressures_flat = torch.randn(batch_size, config['input_dim'])
    x_time_steps = torch.randint(0, config['max_time_steps'], (batch_size,))
    
    try:
        output = model(x_in_pressures_flat, x_time_steps)
        print(f"✓ 模型前向传播成功")
        print(f"输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

def test_time_step_validation():
    """测试时间步索引校验"""
    print("\n=== 测试 4: 时间步索引校验 ===")
    
    # 测试原始嵌入模块
    print("测试原始嵌入模块:")
    embedding = EmbeddingAndEncoding(
        input_dim=49, d_model=64, max_time_steps=10, seq_len=49
    )
    
    batch_size = 2
    x_in_pressures_flat = torch.randn(batch_size, 49)
    
    # 测试正常时间步
    x_time_steps_normal = torch.tensor([5, 8])
    try:
        output = embedding(x_in_pressures_flat, x_time_steps_normal)
        print(f"✓ 正常时间步测试通过，输出形状: {output.shape}")
    except Exception as e:
        print(f"✗ 正常时间步测试失败: {e}")
        return False
    
    # 测试越界时间步
    x_time_steps_invalid = torch.tensor([15, -1])  # 超出范围 [0, 9]
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            output = embedding(x_in_pressures_flat, x_time_steps_invalid)
            if len(w) > 0 and "Time step indices out of range" in str(w[0].message):
                print("✓ 成功捕获时间步越界警告")
            else:
                print("✗ 未捕获到预期的时间步越界警告")
                return False
        except Exception as e:
            print(f"✗ 时间步校验测试失败: {e}")
            return False
    
    # 测试2D嵌入模块
    print("\n测试2D嵌入模块:")
    embedding_2d = EmbeddingAndEncoding2D(
        input_dim=1, d_model=64, max_time_steps=10, seq_len=49, 
        grid_height=7, grid_width=7
    )
    
    x_in_pressures_flat_2d = torch.randn(batch_size, 49)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            output = embedding_2d(x_in_pressures_flat_2d, x_time_steps_invalid)
            if len(w) > 0 and "Time step indices out of range" in str(w[0].message):
                print("✓ 2D嵌入模块成功捕获时间步越界警告")
            else:
                print("✗ 2D嵌入模块未捕获到预期的时间步越界警告")
                return False
        except Exception as e:
            print(f"✗ 2D嵌入模块时间步校验测试失败: {e}")
            return False
    
    return True

def main():
    """运行所有测试"""
    print("开始运行代码优化测试...")
    
    tests = [
        test_attention_adapter_memory_warning,
        test_sparse_attention_vectorized_mask,
        test_decoder_input_separation,
        test_time_step_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试 {test.__name__} 出现异常: {e}")
    
    print(f"\n=== 测试总结 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有优化测试通过！")
        return True
    else:
        print("❌ 部分测试失败，请检查代码")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)