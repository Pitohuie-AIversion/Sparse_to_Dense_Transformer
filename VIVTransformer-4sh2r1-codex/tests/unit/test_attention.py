"""注意力机制单元测试

测试各种注意力机制的功能和正确性
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

try:
    from mymodels.attention import (
        ATTENTION_REGISTRY,
        StandardAttention,
        MultiHeadAttention
    )
except ImportError:
    # 如果导入失败，跳过这些测试
    pytest.skip("Attention modules not available", allow_module_level=True)


class TestStandardAttention:
    """标准注意力机制测试"""
    
    @pytest.fixture
    def attention_layer(self, device):
        """创建标准注意力层"""
        d_model = 64
        attention = StandardAttention(d_model)
        return attention.to(device)
    
    def test_attention_forward(self, attention_layer, attention_test_data):
        """测试注意力前向传播"""
        query, key, value = attention_test_data
        
        with torch.no_grad():
            output, weights = attention_layer(query, key, value)
        
        # 检查输出形状
        assert output.shape == query.shape
        assert weights.shape == (query.size(0), query.size(1), key.size(1))
        
        # 检查注意力权重是否归一化
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6)
    
    def test_attention_gradients(self, attention_layer, attention_test_data):
        """测试注意力梯度计算"""
        query, key, value = attention_test_data
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        
        output, _ = attention_layer(query, key, value)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否存在且不为零
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None
        assert not torch.allclose(query.grad, torch.zeros_like(query.grad))
    
    @pytest.mark.parametrize("batch_size,seq_len,d_model", [
        (1, 4, 32),
        (2, 8, 64),
        (4, 16, 128),
    ])
    def test_attention_different_sizes(self, device, batch_size, seq_len, d_model):
        """测试不同尺寸的注意力计算"""
        attention = StandardAttention(d_model).to(device)
        
        query = torch.randn(batch_size, seq_len, d_model).to(device)
        key = torch.randn(batch_size, seq_len, d_model).to(device)
        value = torch.randn(batch_size, seq_len, d_model).to(device)
        
        with torch.no_grad():
            output, weights = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, seq_len, seq_len)


class TestMultiHeadAttention:
    """多头注意力机制测试"""
    
    @pytest.fixture
    def multihead_attention(self, device):
        """创建多头注意力层"""
        d_model = 64
        num_heads = 4
        attention = MultiHeadAttention(d_model, num_heads)
        return attention.to(device)
    
    def test_multihead_forward(self, multihead_attention, attention_test_data):
        """测试多头注意力前向传播"""
        query, key, value = attention_test_data
        
        with torch.no_grad():
            output = multihead_attention(query, key, value)
        
        # 检查输出形状
        assert output.shape == query.shape
    
    def test_multihead_attention_mask(self, multihead_attention, device):
        """测试注意力掩码功能"""
        batch_size, seq_len, d_model = 2, 8, 64
        
        query = torch.randn(batch_size, seq_len, d_model).to(device)
        key = torch.randn(batch_size, seq_len, d_model).to(device)
        value = torch.randn(batch_size, seq_len, d_model).to(device)
        
        # 创建掩码（掩盖后半部分）
        mask = torch.ones(batch_size, seq_len).to(device)
        mask[:, seq_len//2:] = 0
        
        with torch.no_grad():
            output_masked = multihead_attention(query, key, value, mask=mask)
            output_unmasked = multihead_attention(query, key, value)
        
        # 掩码应该改变输出
        assert not torch.allclose(output_masked, output_unmasked)


class TestAttentionRegistry:
    """注意力机制注册表测试"""
    
    def test_registry_not_empty(self):
        """测试注册表不为空"""
        assert len(ATTENTION_REGISTRY) > 0
    
    def test_registry_contains_standard(self):
        """测试注册表包含标准注意力"""
        assert 'standard' in ATTENTION_REGISTRY or 'StandardAttention' in ATTENTION_REGISTRY
    
    @pytest.mark.parametrize("attention_name", [
        name for name in ATTENTION_REGISTRY.keys() if name != 'test'
    ])
    def test_attention_instantiation(self, attention_name, device):
        """测试所有注册的注意力机制都能正确实例化"""
        try:
            attention_class = ATTENTION_REGISTRY[attention_name]
            # 尝试实例化（可能需要不同的参数）
            if hasattr(attention_class, '__call__'):
                # 如果是函数，尝试调用
                attention = attention_class(d_model=64)
            else:
                # 如果是类，尝试实例化
                attention = attention_class(d_model=64)
            
            attention = attention.to(device)
            assert attention is not None
        except Exception as e:
            pytest.fail(f"Failed to instantiate {attention_name}: {e}")


class TestAttentionNumericalStability:
    """注意力机制数值稳定性测试"""
    
    def test_large_values_stability(self, device):
        """测试大数值的稳定性"""
        d_model = 64
        attention = StandardAttention(d_model).to(device)
        
        # 创建包含大数值的输入
        query = torch.randn(1, 4, d_model).to(device) * 100
        key = torch.randn(1, 4, d_model).to(device) * 100
        value = torch.randn(1, 4, d_model).to(device)
        
        with torch.no_grad():
            output, weights = attention(query, key, value)
        
        # 检查输出是否包含NaN或Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()
    
    def test_zero_values_handling(self, device):
        """测试零值处理"""
        d_model = 64
        attention = StandardAttention(d_model).to(device)
        
        # 创建包含零值的输入
        query = torch.zeros(1, 4, d_model).to(device)
        key = torch.zeros(1, 4, d_model).to(device)
        value = torch.randn(1, 4, d_model).to(device)
        
        with torch.no_grad():
            output, weights = attention(query, key, value)
        
        # 检查输出是否合理
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


@pytest.mark.performance
class TestAttentionPerformance:
    """注意力机制性能测试"""
    
    @pytest.mark.slow
    def test_attention_memory_usage(self, device):
        """测试注意力机制内存使用"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        d_model = 512
        seq_len = 1024
        batch_size = 8
        
        attention = StandardAttention(d_model).to(device)
        
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        query = torch.randn(batch_size, seq_len, d_model).to(device)
        key = torch.randn(batch_size, seq_len, d_model).to(device)
        value = torch.randn(batch_size, seq_len, d_model).to(device)
        
        with torch.no_grad():
            output, weights = attention(query, key, value)
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = peak_memory - initial_memory
        
        # 内存使用应该在合理范围内（这里设置一个宽松的上限）
        expected_memory = batch_size * seq_len * seq_len * 4  # 大致估算
        assert memory_used < expected_memory * 10  # 允许10倍的缓冲
    
    @pytest.mark.slow
    def test_attention_speed(self, device):
        """测试注意力机制速度"""
        import time
        
        d_model = 256
        seq_len = 512
        batch_size = 4
        
        attention = StandardAttention(d_model).to(device)
        
        query = torch.randn(batch_size, seq_len, d_model).to(device)
        key = torch.randn(batch_size, seq_len, d_model).to(device)
        value = torch.randn(batch_size, seq_len, d_model).to(device)
        
        # 预热
        for _ in range(5):
            with torch.no_grad():
                _ = attention(query, key, value)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        start_time = time.time()
        num_iterations = 100
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = attention(query, key, value)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        # 平均时间应该在合理范围内（这里设置一个宽松的上限）
        assert avg_time < 1.0  # 每次前向传播不应超过1秒