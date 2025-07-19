"""损失函数单元测试

测试各种损失函数的功能和正确性
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

try:
    from training.loss_functions import (
        MultiLossFunction,
        SVDLoss,
        SparsityLoss,
        ReconstructionLoss
    )
except ImportError:
    # 如果导入失败，跳过这些测试
    pytest.skip("Loss function modules not available", allow_module_level=True)


class TestMultiLossFunction:
    """多损失函数测试"""
    
    @pytest.fixture
    def multi_loss(self, device):
        """创建多损失函数"""
        loss_config = {
            'mse_weight': 1.0,
            'svd_weight': 0.5,
            'sparsity_weight': 0.1
        }
        return MultiLossFunction(loss_config).to(device)
    
    def test_multi_loss_forward(self, multi_loss, loss_test_data):
        """测试多损失函数前向传播"""
        predictions, targets = loss_test_data
        
        loss_dict = multi_loss(predictions, targets)
        
        # 检查返回的损失字典
        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        assert 'mse_loss' in loss_dict
        
        # 检查损失值是否为标量
        for loss_name, loss_value in loss_dict.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.dim() == 0  # 标量
            assert loss_value.item() >= 0  # 非负
    
    def test_multi_loss_gradients(self, multi_loss, loss_test_data):
        """测试多损失函数梯度计算"""
        predictions, targets = loss_test_data
        predictions.requires_grad_(True)
        
        loss_dict = multi_loss(predictions, targets)
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        # 检查梯度是否存在且不为零
        assert predictions.grad is not None
        assert not torch.allclose(predictions.grad, torch.zeros_like(predictions.grad))
    
    @pytest.mark.parametrize("batch_size,output_dim", [
        (1, 5),
        (4, 10),
        (8, 20),
    ])
    def test_multi_loss_different_sizes(self, device, batch_size, output_dim):
        """测试不同尺寸的损失计算"""
        loss_config = {'mse_weight': 1.0}
        multi_loss = MultiLossFunction(loss_config).to(device)
        
        predictions = torch.randn(batch_size, output_dim).to(device)
        targets = torch.randn(batch_size, output_dim).to(device)
        
        loss_dict = multi_loss(predictions, targets)
        
        assert 'total_loss' in loss_dict
        assert loss_dict['total_loss'].item() >= 0


class TestSVDLoss:
    """SVD损失函数测试"""
    
    @pytest.fixture
    def svd_loss(self, device):
        """创建SVD损失函数"""
        return SVDLoss().to(device)
    
    def test_svd_loss_forward(self, svd_loss, device):
        """测试SVD损失前向传播"""
        batch_size, seq_len, d_model = 2, 8, 16
        
        predictions = torch.randn(batch_size, seq_len, d_model).to(device)
        targets = torch.randn(batch_size, seq_len, d_model).to(device)
        
        loss = svd_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_svd_loss_identical_inputs(self, svd_loss, device):
        """测试相同输入的SVD损失"""
        batch_size, seq_len, d_model = 2, 8, 16
        
        data = torch.randn(batch_size, seq_len, d_model).to(device)
        loss = svd_loss(data, data)
        
        # 相同输入的损失应该很小（接近零）
        assert loss.item() < 1e-5
    
    def test_svd_loss_gradients(self, svd_loss, device):
        """测试SVD损失梯度计算"""
        batch_size, seq_len, d_model = 2, 8, 16
        
        predictions = torch.randn(batch_size, seq_len, d_model, requires_grad=True).to(device)
        targets = torch.randn(batch_size, seq_len, d_model).to(device)
        
        loss = svd_loss(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None
        assert not torch.allclose(predictions.grad, torch.zeros_like(predictions.grad))


class TestSparsityLoss:
    """稀疏性损失函数测试"""
    
    @pytest.fixture
    def sparsity_loss(self, device):
        """创建稀疏性损失函数"""
        return SparsityLoss().to(device)
    
    def test_sparsity_loss_forward(self, sparsity_loss, device):
        """测试稀疏性损失前向传播"""
        batch_size, seq_len, d_model = 2, 8, 16
        
        data = torch.randn(batch_size, seq_len, d_model).to(device)
        loss = sparsity_loss(data)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_sparsity_loss_sparse_input(self, sparsity_loss, device):
        """测试稀疏输入的稀疏性损失"""
        batch_size, seq_len, d_model = 2, 8, 16
        
        # 创建稀疏数据（大部分为零）
        sparse_data = torch.zeros(batch_size, seq_len, d_model).to(device)
        sparse_data[:, :2, :4] = torch.randn(batch_size, 2, 4).to(device)
        
        # 创建密集数据
        dense_data = torch.randn(batch_size, seq_len, d_model).to(device)
        
        sparse_loss = sparsity_loss(sparse_data)
        dense_loss = sparsity_loss(dense_data)
        
        # 稀疏数据的稀疏性损失应该更小
        assert sparse_loss.item() < dense_loss.item()
    
    def test_sparsity_loss_zero_input(self, sparsity_loss, device):
        """测试零输入的稀疏性损失"""
        batch_size, seq_len, d_model = 2, 8, 16
        
        zero_data = torch.zeros(batch_size, seq_len, d_model).to(device)
        loss = sparsity_loss(zero_data)
        
        # 全零输入的稀疏性损失应该为零
        assert loss.item() < 1e-6


class TestReconstructionLoss:
    """重构损失函数测试"""
    
    @pytest.fixture
    def reconstruction_loss(self, device):
        """创建重构损失函数"""
        return ReconstructionLoss().to(device)
    
    def test_reconstruction_loss_forward(self, reconstruction_loss, loss_test_data):
        """测试重构损失前向传播"""
        predictions, targets = loss_test_data
        
        loss = reconstruction_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_reconstruction_loss_identical_inputs(self, reconstruction_loss, device):
        """测试相同输入的重构损失"""
        batch_size, output_dim = 4, 10
        
        data = torch.randn(batch_size, output_dim).to(device)
        loss = reconstruction_loss(data, data)
        
        # 相同输入的重构损失应该为零
        assert loss.item() < 1e-6
    
    def test_reconstruction_loss_mse_equivalence(self, reconstruction_loss, loss_test_data):
        """测试重构损失与MSE损失的等价性"""
        predictions, targets = loss_test_data
        
        reconstruction_loss_value = reconstruction_loss(predictions, targets)
        mse_loss = nn.MSELoss()
        mse_loss_value = mse_loss(predictions, targets)
        
        # 重构损失应该等于MSE损失（或者是其变体）
        assert torch.allclose(reconstruction_loss_value, mse_loss_value, atol=1e-6)


class TestLossFunctionNumericalStability:
    """损失函数数值稳定性测试"""
    
    def test_large_values_stability(self, device):
        """测试大数值的稳定性"""
        loss_config = {'mse_weight': 1.0}
        multi_loss = MultiLossFunction(loss_config).to(device)
        
        # 创建包含大数值的输入
        predictions = torch.randn(2, 10).to(device) * 1000
        targets = torch.randn(2, 10).to(device) * 1000
        
        loss_dict = multi_loss(predictions, targets)
        
        # 检查损失值是否包含NaN或Inf
        for loss_name, loss_value in loss_dict.items():
            assert not torch.isnan(loss_value)
            assert not torch.isinf(loss_value)
    
    def test_small_values_stability(self, device):
        """测试小数值的稳定性"""
        loss_config = {'mse_weight': 1.0}
        multi_loss = MultiLossFunction(loss_config).to(device)
        
        # 创建包含小数值的输入
        predictions = torch.randn(2, 10).to(device) * 1e-6
        targets = torch.randn(2, 10).to(device) * 1e-6
        
        loss_dict = multi_loss(predictions, targets)
        
        # 检查损失值是否合理
        for loss_name, loss_value in loss_dict.items():
            assert not torch.isnan(loss_value)
            assert not torch.isinf(loss_value)
            assert loss_value.item() >= 0


@pytest.mark.performance
class TestLossFunctionPerformance:
    """损失函数性能测试"""
    
    @pytest.mark.slow
    def test_loss_computation_speed(self, device):
        """测试损失计算速度"""
        import time
        
        loss_config = {
            'mse_weight': 1.0,
            'svd_weight': 0.5,
            'sparsity_weight': 0.1
        }
        multi_loss = MultiLossFunction(loss_config).to(device)
        
        batch_size, output_dim = 32, 256
        predictions = torch.randn(batch_size, output_dim).to(device)
        targets = torch.randn(batch_size, output_dim).to(device)
        
        # 预热
        for _ in range(10):
            _ = multi_loss(predictions, targets)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        start_time = time.time()
        num_iterations = 1000
        
        for _ in range(num_iterations):
            _ = multi_loss(predictions, targets)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        # 平均时间应该在合理范围内
        assert avg_time < 0.01  # 每次损失计算不应超过10ms
    
    @pytest.mark.slow
    def test_loss_memory_usage(self, device):
        """测试损失函数内存使用"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        loss_config = {
            'mse_weight': 1.0,
            'svd_weight': 0.5,
            'sparsity_weight': 0.1
        }
        multi_loss = MultiLossFunction(loss_config).to(device)
        
        batch_size, output_dim = 64, 512
        
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        predictions = torch.randn(batch_size, output_dim).to(device)
        targets = torch.randn(batch_size, output_dim).to(device)
        
        loss_dict = multi_loss(predictions, targets)
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = peak_memory - initial_memory
        
        # 内存使用应该在合理范围内
        expected_memory = batch_size * output_dim * 4 * 2  # 大致估算
        assert memory_used < expected_memory * 5  # 允许5倍的缓冲


class TestLossFunctionIntegration:
    """损失函数集成测试"""
    
    def test_loss_backward_compatibility(self, device):
        """测试损失函数向后兼容性"""
        # 测试不同的损失配置
        configs = [
            {'mse_weight': 1.0},
            {'mse_weight': 1.0, 'svd_weight': 0.5},
            {'mse_weight': 1.0, 'svd_weight': 0.5, 'sparsity_weight': 0.1},
        ]
        
        predictions = torch.randn(2, 10).to(device)
        targets = torch.randn(2, 10).to(device)
        
        for config in configs:
            multi_loss = MultiLossFunction(config).to(device)
            loss_dict = multi_loss(predictions, targets)
            
            assert 'total_loss' in loss_dict
            assert loss_dict['total_loss'].item() >= 0
    
    def test_loss_function_serialization(self, device, temp_dir):
        """测试损失函数序列化"""
        loss_config = {
            'mse_weight': 1.0,
            'svd_weight': 0.5
        }
        multi_loss = MultiLossFunction(loss_config).to(device)
        
        # 保存模型
        model_path = temp_dir / "loss_function.pth"
        torch.save(multi_loss.state_dict(), model_path)
        
        # 加载模型
        new_multi_loss = MultiLossFunction(loss_config).to(device)
        new_multi_loss.load_state_dict(torch.load(model_path, map_location=device))
        
        # 测试加载的模型
        predictions = torch.randn(2, 10).to(device)
        targets = torch.randn(2, 10).to(device)
        
        original_loss = multi_loss(predictions, targets)
        loaded_loss = new_multi_loss(predictions, targets)
        
        # 结果应该相同
        for key in original_loss.keys():
            assert torch.allclose(original_loss[key], loaded_loss[key], atol=1e-6)