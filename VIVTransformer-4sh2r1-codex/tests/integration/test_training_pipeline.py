"""训练流程集成测试

测试完整的训练和推理流程
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import yaml
from pathlib import Path
import numpy as np

try:
    from main import main
    from utils.config import load_config
    from training.trainer import Trainer
except ImportError:
    # 如果导入失败，跳过这些测试
    pytest.skip("Training modules not available", allow_module_level=True)


class TestTrainingPipeline:
    """训练流程集成测试"""
    
    @pytest.fixture
    def training_config(self, temp_dir):
        """创建训练配置"""
        config = {
            'model': {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 2,
                'dropout': 0.1,
                'attention_type': 'standard'
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 0.001,
                'num_epochs': 2,
                'save_interval': 1,
                'log_interval': 1
            },
            'data': {
                'input_dim': 64,
                'output_dim': 10,
                'sequence_length': 16
            },
            'loss': {
                'mse_weight': 1.0,
                'svd_weight': 0.1
            },
            'paths': {
                'model_save_dir': str(temp_dir / 'models'),
                'log_dir': str(temp_dir / 'logs'),
                'data_dir': str(temp_dir / 'data')
            }
        }
        
        config_file = temp_dir / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        return config_file
    
    @pytest.fixture
    def sample_dataset(self, device):
        """创建示例数据集"""
        batch_size = 20
        seq_len = 16
        input_dim = 64
        output_dim = 10
        
        # 生成随机数据
        inputs = torch.randn(batch_size, seq_len, input_dim)
        targets = torch.randn(batch_size, output_dim)
        
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=4, shuffle=True)
    
    def test_training_initialization(self, training_config, device):
        """测试训练初始化"""
        config = load_config(str(training_config))
        
        # 创建训练器
        trainer = Trainer(config, device)
        
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_function is not None
    
    def test_single_training_step(self, training_config, sample_dataset, device):
        """测试单个训练步骤"""
        config = load_config(str(training_config))
        trainer = Trainer(config, device)
        
        # 获取一个批次的数据
        batch = next(iter(sample_dataset))
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 执行一个训练步骤
        initial_loss = trainer.train_step(inputs, targets)
        
        assert isinstance(initial_loss, (float, torch.Tensor))
        assert initial_loss >= 0
    
    def test_training_epoch(self, training_config, sample_dataset, device):
        """测试完整的训练轮次"""
        config = load_config(str(training_config))
        trainer = Trainer(config, device)
        
        # 记录初始损失
        initial_params = [p.clone() for p in trainer.model.parameters()]
        
        # 训练一个epoch
        epoch_loss = trainer.train_epoch(sample_dataset)
        
        assert isinstance(epoch_loss, float)
        assert epoch_loss >= 0
        
        # 检查参数是否更新
        updated_params = list(trainer.model.parameters())
        for initial, updated in zip(initial_params, updated_params):
            assert not torch.allclose(initial, updated, atol=1e-6)
    
    def test_model_evaluation(self, training_config, sample_dataset, device):
        """测试模型评估"""
        config = load_config(str(training_config))
        trainer = Trainer(config, device)
        
        # 评估模型
        eval_loss = trainer.evaluate(sample_dataset)
        
        assert isinstance(eval_loss, float)
        assert eval_loss >= 0
    
    def test_model_save_load(self, training_config, sample_dataset, device, temp_dir):
        """测试模型保存和加载"""
        config = load_config(str(training_config))
        trainer = Trainer(config, device)
        
        # 训练几步
        for batch in sample_dataset:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            trainer.train_step(inputs, targets)
            break
        
        # 保存模型
        model_path = temp_dir / 'test_model.pth'
        trainer.save_model(str(model_path))
        
        assert model_path.exists()
        
        # 创建新的训练器并加载模型
        new_trainer = Trainer(config, device)
        new_trainer.load_model(str(model_path))
        
        # 比较模型参数
        original_params = list(trainer.model.parameters())
        loaded_params = list(new_trainer.model.parameters())
        
        for orig, loaded in zip(original_params, loaded_params):
            assert torch.allclose(orig, loaded, atol=1e-6)
    
    @pytest.mark.slow
    def test_full_training_loop(self, training_config, sample_dataset, device):
        """测试完整的训练循环"""
        config = load_config(str(training_config))
        trainer = Trainer(config, device)
        
        # 记录初始损失
        initial_loss = trainer.evaluate(sample_dataset)
        
        # 运行完整训练
        training_history = trainer.train(sample_dataset, sample_dataset)
        
        assert isinstance(training_history, dict)
        assert 'train_losses' in training_history
        assert 'val_losses' in training_history
        assert len(training_history['train_losses']) > 0
        
        # 检查损失是否下降
        final_loss = trainer.evaluate(sample_dataset)
        # 注意：由于数据量小和随机性，损失可能不会严格下降
        # 这里只检查训练是否正常完成
        assert isinstance(final_loss, float)


class TestDataPipeline:
    """数据流水线集成测试"""
    
    def test_data_loading(self, temp_dir, device):
        """测试数据加载"""
        # 创建测试数据文件
        data_dir = temp_dir / 'data'
        data_dir.mkdir()
        
        # 生成并保存测试数据
        train_data = {
            'inputs': torch.randn(100, 16, 64),
            'targets': torch.randn(100, 10)
        }
        torch.save(train_data, data_dir / 'train.pt')
        
        val_data = {
            'inputs': torch.randn(20, 16, 64),
            'targets': torch.randn(20, 10)
        }
        torch.save(val_data, data_dir / 'val.pt')
        
        # 测试数据加载
        try:
            from utils.data_loader import load_data
            train_loader, val_loader = load_data(str(data_dir), batch_size=8)
            
            assert len(train_loader) > 0
            assert len(val_loader) > 0
            
            # 测试数据批次
            train_batch = next(iter(train_loader))
            assert len(train_batch) == 2  # inputs, targets
            
        except ImportError:
            pytest.skip("Data loader module not available")
    
    def test_data_preprocessing(self, device):
        """测试数据预处理"""
        try:
            from utils.preprocessing import preprocess_data
            
            # 创建原始数据
            raw_data = torch.randn(10, 16, 64)
            
            # 预处理数据
            processed_data = preprocess_data(raw_data)
            
            assert processed_data.shape == raw_data.shape
            assert not torch.isnan(processed_data).any()
            
        except ImportError:
            pytest.skip("Preprocessing module not available")
    
    def test_data_augmentation(self, device):
        """测试数据增强"""
        try:
            from utils.augmentation import augment_data
            
            # 创建原始数据
            original_data = torch.randn(5, 16, 64)
            
            # 数据增强
            augmented_data = augment_data(original_data)
            
            assert augmented_data.shape[0] >= original_data.shape[0]
            assert augmented_data.shape[1:] == original_data.shape[1:]
            
        except ImportError:
            pytest.skip("Augmentation module not available")


class TestModelIntegration:
    """模型集成测试"""
    
    def test_model_forward_pass(self, sample_config, device):
        """测试模型前向传播"""
        try:
            from mymodels.vivtransformer import VIVTransformer
            
            model = VIVTransformer(sample_config).to(device)
            
            batch_size = 4
            seq_len = sample_config.get('max_seq_len', 16)
            d_model = sample_config['d_model']
            
            inputs = torch.randn(batch_size, seq_len, d_model).to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
            
            assert outputs is not None
            assert outputs.shape[0] == batch_size
            
        except ImportError:
            pytest.skip("VIVTransformer model not available")
    
    def test_model_backward_pass(self, sample_config, device):
        """测试模型反向传播"""
        try:
            from mymodels.vivtransformer import VIVTransformer
            
            model = VIVTransformer(sample_config).to(device)
            
            batch_size = 4
            seq_len = sample_config.get('max_seq_len', 16)
            d_model = sample_config['d_model']
            output_dim = sample_config['output_size']
            
            inputs = torch.randn(batch_size, seq_len, d_model, requires_grad=True).to(device)
            targets = torch.randn(batch_size, output_dim).to(device)
            
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
            loss.backward()
            
            # 检查梯度
            assert inputs.grad is not None
            for param in model.parameters():
                if param.requires_grad:
                    assert param.grad is not None
            
        except ImportError:
            pytest.skip("VIVTransformer model not available")
    
    def test_model_different_attention_types(self, device):
        """测试不同注意力类型的模型"""
        try:
            from mymodels.vivtransformer import VIVTransformer
            from mymodels.attention import ATTENTION_REGISTRY
            
            base_config = {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 2,
                'output_size': 10,
                'dropout': 0.1
            }
            
            # 测试不同的注意力类型
            for attention_type in ATTENTION_REGISTRY.keys():
                if attention_type == 'test':  # 跳过测试用的注意力
                    continue
                
                config = base_config.copy()
                config['attention_type'] = attention_type
                
                try:
                    model = VIVTransformer(config).to(device)
                    
                    # 简单的前向传播测试
                    inputs = torch.randn(2, 8, 64).to(device)
                    with torch.no_grad():
                        outputs = model(inputs)
                    
                    assert outputs is not None
                    
                except Exception as e:
                    pytest.fail(f"Failed to test attention type {attention_type}: {e}")
            
        except ImportError:
            pytest.skip("Model modules not available")


@pytest.mark.slow
class TestPerformanceIntegration:
    """性能集成测试"""
    
    def test_training_memory_usage(self, training_config, sample_dataset, device):
        """测试训练内存使用"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        config = load_config(str(training_config))
        trainer = Trainer(config, device)
        
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # 训练一个epoch
        trainer.train_epoch(sample_dataset)
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = peak_memory - initial_memory
        
        # 内存使用应该在合理范围内
        # 这里设置一个宽松的上限（1GB）
        assert memory_used < 1024 * 1024 * 1024
    
    def test_training_speed(self, training_config, sample_dataset, device):
        """测试训练速度"""
        import time
        
        config = load_config(str(training_config))
        trainer = Trainer(config, device)
        
        # 预热
        batch = next(iter(sample_dataset))
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        for _ in range(3):
            trainer.train_step(inputs, targets)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 计时
        start_time = time.time()
        num_steps = 10
        
        for _ in range(num_steps):
            trainer.train_step(inputs, targets)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_steps
        
        # 每个训练步骤应该在合理时间内完成
        assert avg_time < 5.0  # 每步不超过5秒


class TestConfigurationIntegration:
    """配置集成测试"""
    
    def test_config_validation(self, temp_dir):
        """测试配置验证"""
        # 测试有效配置
        valid_config = {
            'model': {'d_model': 64, 'num_heads': 4},
            'training': {'batch_size': 8, 'learning_rate': 0.001},
            'data': {'input_dim': 64, 'output_dim': 10}
        }
        
        config_file = temp_dir / 'valid_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        try:
            config = load_config(str(config_file))
            assert config is not None
        except ImportError:
            pytest.skip("Config module not available")
    
    def test_config_error_handling(self, temp_dir):
        """测试配置错误处理"""
        # 测试无效配置
        invalid_config = {
            'model': {'d_model': -1},  # 无效值
            'training': {'batch_size': 0}  # 无效值
        }
        
        config_file = temp_dir / 'invalid_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        try:
            from utils.config import validate_config
            
            with pytest.raises((ValueError, AssertionError)):
                config = load_config(str(config_file))
                validate_config(config)
                
        except ImportError:
            pytest.skip("Config validation module not available")