"""pytest配置文件

提供测试所需的fixtures和配置
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
import os
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "modify_multi_attention"))


@pytest.fixture(scope="session")
def device():
    """测试设备fixture"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def set_random_seed():
    """设置随机种子以确保测试可重现"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """示例配置fixture"""
    return {
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 2,
        'vocab_size': 1000,
        'output_size': 10,
        'dropout': 0.1,
        'max_seq_len': 512,
        'attention_type': 'standard',
        'loss_type': 'mse',
        'learning_rate': 0.001,
        'batch_size': 8
    }


@pytest.fixture
def sample_data(device) -> Dict[str, torch.Tensor]:
    """示例数据fixture"""
    batch_size, seq_len, d_model = 2, 10, 256
    
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'attention_mask': torch.ones(batch_size, seq_len).to(device),
        'labels': torch.randint(0, 10, (batch_size,)).to(device),
        'embeddings': torch.randn(batch_size, seq_len, d_model).to(device),
        'sparse_data': torch.randn(batch_size, seq_len, d_model).to(device),
        'dense_data': torch.randn(batch_size, seq_len, d_model).to(device)
    }


@pytest.fixture
def temp_dir():
    """临时目录fixture"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config_file(temp_dir, sample_config) -> Path:
    """示例配置文件fixture"""
    config_file = temp_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def mock_dataset(sample_data):
    """模拟数据集fixture"""
    class MockDataset:
        def __init__(self, data, size=100):
            self.data = data
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # 返回数据的副本以避免修改原始数据
            return {k: v.clone() if isinstance(v, torch.Tensor) else v 
                   for k, v in self.data.items()}
    
    return MockDataset(sample_data)


@pytest.fixture
def attention_test_data(device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """注意力机制测试数据"""
    batch_size, seq_len, d_model = 2, 8, 64
    
    query = torch.randn(batch_size, seq_len, d_model).to(device)
    key = torch.randn(batch_size, seq_len, d_model).to(device)
    value = torch.randn(batch_size, seq_len, d_model).to(device)
    
    return query, key, value


@pytest.fixture
def loss_test_data(device) -> Tuple[torch.Tensor, torch.Tensor]:
    """损失函数测试数据"""
    batch_size, output_dim = 4, 10
    
    predictions = torch.randn(batch_size, output_dim).to(device)
    targets = torch.randn(batch_size, output_dim).to(device)
    
    return predictions, targets


# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m "not slow"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests for performance benchmarking"
    )


# 跳过GPU测试的条件
skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)


# 测试环境检查
def pytest_collection_modifyitems(config, items):
    """修改测试收集，添加标记"""
    for item in items:
        # 为GPU测试添加标记
        if "gpu" in item.keywords:
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="GPU not available"))
        
        # 为慢速测试添加标记
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """自动清理CUDA缓存"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def capture_logs(caplog):
    """捕获日志输出"""
    import logging
    caplog.set_level(logging.INFO)
    return caplog