#!/usr/bin/env python3
"""PDEBench集成测试

测试PDEBench数据集的集成功能，包括数据加载、模型创建等。
"""

import sys
import unittest
import tempfile
import numpy as np
import torch
import h5py
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from data.pdebench_adapter import PDEBenchDataset, PDEBenchDataLoader
    from data.dataloader import get_pdebench_loaders, get_adaptive_loaders
except ImportError:
    # 如果直接导入失败，尝试相对导入
    import sys
    sys.path.append(str(project_root))
    from data.pdebench_adapter import PDEBenchDataset, PDEBenchDataLoader
    from data.dataloader import get_pdebench_loaders, get_adaptive_loaders


class TestPDEBenchIntegration(unittest.TestCase):
    """PDEBench集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # 创建测试配置
        self.test_config = {
            'data': {
                'use_pdebench': True,
                'batch_size': 4,
                'num_workers': 0,  # 测试时不使用多进程
                'pin_memory': False,
                'normalize': True
            },
            'current_pde': 'ns_incom',
            'pdebench': {
                'data_root': str(self.temp_path),
                'pde_configs': {
                    'ns_incom': {
                        'data_file': 'test_ns_data.h5',
                        'spatial_resolution': [8, 8],
                        'sequence_length': 5,
                        'input_dim': 64,
                        'output_dim': 64
                    }
                }
            }
        }
        
        # 创建测试数据文件
        self._create_test_data_file()
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data_file(self):
        """创建测试用的HDF5数据文件"""
        data_file = self.temp_path / 'test_ns_data.h5'
        
        # 创建模拟的PDEBench数据
        # 形状: [样本数, 时间步, 高度, 宽度, 通道数]
        n_samples = 20
        n_timesteps = 5
        height, width = 8, 8
        n_channels = 2  # 例如u, v速度分量
        
        data = np.random.randn(n_samples, n_timesteps, height, width, n_channels).astype(np.float32)
        
        with h5py.File(data_file, 'w') as f:
            f.create_dataset('data', data=data)
            
            # 添加一些元数据
            f.attrs['description'] = 'Test Navier-Stokes data'
            f.attrs['spatial_resolution'] = [height, width]
            f.attrs['temporal_resolution'] = n_timesteps
    
    def test_pdebench_dataset_creation(self):
        """测试PDEBench数据集创建"""
        data_file = self.temp_path / 'test_ns_data.h5'
        
        dataset = PDEBenchDataset(
            data_path=str(data_file),
            pde_type='ns_incom',
            split='train',
            sequence_length=5,
            spatial_resolution=[8, 8],
            normalize=True
        )
        
        self.assertIsInstance(dataset, PDEBenchDataset)
        self.assertGreater(len(dataset), 0)
        
        # 测试数据获取
        inputs, targets, time_steps = dataset[0]
        
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        # 输入维度应该是 8*8*2 = 128 (高度*宽度*通道数)
        expected_dim = 8 * 8 * 2  # 2个通道 (u, v速度分量)
        self.assertEqual(inputs.shape[-1], expected_dim)
        self.assertEqual(targets.shape[-1], expected_dim)
    
    def test_pdebench_data_loader(self):
        """测试PDEBench数据加载器"""
        try:
            train_loader, valid_loader, test_loader = get_pdebench_loaders(
                config=self.test_config,
                pde_type='ns_incom',
                batch_size=4
            )
            
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(valid_loader)
            self.assertIsNotNone(test_loader)
            
            # 测试数据批次
            for inputs, targets, time_steps in train_loader:
                self.assertEqual(inputs.shape[0], 4)  # batch_size
                self.assertEqual(targets.shape[0], 4)
                break
                
        except FileNotFoundError:
            self.skipTest("测试数据文件未找到")
    
    def test_adaptive_data_loader_pdebench(self):
        """测试自适应数据加载器（PDEBench模式）"""
        try:
            train_loader, valid_loader, test_loader = get_adaptive_loaders(
                config=self.test_config
            )
            
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(valid_loader)
            self.assertIsNotNone(test_loader)
            
        except FileNotFoundError:
            self.skipTest("测试数据文件未找到")
    
    def test_adaptive_data_loader_original(self):
        """测试自适应数据加载器（原始模式）"""
        # 修改配置为使用原始数据加载方式
        config = self.test_config.copy()
        config['data']['use_pdebench'] = False
        config['data']['path'] = 'dummy_path.pt'
        config['data']['crop_size'] = [64, 64]
        
        # 模拟原始数据加载器
        with patch('data.dataloader.get_loaders') as mock_get_loaders:
            mock_get_loaders.return_value = (MagicMock(), MagicMock(), MagicMock())
            
            train_loader, valid_loader, test_loader = get_adaptive_loaders(config)
            
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(valid_loader)
            self.assertIsNotNone(test_loader)
            
            # 验证调用了原始的get_loaders函数
            mock_get_loaders.assert_called_once()
    
    def test_data_normalization(self):
        """测试数据归一化"""
        data_file = self.temp_path / 'test_ns_data.h5'
        
        # 测试启用归一化
        dataset_norm = PDEBenchDataset(
            data_path=str(data_file),
            pde_type='ns_incom',
            split='train',
            sequence_length=5,
            spatial_resolution=[8, 8],
            normalize=True
        )
        
        # 测试禁用归一化
        dataset_no_norm = PDEBenchDataset(
            data_path=str(data_file),
            pde_type='ns_incom',
            split='train',
            sequence_length=5,
            spatial_resolution=[8, 8],
            normalize=False
        )
        
        inputs_norm, _, _ = dataset_norm[0]
        inputs_no_norm, _, _ = dataset_no_norm[0]
        
        # 归一化后的数据应该有不同的统计特性
        self.assertNotEqual(inputs_norm.mean().item(), inputs_no_norm.mean().item())
    
    def test_data_splits(self):
        """测试数据分割"""
        data_file = self.temp_path / 'test_ns_data.h5'
        
        # 创建不同分割的数据集
        train_dataset = PDEBenchDataset(
            data_path=str(data_file),
            pde_type='ns_incom',
            split='train',
            sequence_length=5,
            spatial_resolution=[8, 8]
        )
        
        val_dataset = PDEBenchDataset(
            data_path=str(data_file),
            pde_type='ns_incom',
            split='val',
            sequence_length=5,
            spatial_resolution=[8, 8]
        )
        
        test_dataset = PDEBenchDataset(
            data_path=str(data_file),
            pde_type='ns_incom',
            split='test',
            sequence_length=5,
            spatial_resolution=[8, 8]
        )
        
        # 验证数据集大小
        total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
        self.assertGreater(total_samples, 0)
        
        # 验证训练集是最大的
        self.assertGreaterEqual(len(train_dataset), len(val_dataset))
        self.assertGreaterEqual(len(train_dataset), len(test_dataset))
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试不存在的数据文件
        with self.assertRaises(FileNotFoundError):
            PDEBenchDataset(
                data_path='nonexistent_file.h5',
                pde_type='ns_incom',
                split='train'
            )
        
        # 测试不支持的PDE类型
        config = self.test_config.copy()
        with self.assertRaises(ValueError):
            get_pdebench_loaders(config, pde_type='unsupported_pde')
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试缺少必要配置
        incomplete_config = {
            'data': {'use_pdebench': True},
            'pdebench': {}
        }
        
        with self.assertRaises((KeyError, ValueError)):
            get_pdebench_loaders(incomplete_config)


class TestPDEBenchDataLoader(unittest.TestCase):
    """PDEBench数据加载器专项测试"""
    
    def test_data_loader_initialization(self):
        """测试数据加载器初始化"""
        # 创建模拟数据集
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        
        loader = PDEBenchDataLoader(
            dataset=mock_dataset,
            batch_size=16,
            shuffle=True
        )
        
        self.assertIsInstance(loader, PDEBenchDataLoader)
    
    def test_batch_processing(self):
        """测试批次处理"""
        # 这里可以添加更详细的批次处理测试
        pass


def run_integration_tests():
    """运行集成测试"""
    print("🧪 运行PDEBench集成测试...")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestPDEBenchIntegration))
    test_suite.addTest(unittest.makeSuite(TestPDEBenchDataLoader))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！")
        return True
    else:
        print(f"\n❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
        return False


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)