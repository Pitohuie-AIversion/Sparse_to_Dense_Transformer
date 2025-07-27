#!/usr/bin/env python3
"""
专门的数据集构建脚本

针对多注意力机制Transformer模型的训练需求，构建PDEBench数据集
支持2D Darcy Flow和Compressible Navier-Stokes方程
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import yaml

import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.unified_adapter import create_unified_datasets
from data.pdebench_adapter import create_pdebench_datasets, create_pdebench_loaders
from mymodels.model_factory import create_model
from utils.config import load_config
from utils.logger import setup_logger
from utils.system import set_seed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetBuilder:
    """数据集构建器类"""
    
    def __init__(self, config_path: str):
        """初始化数据集构建器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.setup_environment()
        
    def setup_environment(self):
        """设置环境"""
        # 设置随机种子
        seed = self.config.get('global', {}).get('seed', 42)
        set_seed(seed)
        
        # 设置设备
        self.device = torch.device(
            self.config.get('global', {}).get('device', 'cuda')
            if torch.cuda.is_available() else 'cpu'
        )
        
        # 创建结果目录
        self.result_dir = Path(self.config.get('global', {}).get('result_dir', './results'))
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"结果目录: {self.result_dir}")
        
    def validate_data_files(self) -> bool:
        """验证数据文件是否存在
        
        Returns:
            bool: 所有数据文件是否存在
        """
        pdebench_config = self.config.get('pdebench', {})
        data_root = Path(pdebench_config.get('data_root', './data/pdebench'))
        pde_configs = pdebench_config.get('pde_configs', {})
        
        missing_files = []
        
        for pde_type, pde_config in pde_configs.items():
            data_file = data_root / pde_config['data_file']
            if not data_file.exists():
                missing_files.append(str(data_file))
                
        if missing_files:
            logger.error("以下数据文件不存在:")
            for file in missing_files:
                logger.error(f"  - {file}")
            return False
            
        logger.info("✓ 所有数据文件验证通过")
        return True
        
    def analyze_dataset_info(self, pde_type: str) -> Dict[str, Any]:
        """分析数据集信息
        
        Args:
            pde_type: PDE类型
            
        Returns:
            数据集信息字典
        """
        pdebench_config = self.config.get('pdebench', {})
        pde_configs = pdebench_config.get('pde_configs', {})
        
        if pde_type not in pde_configs:
            raise ValueError(f"不支持的PDE类型: {pde_type}")
            
        pde_config = pde_configs[pde_type]
        data_root = Path(pdebench_config.get('data_root', './data/pdebench'))
        data_file = data_root / pde_config['data_file']
        
        # 分析HDF5文件
        info = {
            'pde_type': pde_type,
            'data_file': str(data_file),
            'spatial_resolution': pde_config['spatial_resolution'],
            'input_dim': pde_config['input_dim'],
            'output_dim': pde_config['output_dim'],
            'sequence_length': pde_config['sequence_length'],
            'physics_type': pde_config['physics_type'],
            'description': pde_config['description']
        }
        
        try:
            with h5py.File(data_file, 'r') as f:
                # 获取数据集大小信息
                if pde_type == 'darcy_flow':
                    if 'tensor' in f:
                        tensor_shape = f['tensor'].shape
                        info['total_samples'] = tensor_shape[0]
                        info['data_shape'] = tensor_shape
                        logger.info(f"Darcy Flow数据形状: {tensor_shape}")
                        
                elif pde_type == 'ns_compressible':
                    if 'density' in f:
                        density_shape = f['density'].shape
                        info['total_samples'] = density_shape[0]
                        info['data_shape'] = density_shape
                        logger.info(f"NS Compressible数据形状: {density_shape}")
                        
                # 列出所有可用的数据字段
                info['available_fields'] = list(f.keys())
                logger.info(f"可用数据字段: {info['available_fields']}")
                
        except Exception as e:
            logger.warning(f"无法分析数据文件 {data_file}: {e}")
            
        return info
        
    def create_datasets(self, pde_type: str) -> Dict[str, Any]:
        """创建数据集
        
        Args:
            pde_type: PDE类型
            
        Returns:
            包含数据集和信息的字典
        """
        logger.info(f"正在创建 {pde_type} 数据集...")
        
        # 更新配置中的当前PDE类型
        self.config['current_pde'] = pde_type
        
        # 使用统一适配器创建数据集
        datasets = create_unified_datasets(self.config)
        
        # 获取数据集信息
        info = datasets['info']
        
        logger.info(f"数据集创建完成:")
        logger.info(f"  - PDE类型: {info.get('pde_type', 'unknown')}")
        logger.info(f"  - 输入维度: {info.get('input_dim')}")
        logger.info(f"  - 输出维度: {info.get('output_dim')}")
        logger.info(f"  - 训练样本: {info.get('train_samples')}")
        logger.info(f"  - 验证样本: {info.get('val_samples')}")
        logger.info(f"  - 测试样本: {info.get('test_samples')}")
        
        return datasets
        
    def test_data_loading(self, datasets: Dict[str, Any]) -> bool:
        """测试数据加载
        
        Args:
            datasets: 数据集字典
            
        Returns:
            bool: 测试是否成功
        """
        logger.info("正在测试数据加载...")
        
        try:
            # 测试训练集
            train_loader = datasets['train']
            for i, batch in enumerate(train_loader):
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get('input', batch.get('inputs'))
                    targets = batch.get('target', batch.get('targets'))
                else:
                    logger.error(f"未知的批次格式: {type(batch)}")
                    return False
                    
                logger.info(f"批次 {i+1}:")
                logger.info(f"  - 输入形状: {inputs.shape}")
                logger.info(f"  - 目标形状: {targets.shape}")
                logger.info(f"  - 输入数据类型: {inputs.dtype}")
                logger.info(f"  - 目标数据类型: {targets.dtype}")
                logger.info(f"  - 输入范围: [{inputs.min():.6f}, {inputs.max():.6f}]")
                logger.info(f"  - 目标范围: [{targets.min():.6f}, {targets.max():.6f}]")
                
                # 只测试前几个批次
                if i >= 2:
                    break
                    
            logger.info("✓ 数据加载测试通过")
            return True
            
        except Exception as e:
            logger.error(f"数据加载测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def create_model_for_testing(self, info: Dict[str, Any]) -> nn.Module:
        """创建用于测试的模型
        
        Args:
            info: 数据集信息
            
        Returns:
            模型实例
        """
        model_config = self.config.get('model', {})
        
        # 更新模型配置以匹配数据集
        model_config['input_dim'] = info.get('input_dim')
        model_config['output_dim'] = info.get('output_dim')
        
        # 创建模型
        model = create_model(model_config, self.device)
        
        logger.info(f"模型创建完成:")
        logger.info(f"  - 注意力类型: {model_config.get('attention_type', 'MultiHeadAttention')}")
        logger.info(f"  - 模型维度: {model_config.get('d_model', 256)}")
        logger.info(f"  - 注意力头数: {model_config.get('num_heads', 8)}")
        logger.info(f"  - 层数: {model_config.get('num_layers', 4)}")
        logger.info(f"  - 参数数量: {sum(p.numel() for p in model.parameters())}")
        
        return model
        
    def test_model_compatibility(self, model: nn.Module, datasets: Dict[str, Any]) -> bool:
        """测试模型兼容性
        
        Args:
            model: 模型实例
            datasets: 数据集字典
            
        Returns:
            bool: 测试是否成功
        """
        logger.info("正在测试模型兼容性...")
        
        try:
            model.eval()
            train_loader = datasets['train']
            
            with torch.no_grad():
                for batch in train_loader:
                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch[0], batch[1]
                    elif isinstance(batch, dict):
                        inputs = batch.get('input', batch.get('inputs'))
                        targets = batch.get('target', batch.get('targets'))
                    else:
                        logger.error(f"未知的批次格式: {type(batch)}")
                        return False
                        
                    # 移动到设备
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向传播
                    outputs = model(inputs)
                    
                    logger.info(f"模型测试:")
                    logger.info(f"  - 输入形状: {inputs.shape}")
                    logger.info(f"  - 输出形状: {outputs.shape}")
                    logger.info(f"  - 目标形状: {targets.shape}")
                    logger.info(f"  - 输出范围: [{outputs.min():.6f}, {outputs.max():.6f}]")
                    
                    # 检查形状匹配
                    if outputs.shape != targets.shape:
                        logger.error(f"输出形状 {outputs.shape} 与目标形状 {targets.shape} 不匹配")
                        return False
                        
                    break  # 只测试一个批次
                    
            logger.info("✓ 模型兼容性测试通过")
            return True
            
        except Exception as e:
            logger.error(f"模型兼容性测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def save_dataset_summary(self, pde_type: str, info: Dict[str, Any]):
        """保存数据集摘要
        
        Args:
            pde_type: PDE类型
            info: 数据集信息
        """
        summary_file = self.result_dir / f"{pde_type}_dataset_summary.yaml"
        
        summary = {
            'dataset_info': info,
            'model_config': self.config.get('model', {}),
            'training_config': self.config.get('training', {}),
            'data_config': self.config.get('data', {})
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
            
        logger.info(f"数据集摘要已保存到: {summary_file}")
        
    def build_dataset(self, pde_type: str) -> bool:
        """构建指定类型的数据集
        
        Args:
            pde_type: PDE类型
            
        Returns:
            bool: 构建是否成功
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始构建 {pde_type} 数据集")
        logger.info(f"{'='*60}")
        
        try:
            # 1. 分析数据集信息
            info = self.analyze_dataset_info(pde_type)
            
            # 2. 创建数据集
            datasets = self.create_datasets(pde_type)
            
            # 3. 测试数据加载
            if not self.test_data_loading(datasets):
                return False
                
            # 4. 创建测试模型
            model = self.create_model_for_testing(datasets['info'])
            
            # 5. 测试模型兼容性
            if not self.test_model_compatibility(model, datasets):
                return False
                
            # 6. 保存数据集摘要
            self.save_dataset_summary(pde_type, datasets['info'])
            
            logger.info(f"\n✓ {pde_type} 数据集构建成功!")
            return True
            
        except Exception as e:
            logger.error(f"\n✗ {pde_type} 数据集构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def build_all_datasets(self) -> Dict[str, bool]:
        """构建所有配置的数据集
        
        Returns:
            每个数据集的构建结果
        """
        pdebench_config = self.config.get('pdebench', {})
        pde_configs = pdebench_config.get('pde_configs', {})
        
        results = {}
        
        for pde_type in pde_configs.keys():
            results[pde_type] = self.build_dataset(pde_type)
            
        return results
        
    def generate_training_script(self, pde_type: str):
        """生成训练脚本
        
        Args:
            pde_type: PDE类型
        """
        script_content = f'''#!/usr/bin/env python3
"""
{pde_type} 数据集训练脚本
自动生成于数据集构建过程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    # 设置配置文件路径
    config_path = "configs/pdebench_extended_config.yaml"
    
    # 设置PDE类型
    os.environ["CURRENT_PDE"] = "{pde_type}"
    
    # 运行训练
    main(config_path)
'''
        
        script_file = self.result_dir / f"train_{pde_type}.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        # 设置执行权限
        os.chmod(script_file, 0o755)
        
        logger.info(f"训练脚本已生成: {script_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PDEBench数据集构建器")
    parser.add_argument(
        "-c", "--config",
        default="configs/pdebench_extended_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "-p", "--pde_type",
        choices=["darcy_flow", "ns_compressible", "shallow_water", "all"],
        default="all",
        help="要构建的PDE类型"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证数据文件是否存在"
    )
    
    args = parser.parse_args()
    
    # 创建数据集构建器
    builder = DatasetBuilder(args.config)
    
    # 验证数据文件
    if not builder.validate_data_files():
        logger.error("数据文件验证失败，请检查数据文件路径")
        return False
        
    if args.validate_only:
        logger.info("数据文件验证完成")
        return True
        
    # 构建数据集
    if args.pde_type == "all":
        results = builder.build_all_datasets()
        
        # 输出总结
        logger.info("\n" + "="*60)
        logger.info("数据集构建总结")
        logger.info("="*60)
        
        success_count = 0
        for pde_type, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            logger.info(f"{pde_type:20s}: {status}")
            if success:
                success_count += 1
                builder.generate_training_script(pde_type)
                
        logger.info(f"\n总计: {success_count}/{len(results)} 个数据集构建成功")
        
        return success_count == len(results)
        
    else:
        success = builder.build_dataset(args.pde_type)
        if success:
            builder.generate_training_script(args.pde_type)
        return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)