#!/usr/bin/env python3
"""PDE比较研究脚本

本脚本比较不同PDE类型（Darcy Flow vs Compressible Navier-Stokes）
和不同注意力机制的性能。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging
import time
from typing import Dict, Any, List, Tuple
import pandas as pd

from data.pdebench_adapter import create_pdebench_loaders
from mymodels.model_factory import create_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDEPerformanceEvaluator:
    """PDE性能评估器"""
    
    def __init__(self):
        self.results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_path = Path(__file__).parent.parent / "configs" / f"{config_name}.yaml"
        
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def check_data_availability(self, data_path: str) -> bool:
        """检查数据文件是否存在"""
        data_file = Path(data_path)
        return data_file.exists()
    
    def create_data_loaders(self, config: Dict[str, Any], pde_type: str):
        """创建数据加载器"""
        data_path = config['data']['path']
        
        if not self.check_data_availability(data_path):
            logger.warning(f"数据文件不存在: {data_path}")
            return None
        
        try:
            pde_config = config['pdebench']['pde_configs'][pde_type]
            
            train_loader, val_loader, test_loader = create_pdebench_loaders(
                data_path=data_path,
                pde_type=pde_type,
                batch_size=config['data']['batch_size'],
                sequence_length=pde_config['sequence_length'],
                spatial_resolution=pde_config['spatial_resolution'],
                normalize=config['data']['normalize'],
                num_workers=config['data']['num_workers'],
                pin_memory=config['data']['pin_memory']
            )
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.error(f"创建数据加载器失败: {e}")
            return None
    
    def create_model_with_attention(self, config: Dict[str, Any], attention_type: str):
        """创建指定注意力机制的模型"""
        model_config = config['model'].copy()
        model_config['attention_type'] = attention_type
        
        try:
            model = create_model(
                attention_type=attention_type,
                input_dim=model_config['input_dim'],
                output_dim=model_config['output_dim'],
                d_model=model_config['d_model'],
                num_heads=model_config['num_heads'],
                num_layers=model_config['num_layers'],
                dropout=model_config.get('dropout', 0.1),
                device=self.device
            )
            
            model = model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"创建模型失败 ({attention_type}): {e}")
            return None
    
    def evaluate_model_performance(self, model, data_loader, max_batches: int = 10) -> Dict[str, float]:
        """评估模型性能"""
        model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        inference_times = []
        
        with torch.no_grad():
            for i, (inputs, targets, _) in enumerate(data_loader):
                if i >= max_batches:
                    break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                outputs = model(inputs)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 计算损失
                mse_loss = torch.nn.functional.mse_loss(outputs, targets)
                mae_loss = torch.nn.functional.l1_loss(outputs, targets)
                
                batch_size = inputs.size(0)
                total_loss += mse_loss.item() * batch_size
                total_mae += mae_loss.item() * batch_size
                total_samples += batch_size
        
        avg_mse = total_loss / total_samples if total_samples > 0 else float('inf')
        avg_mae = total_mae / total_samples if total_samples > 0 else float('inf')
        avg_inference_time = np.mean(inference_times) if inference_times else float('inf')
        
        return {
            'mse_loss': avg_mse,
            'mae_loss': avg_mae,
            'inference_time': avg_inference_time,
            'samples_evaluated': total_samples
        }
    
    def run_comparison_study(self, pde_configs: List[str], attention_types: List[str]):
        """运行比较研究"""
        logger.info("开始PDE比较研究...")
        
        for pde_config_name in pde_configs:
            logger.info(f"\n=== 测试PDE配置: {pde_config_name} ===")
            
            # 加载配置
            config = self.load_config(pde_config_name)
            if config is None:
                continue
            
            pde_type = config['current_pde']
            
            # 创建数据加载器
            data_loaders = self.create_data_loaders(config, pde_type)
            if data_loaders is None:
                continue
            
            train_loader, val_loader, test_loader = data_loaders
            
            # 获取数据信息
            data_info = train_loader.get_data_info()
            logger.info(f"数据集信息: {data_info['pde_type']}, 样本数: {data_info['n_samples']}")
            
            for attention_type in attention_types:
                logger.info(f"\n--- 测试注意力机制: {attention_type} ---")
                
                # 创建模型
                model = self.create_model_with_attention(config, attention_type)
                if model is None:
                    continue
                
                # 计算模型参数
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # 评估性能
                train_perf = self.evaluate_model_performance(model, train_loader, max_batches=5)
                val_perf = self.evaluate_model_performance(model, val_loader, max_batches=3)
                
                # 记录结果
                result = {
                    'pde_type': pde_type,
                    'pde_config': pde_config_name,
                    'attention_type': attention_type,
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'input_dim': data_info['input_dim'],
                    'spatial_resolution': data_info['spatial_resolution'],
                    'sequence_length': config['pdebench']['pde_configs'][pde_type]['sequence_length'],
                    'train_mse': train_perf['mse_loss'],
                    'train_mae': train_perf['mae_loss'],
                    'train_inference_time': train_perf['inference_time'],
                    'val_mse': val_perf['mse_loss'],
                    'val_mae': val_perf['mae_loss'],
                    'val_inference_time': val_perf['inference_time']
                }
                
                self.results.append(result)
                
                logger.info(f"模型参数: {total_params:,}")
                logger.info(f"训练MSE: {train_perf['mse_loss']:.6f}")
                logger.info(f"验证MSE: {val_perf['mse_loss']:.6f}")
                logger.info(f"推理时间: {train_perf['inference_time']:.4f}s")
                
                # 清理GPU内存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def save_results(self, output_path: str = "pde_comparison_results.csv"):
        """保存结果到CSV文件"""
        if not self.results:
            logger.warning("没有结果可保存")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        logger.info(f"结果已保存到: {output_path}")
        
        return df
    
    def create_comparison_plots(self, df: pd.DataFrame):
        """创建比较图表"""
        if df is None or df.empty:
            logger.warning("没有数据可绘图")
            return
        
        # 设置图表样式
        plt.style.use('default')
        
        # 1. MSE损失比较
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 训练MSE
        for pde_type in df['pde_type'].unique():
            pde_data = df[df['pde_type'] == pde_type]
            axes[0, 0].bar(pde_data['attention_type'], pde_data['train_mse'], 
                          alpha=0.7, label=pde_type)
        axes[0, 0].set_title('训练MSE损失比较')
        axes[0, 0].set_ylabel('MSE损失')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 验证MSE
        for pde_type in df['pde_type'].unique():
            pde_data = df[df['pde_type'] == pde_type]
            axes[0, 1].bar(pde_data['attention_type'], pde_data['val_mse'], 
                          alpha=0.7, label=pde_type)
        axes[0, 1].set_title('验证MSE损失比较')
        axes[0, 1].set_ylabel('MSE损失')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 推理时间
        for pde_type in df['pde_type'].unique():
            pde_data = df[df['pde_type'] == pde_type]
            axes[1, 0].bar(pde_data['attention_type'], pde_data['train_inference_time'], 
                          alpha=0.7, label=pde_type)
        axes[1, 0].set_title('推理时间比较')
        axes[1, 0].set_ylabel('时间 (秒)')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 模型参数数量
        for pde_type in df['pde_type'].unique():
            pde_data = df[df['pde_type'] == pde_type]
            axes[1, 1].bar(pde_data['attention_type'], pde_data['total_params'], 
                          alpha=0.7, label=pde_type)
        axes[1, 1].set_title('模型参数数量比较')
        axes[1, 1].set_ylabel('参数数量')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('pde_comparison_plots.png', dpi=300, bbox_inches='tight')
        logger.info("比较图表已保存: pde_comparison_plots.png")
        plt.show()
        
        # 2. 性能效率散点图
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        for pde_type in df['pde_type'].unique():
            pde_data = df[df['pde_type'] == pde_type]
            scatter = ax.scatter(pde_data['val_mse'], pde_data['train_inference_time'], 
                               s=pde_data['total_params']/1000, alpha=0.7, label=pde_type)
            
            # 添加注意力类型标签
            for i, row in pde_data.iterrows():
                ax.annotate(row['attention_type'], 
                           (row['val_mse'], row['train_inference_time']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('验证MSE损失')
        ax.set_ylabel('推理时间 (秒)')
        ax.set_title('性能效率分析\n(气泡大小表示模型参数数量)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("性能效率分析图已保存: performance_efficiency_analysis.png")
        plt.show()
    
    def print_summary(self, df: pd.DataFrame):
        """打印结果摘要"""
        if df is None or df.empty:
            logger.warning("没有结果可总结")
            return
        
        logger.info("\n=== 比较研究结果摘要 ===")
        
        for pde_type in df['pde_type'].unique():
            logger.info(f"\n{pde_type.upper()} 数据集:")
            pde_data = df[df['pde_type'] == pde_type]
            
            # 最佳性能
            best_mse = pde_data.loc[pde_data['val_mse'].idxmin()]
            best_speed = pde_data.loc[pde_data['train_inference_time'].idxmin()]
            
            logger.info(f"  最佳MSE: {best_mse['attention_type']} ({best_mse['val_mse']:.6f})")
            logger.info(f"  最快推理: {best_speed['attention_type']} ({best_speed['train_inference_time']:.4f}s)")
            
            # 参数效率
            pde_data['param_efficiency'] = 1 / (pde_data['val_mse'] * pde_data['total_params'])
            best_efficiency = pde_data.loc[pde_data['param_efficiency'].idxmax()]
            logger.info(f"  最佳参数效率: {best_efficiency['attention_type']}")


def main():
    """主函数"""
    logger.info("=== PDE比较研究脚本 ===")
    
    # 初始化评估器
    evaluator = PDEPerformanceEvaluator()
    
    # 定义要比较的配置和注意力机制
    pde_configs = ['darcy_flow_config', 'ns_compressible_config']
    attention_types = ['MultiHeadAttention', 'ECAAttention', 'SEAttention', 'CBAM', 'CoordinateAttention']
    
    # 运行比较研究
    evaluator.run_comparison_study(pde_configs, attention_types)
    
    # 保存和分析结果
    df = evaluator.save_results()
    
    if df is not None:
        evaluator.create_comparison_plots(df)
        evaluator.print_summary(df)
    
    logger.info("=== PDE比较研究完成 ===")


if __name__ == "__main__":
    main()