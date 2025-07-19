"""增强的训练日志记录器

提供详细的训练过程记录，包括硬件使用情况、性能指标、模型状态等。
"""

import logging
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch
import numpy as np


class EnhancedTrainingLogger:
    """增强的训练日志记录器
    
    记录训练过程中的详细信息，包括：
    - 训练进度和损失
    - 硬件使用情况
    - 模型参数统计
    - 学习率变化
    - 梯度信息
    - 数据加载时间
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "training"):
        """初始化增强日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.experiment_name = experiment_name
        
        # 创建子目录
        self.metrics_dir = self.log_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        # 初始化日志文件
        self.training_log_path = self.log_dir / f"{experiment_name}_training.log"
        self.metrics_json_path = self.metrics_dir / f"{experiment_name}_metrics.json"
        self.metrics_csv_path = self.metrics_dir / f"{experiment_name}_metrics.csv"
        
        # 数据存储
        self.training_metrics = []
        self.epoch_summaries = []
        self.model_checkpoints = []
        
        # 设置基础日志记录器
        self.logger = logging.getLogger(f"enhanced_logger_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            handler = logging.FileHandler(self.training_log_path, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # 记录实验开始
        self.experiment_start_time = datetime.now()
        self.log_experiment_start()
    
    def log_experiment_start(self):
        """记录实验开始信息"""
        start_info = {
            "experiment_name": self.experiment_name,
            "start_time": self.experiment_start_time.isoformat(),
            "log_directory": str(self.log_dir),
        }
        
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 实验开始: {self.experiment_name}")
        self.logger.info(f"📅 开始时间: {self.experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"📁 日志目录: {self.log_dir}")
        self.logger.info("=" * 80)
        
        # 保存实验信息
        experiment_info_path = self.log_dir / "experiment_info.json"
        with open(experiment_info_path, 'w', encoding='utf-8') as f:
            json.dump(start_info, f, indent=2, ensure_ascii=False)
    
    def log_model_info(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                      criterion: torch.nn.Module, config: Dict[str, Any]):
        """记录模型和训练配置信息"""
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "model_class": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # 假设float32
            "optimizer_class": optimizer.__class__.__name__,
            "criterion_class": criterion.__class__.__name__,
            "config": config,
        }
        
        self.logger.info("📊 模型信息:")
        self.logger.info(f"  模型类型: {model_info['model_class']}")
        self.logger.info(f"  总参数数: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        self.logger.info(f"  模型大小: {model_info['model_size_mb']:.2f} MB")
        self.logger.info(f"  优化器: {model_info['optimizer_class']}")
        self.logger.info(f"  损失函数: {model_info['criterion_class']}")
        
        # 保存模型信息
        model_info_path = self.log_dir / "model_info.json"
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False, default=str)
    
    def log_epoch_start(self, epoch: int, total_epochs: int, learning_rate: float):
        """记录epoch开始信息"""
        self.logger.info(f"\n📈 Epoch {epoch}/{total_epochs} 开始")
        self.logger.info(f"📚 学习率: {learning_rate:.2e}")
        
        epoch_start_info = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "learning_rate": learning_rate,
            "start_time": datetime.now().isoformat(),
        }
        
        return epoch_start_info
    
    def log_batch_progress(self, epoch: int, batch_idx: int, total_batches: int, 
                          loss: float, batch_time: float, data_time: float,
                          gpu_memory_mb: Optional[float] = None,
                          gpu_utilization: Optional[float] = None):
        """记录batch训练进度"""
        progress_percent = (batch_idx + 1) / total_batches * 100
        
        log_msg = (
            f"    🔄 Batch [{batch_idx + 1:4d}/{total_batches}] ({progress_percent:5.1f}%) | "
            f"Loss: {loss:.6f} | "
            f"Batch时间: {batch_time:.3f}s | "
            f"数据时间: {data_time:.3f}s"
        )
        
        if gpu_memory_mb is not None:
            log_msg += f" | GPU内存: {gpu_memory_mb:.0f}MB"
        
        if gpu_utilization is not None:
            log_msg += f" | GPU使用率: {gpu_utilization:.0f}%"
        
        self.logger.info(log_msg)
        
        # 记录详细的batch指标
        batch_metrics = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "batch_idx": batch_idx,
            "total_batches": total_batches,
            "progress_percent": progress_percent,
            "loss": loss,
            "batch_time_seconds": batch_time,
            "data_time_seconds": data_time,
            "gpu_memory_mb": gpu_memory_mb,
            "gpu_utilization_percent": gpu_utilization,
        }
        
        self.training_metrics.append(batch_metrics)
    
    def log_epoch_summary(self, epoch: int, epoch_start_info: Dict[str, Any],
                         train_loss: float, valid_loss: float, test_loss: Optional[float],
                         epoch_time: float, best_loss: float, patience_counter: int,
                         model_saved: bool = False,
                         gradient_norm: Optional[float] = None,
                         learning_rate: Optional[float] = None):
        """记录epoch总结信息"""
        
        epoch_summary = {
            "epoch": epoch,
            "start_time": epoch_start_info["start_time"],
            "end_time": datetime.now().isoformat(),
            "epoch_duration_seconds": epoch_time,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "test_loss": test_loss,
            "best_loss": best_loss,
            "patience_counter": patience_counter,
            "model_saved": model_saved,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
        }
        
        self.epoch_summaries.append(epoch_summary)
        
        # 记录epoch总结
        self.logger.info(f"\n📊 Epoch {epoch} 总结:")
        self.logger.info(f"  ⏱️  训练时长: {epoch_time:.2f}秒")
        self.logger.info(f"  📉 训练损失: {train_loss:.6f}")
        self.logger.info(f"  📊 验证损失: {valid_loss:.6f}")
        if test_loss is not None:
            self.logger.info(f"  🧪 测试损失: {test_loss:.6f}")
        self.logger.info(f"  🏆 最佳损失: {best_loss:.6f}")
        self.logger.info(f"  ⏳ 早停计数: {patience_counter}")
        if gradient_norm is not None:
            self.logger.info(f"  📐 梯度范数: {gradient_norm:.6f}")
        if learning_rate is not None:
            self.logger.info(f"  📚 学习率: {learning_rate:.2e}")
        if model_saved:
            self.logger.info(f"  💾 模型已保存 ✅")
        
        self.logger.info("-" * 60)
    
    def log_model_checkpoint(self, epoch: int, model_path: str, 
                           checkpoint_info: Dict[str, Any]):
        """记录模型检查点信息"""
        checkpoint_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "checkpoint_info": checkpoint_info,
        }
        
        self.model_checkpoints.append(checkpoint_data)
        
        self.logger.info(f"💾 模型检查点已保存: {model_path}")
    
    def log_training_complete(self, total_epochs: int, best_epoch: int, 
                            best_loss: float, total_time: float,
                            final_model_path: Optional[str] = None):
        """记录训练完成信息"""
        completion_time = datetime.now()
        
        completion_info = {
            "experiment_name": self.experiment_name,
            "start_time": self.experiment_start_time.isoformat(),
            "completion_time": completion_time.isoformat(),
            "total_duration_seconds": total_time,
            "total_epochs": total_epochs,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "final_model_path": final_model_path,
        }
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎉 训练完成!")
        self.logger.info(f"📅 完成时间: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"⏱️  总训练时长: {total_time:.2f}秒 ({total_time/3600:.2f}小时)")
        self.logger.info(f"📈 总epoch数: {total_epochs}")
        self.logger.info(f"🏆 最佳epoch: {best_epoch}")
        self.logger.info(f"📉 最佳损失: {best_loss:.6f}")
        if final_model_path:
            self.logger.info(f"💾 最终模型: {final_model_path}")
        self.logger.info("=" * 80)
        
        # 保存完成信息
        completion_info_path = self.log_dir / "training_completion.json"
        with open(completion_info_path, 'w', encoding='utf-8') as f:
            json.dump(completion_info, f, indent=2, ensure_ascii=False)
        
        # 保存所有指标
        self.save_all_metrics()
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """记录错误信息"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_message": error_msg,
            "exception_type": type(exception).__name__ if exception else None,
            "exception_details": str(exception) if exception else None,
        }
        
        self.logger.error(f"❌ 错误: {error_msg}")
        if exception:
            self.logger.error(f"   异常类型: {type(exception).__name__}")
            self.logger.error(f"   异常详情: {str(exception)}")
        
        # 保存错误信息
        error_log_path = self.log_dir / "errors.json"
        errors = []
        if error_log_path.exists():
            with open(error_log_path, 'r', encoding='utf-8') as f:
                errors = json.load(f)
        
        errors.append(error_info)
        
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
    
    def save_all_metrics(self):
        """保存所有收集的指标"""
        # 保存为JSON格式
        all_metrics = {
            "experiment_info": {
                "name": self.experiment_name,
                "start_time": self.experiment_start_time.isoformat(),
                "completion_time": datetime.now().isoformat(),
            },
            "training_metrics": self.training_metrics,
            "epoch_summaries": self.epoch_summaries,
            "model_checkpoints": self.model_checkpoints,
        }
        
        with open(self.metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        
        # 保存epoch摘要为CSV格式
        if self.epoch_summaries:
            with open(self.metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = self.epoch_summaries[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.epoch_summaries)
        
        self.logger.info(f"📊 所有指标已保存到: {self.metrics_dir}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.epoch_summaries:
            return {}
        
        train_losses = [epoch['train_loss'] for epoch in self.epoch_summaries]
        valid_losses = [epoch['valid_loss'] for epoch in self.epoch_summaries]
        epoch_times = [epoch['epoch_duration_seconds'] for epoch in self.epoch_summaries]
        
        stats = {
            "total_epochs": len(self.epoch_summaries),
            "best_train_loss": min(train_losses),
            "best_valid_loss": min(valid_losses),
            "final_train_loss": train_losses[-1],
            "final_valid_loss": valid_losses[-1],
            "avg_epoch_time_seconds": sum(epoch_times) / len(epoch_times),
            "total_training_time_seconds": sum(epoch_times),
            "best_epoch": valid_losses.index(min(valid_losses)) + 1,
        }
        
        return stats


def create_enhanced_logger(log_dir: str, experiment_name: str) -> EnhancedTrainingLogger:
    """创建增强训练日志记录器的工厂函数"""
    return EnhancedTrainingLogger(log_dir=log_dir, experiment_name=experiment_name)