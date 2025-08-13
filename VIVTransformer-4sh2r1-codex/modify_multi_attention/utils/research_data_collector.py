"""
研究数据收集器 - 为论文生成提供全面的实验数据支持

该模块自动收集和整理训练过程中的各种数据，包括：
- 实验配置归档
- 系统环境信息
- 模型复杂度分析
- 训练性能监控
- 收敛性分析
- 注意力机制分析
"""

import json
import yaml
import shutil
import time
import platform
import psutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class ResearchDataCollector:
    """研究数据收集器 - 集成到训练流程中的数据收集系统"""
    
    def __init__(self, result_dir: Union[str, Path], experiment_name: str = "", config: Optional[Dict] = None):
        """
        初始化研究数据收集器
        
        Args:
            result_dir: 结果保存目录
            experiment_name: 实验名称
            config: 研究数据收集配置字典
        """
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # 配置设置（带默认值）
        self.config = config or {}
        self.lightweight_mode = self.config.get("lightweight_mode", False)
        self.buffered_writes = self.config.get("buffered_writes", True)
        self.log_every_n = max(1, int(self.config.get("log_every_n_epochs", 1)))
        
        # 功能开关（避免与方法名冲突，使用 enable_* 前缀）
        self.enable_collect_system_info = self.config.get("collect_system_info", True)
        self.enable_collect_model_complexity = self.config.get("collect_model_complexity", True)
        self.enable_collect_dataset_info = self.config.get("collect_dataset_info", True)
        self.enable_collect_training_metrics = self.config.get("collect_training_metrics", True)
        
        # 创建数据收集子目录
        self.research_data_dir = self.result_dir / "research_data"
        self.research_data_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.metrics = {}
        self.training_log = []
        self.system_info = {}
        
        # 缓冲区（如果启用缓冲写入）
        self._training_buffer = []
        self._buffer_size = 10  # 缓冲区大小
        
        logger.info(f"研究数据收集器已初始化: {self.research_data_dir}, 轻量级模式: {self.lightweight_mode}")
    
    def save_experiment_config(self, config: Dict, attention_type: str, loss_config_id: Optional[str] = None):
        """
        保存实验配置文件
        
        Args:
            config: 主配置字典
            attention_type: 注意力机制类型
            loss_config_id: 损失配置ID
        """
        try:
            config_dir = self.research_data_dir / "configs"
            config_dir.mkdir(exist_ok=True)
            
            # 保存主配置
            main_config_path = config_dir / "main_config.yaml"
            with open(main_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 保存损失配置（如果存在）
            if loss_config_id is not None:
                loss_config_src = Path("configs/loss_configs") / f"loss_config_{loss_config_id}.yaml"
                if loss_config_src.exists():
                    shutil.copy2(loss_config_src, config_dir / f"loss_config_{loss_config_id}.yaml")
            
            # 保存性能配置
            perf_config_src = Path("configs/performance_config.yaml")
            if perf_config_src.exists():
                shutil.copy2(perf_config_src, config_dir / "performance_config.yaml")
            
            # 保存实验元信息
            experiment_meta = {
                "experiment_name": self.experiment_name,
                "attention_type": attention_type,
                "loss_config_id": loss_config_id,
                "start_time": self.start_time.isoformat(),
                "config_files_saved": {
                    "main_config": "main_config.yaml",
                    "loss_config": f"loss_config_{loss_config_id}.yaml" if loss_config_id else None,
                    "performance_config": "performance_config.yaml" if perf_config_src.exists() else None
                }
            }
            
            meta_path = config_dir / "experiment_meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(experiment_meta, f, indent=2, ensure_ascii=False)
            
            logger.info(f"实验配置已保存到: {config_dir}")
            
        except Exception as e:
            logger.warning(f"保存实验配置失败: {e}")
    
    def save_system_info(self):
        """保存系统和环境信息"""
        try:
            self.system_info = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "platform": platform.platform(),
                    "architecture": platform.architecture(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version(),
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                },
                "pytorch": {
                    "version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                    "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
            }
            
            # 获取GPU详细信息
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        "device_id": i,
                        "name": gpu_props.name,
                        "memory_total_gb": round(gpu_props.total_memory / (1024**3), 2),
                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                        "multiprocessor_count": gpu_props.multi_processor_count
                    })
                self.system_info["pytorch"]["gpu_details"] = gpu_info
            
            system_info_path = self.research_data_dir / "system_info.json"
            with open(system_info_path, 'w', encoding='utf-8') as f:
                json.dump(self.system_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"系统信息已保存到: {system_info_path}")
            
        except Exception as e:
            logger.warning(f"保存系统信息失败: {e}")
    
    def collect_model_complexity(self, model: torch.nn.Module, sample_input: torch.Tensor, attention_type: str):
        """
        收集模型复杂度信息
        
        Args:
            model: 模型实例
            sample_input: 示例输入
            attention_type: 注意力机制类型
        """
        try:
            # 基本参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 模型大小估算 (MB)
            model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
            
            complexity_info = {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": total_params - trainable_params,
                "model_size_mb": round(model_size_mb, 2),
                "parameter_efficiency": round(trainable_params / total_params * 100, 2)
            }
            
            # 尝试计算FLOPs（根据配置控制）
            model_analysis_config = self.config.get("model_analysis", {})
            compute_flops = model_analysis_config.get("compute_flops", True)
            
            if compute_flops and not self.lightweight_mode:
                try:
                    from thop import profile, clever_format
                    # 创建输入副本避免修改原始数据
                    sample_batch_size = model_analysis_config.get("sample_batch_size", 2)
                    dummy_input = sample_input.clone()[:sample_batch_size] if sample_input.dim() > 0 else sample_input
                    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                    flops_human, params_human = clever_format([flops, params], "%.3f")
                    
                    complexity_info.update({
                        "flops": flops,
                        "flops_human": flops_human,
                        "params_profile": params_human
                    })
                except Exception as e:
                    logger.warning(f"FLOPs 计算失败: {e}")
                    complexity_info.update({
                        "flops": "计算失败",
                        "flops_human": "N/A",
                        "params_profile": "N/A"
                    })
            else:
                complexity_info.update({
                    "flops": "已跳过（配置或轻量级模式）",
                    "flops_human": "N/A",
                    "params_profile": "N/A"
                })
            
            # 保存到metrics
            if attention_type not in self.metrics:
                self.metrics[attention_type] = {}
            self.metrics[attention_type]["model_complexity"] = complexity_info
            
            logger.info(f"模型复杂度信息已收集: {attention_type}")
            
        except Exception as e:
            logger.warning(f"收集模型复杂度信息失败: {e}")
    
    def collect_dataset_info(self, train_loader, val_loader, test_loader):
        """收集数据集信息"""
        try:
            dataset_info = {
                "train_size": len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else "Unknown",
                "val_size": len(val_loader.dataset) if hasattr(val_loader.dataset, '__len__') else "Unknown",
                "test_size": len(test_loader.dataset) if hasattr(test_loader.dataset, '__len__') else "Unknown",
                "batch_size": train_loader.batch_size,
                "num_workers": train_loader.num_workers,
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                "test_batches": len(test_loader),
                "pin_memory": getattr(train_loader, 'pin_memory', False),
                "drop_last": getattr(train_loader, 'drop_last', False),
            }
            
            # 尝试获取样本形状
            try:
                sample_batch = next(iter(train_loader))
                if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
                    dataset_info["input_shape"] = list(sample_batch[0].shape)
                    dataset_info["target_shape"] = list(sample_batch[1].shape)
                    dataset_info["input_dtype"] = str(sample_batch[0].dtype)
                    dataset_info["target_dtype"] = str(sample_batch[1].dtype)
            except Exception as e:
                logger.warning(f"获取样本形状失败: {e}")
            
            # 保存数据集信息
            dataset_path = self.research_data_dir / "dataset_info.json"
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"数据集信息已保存到: {dataset_path}")
            
        except Exception as e:
            logger.warning(f"收集数据集信息失败: {e}")
    
    def log_training_metrics(self, epoch: int, train_loss: float, val_loss: float, 
                           test_loss: Optional[float] = None, learning_rate: float = None,
                           epoch_time: float = None, memory_usage: float = None):
        """记录训练过程中的指标"""
        try:
            metrics_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "timestamp": datetime.now().isoformat(),
            }
            
            if test_loss is not None:
                metrics_entry["test_loss"] = test_loss
            if learning_rate is not None:
                metrics_entry["learning_rate"] = learning_rate
            if epoch_time is not None:
                metrics_entry["epoch_time"] = epoch_time
            if memory_usage is not None:
                metrics_entry["memory_usage_mb"] = memory_usage
            
            # 添加到训练日志
            self.training_log.append(metrics_entry)
            
        except Exception as e:
            logger.warning(f"记录训练指标失败: {e}")
    
    def save_convergence_analysis(self, attention_type: str):
        """保存收敛性分析数据"""
        try:
            if not self.training_log:
                logger.warning("没有训练日志数据可用于收敛性分析")
                return
            
            # 提取损失数据
            epochs = [entry["epoch"] for entry in self.training_log]
            train_losses = [entry["train_loss"] for entry in self.training_log]
            val_losses = [entry["val_loss"] for entry in self.training_log]
            test_losses = [entry.get("test_loss") for entry in self.training_log if entry.get("test_loss") is not None]
            
            # 收敛性统计
            convergence_stats = {
                "total_epochs": len(epochs),
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "best_val_loss": min(val_losses) if val_losses else None,
                "best_val_epoch": val_losses.index(min(val_losses)) + 1 if val_losses else None,
            }
            
            # 计算收敛指标
            if len(train_losses) > 10:
                # 早期vs后期损失改善
                early_loss = np.mean(train_losses[:10])
                late_loss = np.mean(train_losses[-10:])
                convergence_stats["loss_improvement_ratio"] = early_loss / late_loss if late_loss > 0 else float('inf')
                
                # 损失稳定性 (最后10个epoch的标准差)
                convergence_stats["late_stage_stability"] = float(np.std(train_losses[-10:]))
            
            # 保存原始数据
            convergence_data = {
                "attention_type": attention_type,
                "convergence_statistics": convergence_stats,
                "training_history": {
                    "epochs": epochs,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "test_losses": test_losses if test_losses else []
                }
            }
            
            convergence_path = self.research_data_dir / f"convergence_analysis_{attention_type}.json"
            with open(convergence_path, 'w', encoding='utf-8') as f:
                json.dump(convergence_data, f, indent=2, ensure_ascii=False)
            
            # 生成收敛性可视化（根据配置控制）
            conv_cfg = self.config.get("convergence_analysis", {})
            if conv_cfg.get("enabled", True) and conv_cfg.get("generate_plots", True) and not self.lightweight_mode:
                self._create_convergence_plots(attention_type, epochs, train_losses, val_losses, test_losses)
            
            logger.info(f"收敛性分析已保存: {convergence_path}")
            
        except Exception as e:
            logger.warning(f"保存收敛性分析失败: {e}")
    
    def _create_convergence_plots(self, attention_type: str, epochs: List[int], 
                                train_losses: List[float], val_losses: List[float], 
                                test_losses: List[float]):
        """创建收敛性分析图表"""
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{attention_type.upper()} - Convergence Analysis', fontsize=16, fontweight='bold')
            
            # 子图1: 基本损失曲线
            axes[0, 0].plot(epochs, train_losses, label='Training Loss', alpha=0.8, linewidth=1.5)
            axes[0, 0].plot(epochs, val_losses, label='Validation Loss', alpha=0.8, linewidth=1.5)
            if test_losses:
                test_epochs = epochs[:len(test_losses)]
                axes[0, 0].plot(test_epochs, test_losses, label='Test Loss', alpha=0.8, linewidth=1.5)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 子图2: 对数尺度损失
            axes[0, 1].semilogy(epochs, train_losses, label='Training Loss', alpha=0.8)
            axes[0, 1].semilogy(epochs, val_losses, label='Validation Loss', alpha=0.8)
            if test_losses:
                axes[0, 1].semilogy(test_epochs, test_losses, label='Test Loss', alpha=0.8)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss (log scale)')
            axes[0, 1].set_title('Loss Curves (Log Scale)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 子图3: 平滑损失曲线
            if len(train_losses) > 10:
                window = min(10, len(train_losses) // 10)
                smoothed_train = np.convolve(train_losses, np.ones(window)/window, mode='valid')
                smoothed_val = np.convolve(val_losses, np.ones(window)/window, mode='valid')
                smooth_epochs = epochs[window-1:]
                
                axes[1, 0].plot(smooth_epochs, smoothed_train, label='Smoothed Training', alpha=0.8)
                axes[1, 0].plot(smooth_epochs, smoothed_val, label='Smoothed Validation', alpha=0.8)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Smoothed Loss')
                axes[1, 0].set_title(f'Smoothed Loss (window={window})')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # 子图4: 过拟合检测
            overfitting_gap = np.array(val_losses) - np.array(train_losses)
            axes[1, 1].plot(epochs, overfitting_gap, label='Validation - Training', alpha=0.8, color='red')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Gap')
            axes[1, 1].set_title('Overfitting Detection')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            plots_dir = self.research_data_dir / "convergence_plots"
            plots_dir.mkdir(exist_ok=True)
            plot_path = plots_dir / f"convergence_analysis_{attention_type}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"收敛性分析图表已保存: {plot_path}")
            
        except Exception as e:
            logger.warning(f"创建收敛性图表失败: {e}")
    
    def finalize_experiment(self, attention_type: str, final_test_loss: float = None):
        """完成实验数据收集"""
        try:
            # 保存收敛性分析
            self.save_convergence_analysis(attention_type)
            
            # 保存综合指标
            if attention_type not in self.metrics:
                self.metrics[attention_type] = {}
            
            # 添加实验总结
            end_time = datetime.now()
            total_duration = (end_time - self.start_time).total_seconds()
            
            experiment_summary = {
                "experiment_name": self.experiment_name,
                "attention_type": attention_type,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_duration_human": str(timedelta(seconds=int(total_duration))),
                "total_epochs": len(self.training_log),
                "final_test_loss": final_test_loss
            }
            
            self.metrics[attention_type]["experiment_summary"] = experiment_summary
            
            # 保存所有指标
            metrics_path = self.research_data_dir / f"comprehensive_metrics_{attention_type}.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics[attention_type], f, indent=2, ensure_ascii=False, default=str)
            
            # 保存训练日志
            training_log_path = self.research_data_dir / f"training_log_{attention_type}.json"
            with open(training_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_log, f, indent=2, ensure_ascii=False)
            
            logger.info(f"实验数据收集完成: {self.research_data_dir}")
            
        except Exception as e:
            logger.warning(f"完成实验数据收集失败: {e}")
    
    def generate_summary_report(self, attention_types: List[str]):
        """生成多个注意力机制的对比总结报告"""
        try:
            summary_data = {
                "report_generated": datetime.now().isoformat(),
                "experiment_name": self.experiment_name,
                "system_info": self.system_info,
                "attention_mechanisms": {}
            }
            
            # 收集所有注意力机制的数据
            for attention_type in attention_types:
                metrics_file = self.research_data_dir / f"comprehensive_metrics_{attention_type}.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        summary_data["attention_mechanisms"][attention_type] = json.load(f)
            
            # 生成对比表格数据
            comparison_table = self._generate_comparison_table(summary_data["attention_mechanisms"])
            summary_data["performance_comparison"] = comparison_table
            
            # 保存总结报告
            summary_path = self.research_data_dir / "experiment_summary_report.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"总结报告已生成: {summary_path}")
            return summary_path
            
        except Exception as e:
            logger.warning(f"生成总结报告失败: {e}")
            return None
    
    def _generate_comparison_table(self, attention_data: Dict) -> Dict:
        """生成注意力机制对比表格"""
        try:
            comparison = {
                "performance_metrics": {},
                "model_complexity": {},
                "training_efficiency": {}
            }
            
            for attention_type, data in attention_data.items():
                # 性能指标
                if "experiment_summary" in data:
                    summary = data["experiment_summary"]
                    comparison["performance_metrics"][attention_type] = {
                        "final_test_loss": summary.get("final_test_loss"),
                        "total_epochs": summary.get("total_epochs"),
                        "training_duration_hours": round(summary.get("total_duration_seconds", 0) / 3600, 2)
                    }
                
                # 模型复杂度
                if "model_complexity" in data:
                    complexity = data["model_complexity"]
                    comparison["model_complexity"][attention_type] = {
                        "total_parameters": complexity.get("total_parameters"),
                        "trainable_parameters": complexity.get("trainable_parameters"),
                        "model_size_mb": complexity.get("model_size_mb"),
                        "flops": complexity.get("flops")
                    }
            
            return comparison
            
        except Exception as e:
            logger.warning(f"生成对比表格失败: {e}")
            return {}