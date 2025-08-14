"""硬件监控模块

提供GPU使用情况、显存占用、训练时间等硬件和性能指标的监控功能。
"""

import time
import psutil
import platform
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
    import GPUtil
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HardwareMonitor:
    """硬件监控器类
    
    监控和记录训练过程中的硬件使用情况，包括：
    - GPU使用率和显存占用
    - CPU使用率和内存占用
    - 训练时间统计
    - 系统信息
    """
    
    def __init__(self, log_dir: Optional[str] = None, enable_gpu_monitoring: bool = True):
        """初始化硬件监控器
        
        Args:
            log_dir: 日志保存目录
            enable_gpu_monitoring: 是否启用GPU监控
        """
        self.logger = logging.getLogger(__name__)
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPU_AVAILABLE
        self.start_time = None
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # 监控数据存储
        self.training_metrics = []
        self.epoch_metrics = []
        self.batch_metrics = []
        
        # 初始化GPU监控
        if self.enable_gpu_monitoring:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.logger.info(f"检测到 {self.gpu_count} 个GPU设备")
            except Exception as e:
                self.logger.warning(f"GPU监控初始化失败: {e}")
                self.enable_gpu_monitoring = False
        
        # 记录系统信息
        self._log_system_info()
    
    def _log_system_info(self):
        """记录系统信息"""
        system_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": platform.python_version(),
        }
        
        if TORCH_AVAILABLE:
            system_info["torch_version"] = torch.__version__
            system_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info["cuda_version"] = torch.version.cuda
                system_info["cudnn_version"] = torch.backends.cudnn.version()
        
        if self.enable_gpu_monitoring:
            gpu_info = []
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                # Normalize to str for different pynvml versions (bytes on some, str on others)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info.append({
                    "id": i,
                    "name": name,
                    "memory_total_mb": round(memory_info.total / (1024**2), 2),
                })
            system_info["gpu_devices"] = gpu_info
        
        # 保存系统信息
        system_info_path = self.log_dir / "system_info.json"
        with open(system_info_path, 'w', encoding='utf-8') as f:
            json.dump(system_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"系统信息已保存到: {system_info_path}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前硬件指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        }
        
        if self.enable_gpu_monitoring:
            gpu_metrics = []
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                    
                    gpu_metrics.append({
                        "gpu_id": i,
                        "gpu_utilization_percent": utilization.gpu,
                        "memory_utilization_percent": utilization.memory,
                        "memory_used_mb": round(memory_info.used / (1024**2), 2),
                        "memory_total_mb": round(memory_info.total / (1024**2), 2),
                        "memory_free_mb": round(memory_info.free / (1024**2), 2),
                        "temperature_celsius": temperature,
                        "power_usage_watts": round(power, 2),
                    })
                except Exception as e:
                    self.logger.warning(f"获取GPU {i} 指标失败: {e}")
            
            metrics["gpu_metrics"] = gpu_metrics
        
        # 添加PyTorch GPU内存信息（如果可用）
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch_gpu_metrics = []
            for i in range(torch.cuda.device_count()):
                torch_gpu_metrics.append({
                    "gpu_id": i,
                    "allocated_mb": round(torch.cuda.memory_allocated(i) / (1024**2), 2),
                    "cached_mb": round(torch.cuda.memory_reserved(i) / (1024**2), 2),
                    "max_allocated_mb": round(torch.cuda.max_memory_allocated(i) / (1024**2), 2),
                })
            metrics["torch_gpu_metrics"] = torch_gpu_metrics
        
        return metrics
    
    def start_training(self):
        """开始训练监控"""
        self.start_time = time.time()
        self.logger.info("🚀 开始训练监控")
        
        # 记录训练开始时的硬件状态
        initial_metrics = self.get_current_metrics()
        initial_metrics["event"] = "training_start"
        self.training_metrics.append(initial_metrics)
    
    def start_epoch(self, epoch: int):
        """开始epoch监控"""
        self.epoch_start_time = time.time()
        
        epoch_metrics = self.get_current_metrics()
        epoch_metrics["event"] = "epoch_start"
        epoch_metrics["epoch"] = epoch
        self.epoch_metrics.append(epoch_metrics)
    
    def end_epoch(self, epoch: int, train_loss: float, valid_loss: float, test_loss: Optional[float] = None):
        """结束epoch监控"""
        if self.epoch_start_time is None:
            self.logger.warning("Epoch开始时间未记录")
            return
        
        epoch_duration = time.time() - self.epoch_start_time
        
        epoch_metrics = self.get_current_metrics()
        epoch_metrics.update({
            "event": "epoch_end",
            "epoch": epoch,
            "epoch_duration_seconds": round(epoch_duration, 2),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        })
        
        if test_loss is not None:
            epoch_metrics["test_loss"] = test_loss
        
        self.epoch_metrics.append(epoch_metrics)
        
        # 记录详细的epoch信息
        self.logger.info(
            f"📊 Epoch {epoch} 完成 - "
            f"时长: {epoch_duration:.2f}s, "
            f"训练损失: {train_loss:.6f}, "
            f"验证损失: {valid_loss:.6f}"
            + (f", 测试损失: {test_loss:.6f}" if test_loss else "")
        )
        
        # 记录GPU信息（如果可用）
        if self.enable_gpu_monitoring and "gpu_metrics" in epoch_metrics:
            for gpu_metric in epoch_metrics["gpu_metrics"]:
                self.logger.info(
                    f"🎮 GPU {gpu_metric['gpu_id']}: "
                    f"使用率 {gpu_metric['gpu_utilization_percent']}%, "
                    f"显存 {gpu_metric['memory_used_mb']:.0f}/{gpu_metric['memory_total_mb']:.0f}MB "
                    f"({gpu_metric['memory_used_mb']/gpu_metric['memory_total_mb']*100:.1f}%), "
                    f"温度 {gpu_metric['temperature_celsius']}°C, "
                    f"功耗 {gpu_metric['power_usage_watts']}W"
                )
    
    def start_batch(self, epoch: int, batch_idx: int):
        """开始batch监控（可选，用于详细监控）"""
        self.batch_start_time = time.time()
    
    def end_batch(self, epoch: int, batch_idx: int, batch_loss: float, log_interval: int = 50):
        """结束batch监控"""
        if self.batch_start_time is None:
            return
        
        batch_duration = time.time() - self.batch_start_time
        
        # 只在指定间隔记录详细的batch指标
        if batch_idx % log_interval == 0:
            batch_metrics = self.get_current_metrics()
            batch_metrics.update({
                "event": "batch_end",
                "epoch": epoch,
                "batch_idx": batch_idx,
                "batch_duration_seconds": round(batch_duration, 4),
                "batch_loss": batch_loss,
            })
            self.batch_metrics.append(batch_metrics)
    
    def end_training(self):
        """结束训练监控"""
        if self.start_time is None:
            self.logger.warning("训练开始时间未记录")
            return
        
        total_duration = time.time() - self.start_time
        
        final_metrics = self.get_current_metrics()
        final_metrics.update({
            "event": "training_end",
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_formatted": str(timedelta(seconds=int(total_duration))),
        })
        self.training_metrics.append(final_metrics)
        
        self.logger.info(f"✅ 训练完成，总时长: {timedelta(seconds=int(total_duration))}")
        
        # 保存所有监控数据
        self.save_metrics()
    
    def save_metrics(self):
        """保存监控指标到文件"""
        try:
            # 确保目录存在
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"硬件监控目录: {self.log_dir}")
            
            metrics_data = {
                "training_metrics": self.training_metrics,
                "epoch_metrics": self.epoch_metrics,
                "batch_metrics": self.batch_metrics,
            }
            
            self.logger.info(f"准备保存硬件监控数据，包含 {len(self.training_metrics)} 个训练指标，{len(self.epoch_metrics)} 个epoch指标")
            
            # 保存为JSON格式
            metrics_path = self.log_dir / "hardware_metrics.json"
            self.logger.info(f"保存路径: {metrics_path}")
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            # 验证文件是否成功保存
            if metrics_path.exists():
                file_size = metrics_path.stat().st_size
                self.logger.info(f"硬件监控数据已成功保存到: {metrics_path}，文件大小: {file_size} 字节")
            else:
                self.logger.error(f"文件保存失败，文件不存在: {metrics_path}")
            
            # 保存为CSV格式（便于分析）
            self._save_metrics_csv()
            
        except Exception as e:
            self.logger.error(f"保存硬件监控数据失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
    
    def _save_metrics_csv(self):
        """保存指标为CSV格式"""
        import csv
        
        # 保存epoch指标
        if self.epoch_metrics:
            epoch_csv_path = self.log_dir / "epoch_metrics.csv"
            with open(epoch_csv_path, 'w', newline='', encoding='utf-8') as f:
                if self.epoch_metrics:
                    # 获取所有可能的字段
                    fieldnames = set()
                    for metric in self.epoch_metrics:
                        fieldnames.update(metric.keys())
                        if "gpu_metrics" in metric:
                            for gpu_metric in metric["gpu_metrics"]:
                                for key in gpu_metric.keys():
                                    fieldnames.add(f"gpu_{key}")
                    
                    fieldnames = sorted(list(fieldnames))
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for metric in self.epoch_metrics:
                        row = {k: v for k, v in metric.items() if k != "gpu_metrics"}
                        if "gpu_metrics" in metric:
                            # 假设只有一个GPU或取第一个GPU的数据
                            if metric["gpu_metrics"]:
                                gpu_data = metric["gpu_metrics"][0]
                                for key, value in gpu_data.items():
                                    row[f"gpu_{key}"] = value
                        writer.writerow(row)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取训练过程的统计摘要"""
        if not self.epoch_metrics:
            return {}
        
        # 计算训练统计
        epoch_durations = [m.get("epoch_duration_seconds", 0) for m in self.epoch_metrics if m.get("event") == "epoch_end"]
        train_losses = [m.get("train_loss", 0) for m in self.epoch_metrics if m.get("event") == "epoch_end" and "train_loss" in m]
        
        summary = {
            "total_epochs": len([m for m in self.epoch_metrics if m.get("event") == "epoch_end"]),
            "avg_epoch_duration_seconds": round(sum(epoch_durations) / len(epoch_durations), 2) if epoch_durations else 0,
            "total_training_time_seconds": round(sum(epoch_durations), 2),
            "final_train_loss": train_losses[-1] if train_losses else None,
            "best_train_loss": min(train_losses) if train_losses else None,
        }
        
        # GPU统计（如果可用）
        if self.enable_gpu_monitoring:
            gpu_utilizations = []
            gpu_memory_usages = []
            gpu_temperatures = []
            
            for metric in self.epoch_metrics:
                if "gpu_metrics" in metric and metric["gpu_metrics"]:
                    gpu_data = metric["gpu_metrics"][0]  # 取第一个GPU
                    gpu_utilizations.append(gpu_data.get("gpu_utilization_percent", 0))
                    gpu_memory_usages.append(gpu_data.get("memory_used_mb", 0))
                    gpu_temperatures.append(gpu_data.get("temperature_celsius", 0))
            
            if gpu_utilizations:
                summary.update({
                    "avg_gpu_utilization_percent": round(sum(gpu_utilizations) / len(gpu_utilizations), 2),
                    "max_gpu_utilization_percent": max(gpu_utilizations),
                    "avg_gpu_memory_usage_mb": round(sum(gpu_memory_usages) / len(gpu_memory_usages), 2),
                    "max_gpu_memory_usage_mb": max(gpu_memory_usages),
                    "avg_gpu_temperature_celsius": round(sum(gpu_temperatures) / len(gpu_temperatures), 2),
                    "max_gpu_temperature_celsius": max(gpu_temperatures),
                })
        
        return summary


def create_hardware_monitor(log_dir: str, enable_gpu_monitoring: bool = True) -> HardwareMonitor:
    """创建硬件监控器的工厂函数"""
    return HardwareMonitor(log_dir=log_dir, enable_gpu_monitoring=enable_gpu_monitoring)