# 🚀 性能优化指南

> 全面的VIVTransformer性能优化策略和最佳实践

---

## 📋 目录

- [🎯 优化概览](#-优化概览)
- [💾 内存优化](#-内存优化)
- [⚡ 计算优化](#-计算优化)
- [🔄 数据流优化](#-数据流优化)
- [🖥️ GPU优化](#️-gpu优化)
- [📊 分布式训练](#-分布式训练)
- [🔧 模型优化](#-模型优化)
- [📈 监控与分析](#-监控与分析)
- [🎛️ 自动调优](#️-自动调优)
- [📝 最佳实践](#-最佳实践)

---

## 🎯 优化概览

### 性能瓶颈识别

```python
# performance_profiler.py
import torch
import time
import psutil
import GPUtil
from typing import Dict, List, Any
from contextlib import contextmanager

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.gpu_stats = []
        
    @contextmanager
    def profile(self, name: str):
        """性能分析上下文管理器"""
        
        # 记录开始状态
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        start_cpu = psutil.cpu_percent()
        
        try:
            yield
        finally:
            # 记录结束状态
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            end_cpu = psutil.cpu_percent()
            
            # 计算指标
            self.metrics[name] = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'cpu_usage': (start_cpu + end_cpu) / 2,
                'gpu_memory': end_memory / (1024**3) if torch.cuda.is_available() else 0
            }
    
    def get_gpu_utilization(self) -> Dict[str, float]:
        """获取GPU利用率"""
        
        if not torch.cuda.is_available():
            return {}
        
        gpus = GPUtil.getGPUs()
        gpu_stats = {}
        
        for i, gpu in enumerate(gpus):
            gpu_stats[f'gpu_{i}'] = {
                'utilization': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': gpu.memoryUtil * 100,
                'temperature': gpu.temperature
            }
        
        return gpu_stats
    
    def analyze_bottlenecks(self) -> Dict[str, str]:
        """分析性能瓶颈"""
        
        bottlenecks = {}
        
        # 分析内存使用
        memory_usage = sum(m.get('memory_delta', 0) for m in self.metrics.values())
        if memory_usage > 1e9:  # 1GB
            bottlenecks['memory'] = f"高内存使用: {memory_usage/1e9:.2f}GB"
        
        # 分析计算时间
        total_time = sum(m.get('duration', 0) for m in self.metrics.values())
        slow_operations = [name for name, metrics in self.metrics.items() 
                          if metrics.get('duration', 0) > total_time * 0.3]
        
        if slow_operations:
            bottlenecks['computation'] = f"慢操作: {', '.join(slow_operations)}"
        
        # 分析GPU利用率
        gpu_stats = self.get_gpu_utilization()
        for gpu_name, stats in gpu_stats.items():
            if stats['utilization'] < 50:
                bottlenecks[f'{gpu_name}_underutilized'] = f"GPU利用率低: {stats['utilization']:.1f}%"
            if stats['memory_percent'] > 90:
                bottlenecks[f'{gpu_name}_memory'] = f"GPU内存不足: {stats['memory_percent']:.1f}%"
        
        return bottlenecks
    
    def generate_report(self) -> str:
        """生成性能报告"""
        
        report = ["\n=== 性能分析报告 ==="]
        
        # 总体统计
        total_time = sum(m.get('duration', 0) for m in self.metrics.values())
        total_memory = sum(m.get('memory_delta', 0) for m in self.metrics.values())
        
        report.append(f"总执行时间: {total_time:.3f}秒")
        report.append(f"总内存使用: {total_memory/1e6:.2f}MB")
        
        # 详细指标
        report.append("\n详细指标:")
        for name, metrics in self.metrics.items():
            report.append(f"  {name}:")
            report.append(f"    时间: {metrics.get('duration', 0):.3f}秒")
            report.append(f"    内存: {metrics.get('memory_delta', 0)/1e6:.2f}MB")
            report.append(f"    CPU: {metrics.get('cpu_usage', 0):.1f}%")
        
        # 瓶颈分析
        bottlenecks = self.analyze_bottlenecks()
        if bottlenecks:
            report.append("\n发现的瓶颈:")
            for bottleneck_type, description in bottlenecks.items():
                report.append(f"  - {description}")
        
        return "\n".join(report)

# 使用示例
profiler = PerformanceProfiler()

with profiler.profile("model_forward"):
    output = model(input_data)

with profiler.profile("loss_computation"):
    loss = criterion(output, target)

print(profiler.generate_report())
```

### 性能基准测试

```python
# benchmark.py
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple

class VIVTransformerBenchmark:
    """VIVTransformer性能基准测试"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.results = {}
    
    def benchmark_forward_pass(self, 
                              batch_sizes: List[int] = [1, 8, 16, 32, 64],
                              sequence_lengths: List[int] = [50, 100, 200, 500],
                              num_runs: int = 100) -> Dict[str, float]:
        """前向传播基准测试"""
        
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # 准备数据
                input_data = torch.randn(batch_size, seq_len, 512).to(self.device)
                
                # 预热
                for _ in range(10):
                    with torch.no_grad():
                        _ = self.model(input_data)
                
                # 基准测试
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                for _ in range(num_runs):
                    with torch.no_grad():
                        _ = self.model(input_data)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                avg_time = (end_time - start_time) / num_runs
                throughput = batch_size / avg_time
                
                key = f"bs_{batch_size}_seq_{seq_len}"
                results[key] = {
                    'avg_time': avg_time,
                    'throughput': throughput,
                    'memory_used': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                }
                
                # 清理内存
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.results['forward_pass'] = results
        return results
    
    def benchmark_attention_mechanisms(self, 
                                     attention_types: List[str],
                                     input_size: Tuple[int, int, int] = (32, 100, 512)) -> Dict[str, float]:
        """注意力机制基准测试"""
        
        from modify_multi_attention.mymodels.components.attention_factory import get_attention_module
        
        results = {}
        batch_size, seq_len, d_model = input_size
        input_data = torch.randn(batch_size, seq_len, d_model).to(self.device)
        
        for attention_type in attention_types:
            try:
                # 创建注意力模块
                attention_module = get_attention_module(attention_type, d_model=d_model)
                attention_module = attention_module.to(self.device)
                
                # 预热
                for _ in range(10):
                    with torch.no_grad():
                        _ = attention_module(input_data)
                
                # 基准测试
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                num_runs = 50
                for _ in range(num_runs):
                    with torch.no_grad():
                        _ = attention_module(input_data)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                avg_time = (end_time - start_time) / num_runs
                
                results[attention_type] = {
                    'avg_time': avg_time,
                    'throughput': batch_size / avg_time,
                    'memory_used': torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                }
                
                # 清理内存
                del attention_module
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                results[attention_type] = {'error': str(e)}
        
        self.results['attention_mechanisms'] = results
        return results
    
    def generate_benchmark_report(self) -> str:
        """生成基准测试报告"""
        
        report = ["\n=== VIVTransformer 性能基准测试报告 ==="]
        
        # 前向传播结果
        if 'forward_pass' in self.results:
            report.append("\n前向传播性能:")
            report.append(f"{'配置':<20} {'平均时间(ms)':<15} {'吞吐量(samples/s)':<20} {'内存使用(GB)':<15}")
            report.append("-" * 70)
            
            for config, metrics in self.results['forward_pass'].items():
                report.append(f"{config:<20} {metrics['avg_time']*1000:<15.2f} {metrics['throughput']:<20.2f} {metrics['memory_used']:<15.2f}")
        
        # 注意力机制结果
        if 'attention_mechanisms' in self.results:
            report.append("\n注意力机制性能:")
            report.append(f"{'注意力类型':<25} {'平均时间(ms)':<15} {'吞吐量(samples/s)':<20} {'内存使用(GB)':<15}")
            report.append("-" * 75)
            
            for attention_type, metrics in self.results['attention_mechanisms'].items():
                if 'error' in metrics:
                    report.append(f"{attention_type:<25} ERROR: {metrics['error']}")
                else:
                    report.append(f"{attention_type:<25} {metrics['avg_time']*1000:<15.2f} {metrics['throughput']:<20.2f} {metrics['memory_used']:<15.2f}")
        
        return "\n".join(report)

# 使用示例
benchmark = VIVTransformerBenchmark(model)

# 前向传播基准测试
forward_results = benchmark.benchmark_forward_pass()

# 注意力机制基准测试
attention_types = ['ScaledDotProductAttention', 'ExternalAttention', 'SEAttention']
attention_results = benchmark.benchmark_attention_mechanisms(attention_types)

print(benchmark.generate_benchmark_report())
```

---

## 💾 内存优化

### 梯度累积

```python
# gradient_accumulation.py
import torch
import torch.nn as nn
from typing import Optional

class GradientAccumulator:
    """梯度累积器"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 accumulation_steps: int = 4,
                 max_grad_norm: Optional[float] = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
    
    def accumulate_gradients(self, loss: torch.Tensor) -> bool:
        """累积梯度"""
        
        # 缩放损失
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.step_count += 1
        
        # 检查是否需要更新参数
        if self.step_count % self.accumulation_steps == 0:
            # 梯度裁剪
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            return True  # 表示进行了参数更新
        
        return False  # 表示只是累积了梯度

# 使用示例
accumulator = GradientAccumulator(model, optimizer, accumulation_steps=8)

for batch_idx, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target)
    
    # 累积梯度
    updated = accumulator.accumulate_gradients(loss)
    
    if updated:
        print(f"参数更新在批次 {batch_idx}")
```

### 混合精度训练

```python
# mixed_precision.py
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn

class MixedPrecisionTrainer:
    """混合精度训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 enabled: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler(enabled=enabled)
        self.enabled = enabled
    
    def train_step(self, data: torch.Tensor, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """训练步骤"""
        
        self.optimizer.zero_grad()
        
        # 使用自动混合精度
        with autocast(enabled=self.enabled):
            output = self.model(data)
            loss = criterion(output, target)
        
        # 缩放损失并反向传播
        self.scaler.scale(loss).backward()
        
        # 更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def get_memory_usage(self) -> dict:
        """获取内存使用情况"""
        
        if not torch.cuda.is_available():
            return {}
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'cached': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9
        }

# 使用示例
trainer = MixedPrecisionTrainer(model, optimizer, enabled=True)

for epoch in range(num_epochs):
    for data, target in dataloader:
        loss = trainer.train_step(data, target, criterion)
        
        if batch_idx % 100 == 0:
            memory_usage = trainer.get_memory_usage()
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
            print(f"Memory: {memory_usage}")
```

### 内存映射数据集

```python
# memory_mapped_dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from typing import Tuple, Optional

class MemoryMappedDataset(Dataset):
    """内存映射数据集"""
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 100,
                 cache_size: int = 1000):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.cache_size = cache_size
        self.cache = {}
        
        # 打开HDF5文件
        self.h5_file = h5py.File(data_path, 'r')
        self.data = self.h5_file['data']
        self.targets = self.h5_file['targets']
        
        self.length = len(self.data)
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]
        
        # 从磁盘读取数据
        data = torch.from_numpy(self.data[idx]).float()
        target = torch.from_numpy(self.targets[idx]).float()
        
        # 添加到缓存（如果缓存未满）
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (data, target)
        
        return data, target
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

class DataPrefetcher:
    """数据预取器"""
    
    def __init__(self, dataloader, device: str = 'cuda'):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    def __iter__(self):
        first = True
        
        for next_data, next_target in self.dataloader:
            if not first:
                yield data, target
            else:
                first = False
            
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    next_data = next_data.to(self.device, non_blocking=True)
                    next_target = next_target.to(self.device, non_blocking=True)
            else:
                next_data = next_data.to(self.device)
                next_target = next_target.to(self.device)
            
            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)
            
            data, target = next_data, next_target
        
        yield data, target

# 使用示例
dataset = MemoryMappedDataset('large_dataset.h5')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)
prefetcher = DataPrefetcher(dataloader)

for data, target in prefetcher:
    # 训练代码
    pass
```

---

## ⚡ 计算优化

### 算子融合

```python
# operator_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FusedLinearGELU(nn.Module):
    """融合的线性层和GELU激活"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 融合线性变换和GELU激活
        return F.gelu(self.linear(x))

class FusedLayerNormLinear(nn.Module):
    """融合的LayerNorm和线性层"""
    
    def __init__(self, 
                 normalized_shape: int,
                 linear_out_features: int,
                 eps: float = 1e-5,
                 bias: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # LayerNorm参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias_ln = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        
        # Linear参数
        self.linear_weight = nn.Parameter(torch.randn(linear_out_features, normalized_shape))
        self.linear_bias = nn.Parameter(torch.zeros(linear_out_features)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 融合LayerNorm和线性变换
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.bias_ln is not None:
            normalized = normalized * self.weight + self.bias_ln
        else:
            normalized = normalized * self.weight
        
        # 线性变换
        output = F.linear(normalized, self.linear_weight, self.linear_bias)
        return output

class OptimizedAttention(nn.Module):
    """优化的注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 使用单个线性层计算Q、K、V
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.d_k ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # 一次性计算Q、K、V
        qkv = self.qkv_proj(x)  # [batch_size, seq_len, d_model * 3]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, n_heads, seq_len, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 使用Flash Attention（如果可用）
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # 标准注意力计算
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.out_proj(attn_output)

# 使用示例
# 替换标准模块
model.attention = OptimizedAttention(d_model=512, n_heads=8)
model.feed_forward[0] = FusedLinearGELU(512, 2048)
model.layer_norm = FusedLayerNormLinear(512, 512)
```

### JIT编译优化

```python
# jit_optimization.py
import torch
import torch.nn as nn
from torch.jit import script, trace
from typing import Tuple

@script
def fused_attention_forward(q: torch.Tensor, 
                           k: torch.Tensor, 
                           v: torch.Tensor,
                           scale: float) -> torch.Tensor:
    """JIT编译的注意力前向传播"""
    
    # 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 应用注意力权重
    output = torch.matmul(attn_weights, v)
    
    return output

class JITOptimizedModel(nn.Module):
    """JIT优化的模型"""
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        
        # 编译关键组件
        self._compile_components()
    
    def _compile_components(self):
        """编译模型组件"""
        
        # 示例输入
        example_input = torch.randn(1, 100, 512)
        
        # 追踪模型
        try:
            self.traced_model = trace(self.base_model, example_input)
            print("模型追踪成功")
        except Exception as e:
            print(f"模型追踪失败: {e}")
            self.traced_model = self.base_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.traced_model(x)
    
    def optimize_for_inference(self):
        """为推理优化模型"""
        
        # 冻结模型
        self.traced_model.eval()
        
        # 优化图
        if hasattr(self.traced_model, 'graph'):
            torch.jit.optimize_for_inference(self.traced_model)
        
        return self

# 使用示例
optimized_model = JITOptimizedModel(model)
optimized_model.optimize_for_inference()

# 保存优化后的模型
torch.jit.save(optimized_model.traced_model, 'optimized_model.pt')
```

---

## 🔄 数据流优化

### 异步数据加载

```python
# async_dataloader.py
import torch
import asyncio
import threading
from torch.utils.data import DataLoader
from queue import Queue
from typing import Iterator, Tuple

class AsyncDataLoader:
    """异步数据加载器"""
    
    def __init__(self, 
                 dataloader: DataLoader,
                 device: str = 'cuda',
                 queue_size: int = 2):
        self.dataloader = dataloader
        self.device = device
        self.queue_size = queue_size
        self.queue = Queue(maxsize=queue_size)
        self.thread = None
        self.stop_event = threading.Event()
    
    def _producer(self):
        """数据生产者线程"""
        
        for batch in self.dataloader:
            if self.stop_event.is_set():
                break
            
            # 将数据移动到设备
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device, non_blocking=True) for b in batch]
            else:
                batch = batch.to(self.device, non_blocking=True)
            
            self.queue.put(batch)
        
        # 添加结束标记
        self.queue.put(None)
    
    def __iter__(self) -> Iterator:
        # 启动生产者线程
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._producer)
        self.thread.start()
        
        # 消费数据
        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch
        
        # 等待线程结束
        self.thread.join()
    
    def __del__(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()

class PipelinedDataLoader:
    """流水线数据加载器"""
    
    def __init__(self, 
                 dataloader: DataLoader,
                 device: str = 'cuda',
                 prefetch_factor: int = 2):
        self.dataloader = dataloader
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    def __iter__(self):
        # 预取第一批数据
        dataiter = iter(self.dataloader)
        
        try:
            next_batch = next(dataiter)
        except StopIteration:
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                next_batch = self._to_device(next_batch)
        else:
            next_batch = self._to_device(next_batch)
        
        for batch in dataiter:
            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)
            
            current_batch = next_batch
            
            # 预取下一批数据
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    next_batch = self._to_device(batch)
            else:
                next_batch = self._to_device(batch)
            
            yield current_batch
        
        # 返回最后一批数据
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        yield next_batch
    
    def _to_device(self, batch):
        """将批次数据移动到设备"""
        
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device, non_blocking=True) for b in batch]
        else:
            return batch.to(self.device, non_blocking=True)

# 使用示例
# 标准数据加载器
standard_loader = DataLoader(dataset, batch_size=32, num_workers=4)

# 异步数据加载器
async_loader = AsyncDataLoader(standard_loader, device='cuda')

# 流水线数据加载器
pipelined_loader = PipelinedDataLoader(standard_loader, device='cuda')

# 性能对比
import time

def benchmark_dataloader(loader, name: str):
    start_time = time.time()
    for i, batch in enumerate(loader):
        if i >= 100:  # 只测试前100个批次
            break
        # 模拟处理时间
        time.sleep(0.01)
    
    end_time = time.time()
    print(f"{name}: {end_time - start_time:.2f}秒")

benchmark_dataloader(standard_loader, "标准加载器")
benchmark_dataloader(async_loader, "异步加载器")
benchmark_dataloader(pipelined_loader, "流水线加载器")
```

---

## 🖥️ GPU优化

### CUDA内核优化

```python
# cuda_kernels.py
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 自定义CUDA内核
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k, 
    const float* __restrict__ v,
    float* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int d_model,
    const float scale) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * d_model;
    
    if (idx < total_elements) {
        // 简化的注意力计算
        int batch_idx = idx / (seq_len * d_model);
        int seq_idx = (idx % (seq_len * d_model)) / d_model;
        int dim_idx = idx % d_model;
        
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            int k_idx = batch_idx * seq_len * d_model + i * d_model + dim_idx;
            int v_idx = batch_idx * seq_len * d_model + i * d_model + dim_idx;
            
            float attention_weight = expf(q[idx] * k[k_idx] * scale);
            sum += attention_weight * v[v_idx];
        }
        
        output[idx] = sum;
    }
}

torch::Tensor fused_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale) {
    
    auto output = torch::zeros_like(q);
    
    const int batch_size = q.size(0);
    const int seq_len = q.size(1);
    const int d_model = q.size(2);
    
    const int total_elements = batch_size * seq_len * d_model;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_attention_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        d_model,
        scale
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fused_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale) {
    
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    
    return fused_attention_cuda(q, k, v, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention", &fused_attention, "Fused attention forward (CUDA)");
}
"""

# 编译CUDA扩展
try:
    fused_attention_cuda = load_inline(
        name='fused_attention_cuda',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['fused_attention'],
        verbose=True
    )
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"CUDA扩展编译失败: {e}")
    CUDA_AVAILABLE = False

class CUDAOptimizedAttention(nn.Module):
    """CUDA优化的注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if CUDA_AVAILABLE and x.is_cuda:
            # 使用自定义CUDA内核
            output = fused_attention_cuda.fused_attention(q, k, v, self.scale)
        else:
            # 回退到标准实现
            q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)
            
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(output)
```

### GPU内存管理

```python
# gpu_memory_manager.py
import torch
import gc
from typing import Dict, List, Optional

class GPUMemoryManager:
    """GPU内存管理器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.memory_pool = {}
        self.peak_memory = 0
        
    def get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'free': reserved - allocated
        }
    
    def clear_cache(self):
        """清理缓存"""
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def optimize_memory_usage(self, model: torch.nn.Module):
        """优化内存使用"""
        
        # 启用内存高效的注意力
        if hasattr(model, 'enable_memory_efficient_attention'):
            model.enable_memory_efficient_attention()
        
        # 使用检查点
        if hasattr(torch.utils.checkpoint, 'checkpoint'):
            for module in model.modules():
                if hasattr(module, 'use_checkpoint'):
                    module.use_checkpoint = True
    
    def monitor_memory(self, operation_name: str):
        """内存监控装饰器"""
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 记录操作前内存
                before = self.get_memory_info()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # 记录操作后内存
                    after = self.get_memory_info()
                    
                    memory_delta = after['allocated'] - before['allocated']
                    print(f"{operation_name}: 内存变化 {memory_delta:.2f}GB")
                    
                    return result
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"{operation_name}: GPU内存不足")
                    print(f"当前内存使用: {before}")
                    self.clear_cache()
                    raise e
            
            return wrapper
        return decorator
    
    def adaptive_batch_size(self, 
                          model: torch.nn.Module,
                          sample_input: torch.Tensor,
                          max_batch_size: int = 128,
                          memory_limit: float = 0.9) -> int:
        """自适应批次大小"""
        
        model.eval()
        optimal_batch_size = 1
        
        for batch_size in [2**i for i in range(int(torch.log2(torch.tensor(max_batch_size)).item()) + 1)]:
            try:
                # 创建测试批次
                test_input = sample_input.repeat(batch_size, 1, 1)
                
                # 测试前向传播
                with torch.no_grad():
                    _ = model(test_input)
                
                # 检查内存使用
                memory_info = self.get_memory_info()
                memory_usage = memory_info['allocated'] / memory_info['reserved']
                
                if memory_usage < memory_limit:
                    optimal_batch_size = batch_size
                else:
                    break
                    
                # 清理
                del test_input
                self.clear_cache()
                
            except torch.cuda.OutOfMemoryError:
                break
        
        model.train()
        return optimal_batch_size

# 使用示例
memory_manager = GPUMemoryManager()

# 监控内存使用
@memory_manager.monitor_memory("模型前向传播")
def forward_pass(model, data):
    return model(data)

# 自适应批次大小
sample_input = torch.randn(1, 100, 512).cuda()
optimal_batch_size = memory_manager.adaptive_batch_size(model, sample_input)
print(f"推荐批次大小: {optimal_batch_size}")

# 内存优化
memory_manager.optimize_memory_usage(model)
```

---

*本性能优化指南提供了全面的优化策略和实用工具。更多详细信息请参考[完整文档](/)。*