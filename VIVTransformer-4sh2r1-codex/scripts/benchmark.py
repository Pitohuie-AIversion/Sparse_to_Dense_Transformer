#!/usr/bin/env python3
"""性能基准测试脚本

用于测试VIVTransformer模型的性能指标，包括速度、内存使用、准确性等
"""

import argparse
import time
import json
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent))

from modify_multi_attention.attention_test import (
    StandardAttention, MultiHeadAttention, 
    MultiLossFunction, SVDLoss, SparsityLoss, ReconstructionLoss
)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    duration: float
    memory_peak: float
    memory_allocated: float
    gpu_memory_peak: float
    throughput: float
    accuracy: Optional[float] = None
    additional_metrics: Optional[Dict] = None


class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.results = []
        print(f"Using device: {self.device}")
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 预热GPU
        if self.device.type == 'cuda':
            self._warmup_gpu()
    
    def _warmup_gpu(self):
        """GPU预热"""
        print("Warming up GPU...")
        dummy_tensor = torch.randn(1000, 1000, device=self.device)
        for _ in range(10):
            _ = torch.mm(dummy_tensor, dummy_tensor)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    def _get_memory_usage(self) -> Tuple[float, float]:
        """获取内存使用情况"""
        # CPU内存
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU内存
        gpu_memory = 0.0
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated(self.device) / 1024 / 1024  # MB
        
        return cpu_memory, gpu_memory
    
    def _clear_memory(self):
        """清理内存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def benchmark_attention_forward(self, batch_sizes: List[int], seq_lengths: List[int], 
                                  embed_dims: List[int]) -> List[BenchmarkResult]:
        """基准测试注意力机制前向传播"""
        print("\n=== Benchmarking Attention Forward Pass ===")
        results = []
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                for embed_dim in embed_dims:
                    print(f"Testing: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")
                    
                    # 创建测试数据
                    x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
                    
                    # 测试StandardAttention
                    attention = StandardAttention(embed_dim).to(self.device)
                    
                    # 记录开始状态
                    self._clear_memory()
                    start_memory_cpu, start_memory_gpu = self._get_memory_usage()
                    
                    # 性能测试
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    start_time = time.time()
                    
                    # 多次运行取平均
                    num_runs = 100
                    for _ in range(num_runs):
                        with torch.no_grad():
                            output = attention(x)
                    
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end_time = time.time()
                    
                    # 记录结束状态
                    end_memory_cpu, end_memory_gpu = self._get_memory_usage()
                    
                    duration = (end_time - start_time) / num_runs
                    memory_used = end_memory_cpu - start_memory_cpu
                    gpu_memory_used = end_memory_gpu - start_memory_gpu
                    
                    # 计算吞吐量 (samples per second)
                    throughput = batch_size / duration
                    
                    result = BenchmarkResult(
                        test_name=f"StandardAttention_B{batch_size}_S{seq_len}_E{embed_dim}",
                        duration=duration,
                        memory_peak=memory_used,
                        memory_allocated=end_memory_cpu,
                        gpu_memory_peak=gpu_memory_used,
                        throughput=throughput,
                        additional_metrics={
                            'batch_size': batch_size,
                            'seq_length': seq_len,
                            'embed_dim': embed_dim,
                            'num_runs': num_runs
                        }
                    )
                    
                    results.append(result)
                    self.results.append(result)
                    
                    print(f"  Duration: {duration*1000:.2f}ms, Throughput: {throughput:.1f} samples/s")
        
        return results
    
    def benchmark_attention_backward(self, batch_sizes: List[int], seq_lengths: List[int], 
                                   embed_dims: List[int]) -> List[BenchmarkResult]:
        """基准测试注意力机制反向传播"""
        print("\n=== Benchmarking Attention Backward Pass ===")
        results = []
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                for embed_dim in embed_dims:
                    print(f"Testing: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}")
                    
                    # 创建测试数据
                    x = torch.randn(batch_size, seq_len, embed_dim, device=self.device, requires_grad=True)
                    target = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
                    
                    # 测试StandardAttention
                    attention = StandardAttention(embed_dim).to(self.device)
                    criterion = torch.nn.MSELoss()
                    
                    # 记录开始状态
                    self._clear_memory()
                    start_memory_cpu, start_memory_gpu = self._get_memory_usage()
                    
                    # 性能测试
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    start_time = time.time()
                    
                    # 多次运行取平均
                    num_runs = 50
                    for _ in range(num_runs):
                        # 前向传播
                        output = attention(x)
                        loss = criterion(output, target)
                        
                        # 反向传播
                        loss.backward(retain_graph=True)
                        
                        # 清理梯度
                        attention.zero_grad()
                        if x.grad is not None:
                            x.grad.zero_()
                    
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end_time = time.time()
                    
                    # 记录结束状态
                    end_memory_cpu, end_memory_gpu = self._get_memory_usage()
                    
                    duration = (end_time - start_time) / num_runs
                    memory_used = end_memory_cpu - start_memory_cpu
                    gpu_memory_used = end_memory_gpu - start_memory_gpu
                    
                    # 计算吞吐量
                    throughput = batch_size / duration
                    
                    result = BenchmarkResult(
                        test_name=f"StandardAttention_Backward_B{batch_size}_S{seq_len}_E{embed_dim}",
                        duration=duration,
                        memory_peak=memory_used,
                        memory_allocated=end_memory_cpu,
                        gpu_memory_peak=gpu_memory_used,
                        throughput=throughput,
                        additional_metrics={
                            'batch_size': batch_size,
                            'seq_length': seq_len,
                            'embed_dim': embed_dim,
                            'num_runs': num_runs
                        }
                    )
                    
                    results.append(result)
                    self.results.append(result)
                    
                    print(f"  Duration: {duration*1000:.2f}ms, Throughput: {throughput:.1f} samples/s")
        
        return results
    
    def benchmark_multihead_attention(self, batch_sizes: List[int], seq_lengths: List[int], 
                                    embed_dims: List[int], num_heads_list: List[int]) -> List[BenchmarkResult]:
        """基准测试多头注意力机制"""
        print("\n=== Benchmarking MultiHead Attention ===")
        results = []
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                for embed_dim in embed_dims:
                    for num_heads in num_heads_list:
                        if embed_dim % num_heads != 0:
                            continue
                            
                        print(f"Testing: batch_size={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, heads={num_heads}")
                        
                        # 创建测试数据
                        x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
                        
                        # 测试MultiHeadAttention
                        attention = MultiHeadAttention(embed_dim, num_heads).to(self.device)
                        
                        # 记录开始状态
                        self._clear_memory()
                        start_memory_cpu, start_memory_gpu = self._get_memory_usage()
                        
                        # 性能测试
                        torch.cuda.synchronize() if self.device.type == 'cuda' else None
                        start_time = time.time()
                        
                        # 多次运行取平均
                        num_runs = 50
                        for _ in range(num_runs):
                            with torch.no_grad():
                                output = attention(x)
                        
                        torch.cuda.synchronize() if self.device.type == 'cuda' else None
                        end_time = time.time()
                        
                        # 记录结束状态
                        end_memory_cpu, end_memory_gpu = self._get_memory_usage()
                        
                        duration = (end_time - start_time) / num_runs
                        memory_used = end_memory_cpu - start_memory_cpu
                        gpu_memory_used = end_memory_gpu - start_memory_gpu
                        
                        # 计算吞吐量
                        throughput = batch_size / duration
                        
                        result = BenchmarkResult(
                            test_name=f"MultiHeadAttention_B{batch_size}_S{seq_len}_E{embed_dim}_H{num_heads}",
                            duration=duration,
                            memory_peak=memory_used,
                            memory_allocated=end_memory_cpu,
                            gpu_memory_peak=gpu_memory_used,
                            throughput=throughput,
                            additional_metrics={
                                'batch_size': batch_size,
                                'seq_length': seq_len,
                                'embed_dim': embed_dim,
                                'num_heads': num_heads,
                                'num_runs': num_runs
                            }
                        )
                        
                        results.append(result)
                        self.results.append(result)
                        
                        print(f"  Duration: {duration*1000:.2f}ms, Throughput: {throughput:.1f} samples/s")
        
        return results
    
    def benchmark_loss_functions(self, batch_sizes: List[int], feature_dims: List[int]) -> List[BenchmarkResult]:
        """基准测试损失函数"""
        print("\n=== Benchmarking Loss Functions ===")
        results = []
        
        loss_functions = {
            'SVDLoss': SVDLoss(),
            'SparsityLoss': SparsityLoss(),
            'ReconstructionLoss': ReconstructionLoss(),
            'MultiLossFunction': MultiLossFunction()
        }
        
        for loss_name, loss_fn in loss_functions.items():
            loss_fn = loss_fn.to(self.device)
            
            for batch_size in batch_sizes:
                for feature_dim in feature_dims:
                    print(f"Testing {loss_name}: batch_size={batch_size}, feature_dim={feature_dim}")
                    
                    # 创建测试数据
                    if loss_name == 'MultiLossFunction':
                        pred = torch.randn(batch_size, feature_dim, device=self.device)
                        target = torch.randn(batch_size, feature_dim, device=self.device)
                        attention_weights = torch.randn(batch_size, feature_dim, feature_dim, device=self.device)
                        data = {'pred': pred, 'target': target, 'attention_weights': attention_weights}
                    else:
                        data = torch.randn(batch_size, feature_dim, feature_dim, device=self.device)
                    
                    # 记录开始状态
                    self._clear_memory()
                    start_memory_cpu, start_memory_gpu = self._get_memory_usage()
                    
                    # 性能测试
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    start_time = time.time()
                    
                    # 多次运行取平均
                    num_runs = 100
                    for _ in range(num_runs):
                        if loss_name == 'MultiLossFunction':
                            loss = loss_fn(**data)
                        else:
                            loss = loss_fn(data)
                    
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end_time = time.time()
                    
                    # 记录结束状态
                    end_memory_cpu, end_memory_gpu = self._get_memory_usage()
                    
                    duration = (end_time - start_time) / num_runs
                    memory_used = end_memory_cpu - start_memory_cpu
                    gpu_memory_used = end_memory_gpu - start_memory_gpu
                    
                    # 计算吞吐量
                    throughput = batch_size / duration
                    
                    result = BenchmarkResult(
                        test_name=f"{loss_name}_B{batch_size}_F{feature_dim}",
                        duration=duration,
                        memory_peak=memory_used,
                        memory_allocated=end_memory_cpu,
                        gpu_memory_peak=gpu_memory_used,
                        throughput=throughput,
                        additional_metrics={
                            'batch_size': batch_size,
                            'feature_dim': feature_dim,
                            'num_runs': num_runs
                        }
                    )
                    
                    results.append(result)
                    self.results.append(result)
                    
                    print(f"  Duration: {duration*1000:.2f}ms, Throughput: {throughput:.1f} samples/s")
        
        return results
    
    def benchmark_memory_scaling(self, max_batch_size: int = 64, max_seq_length: int = 512) -> List[BenchmarkResult]:
        """基准测试内存扩展性"""
        print("\n=== Benchmarking Memory Scaling ===")
        results = []
        
        embed_dim = 256
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        seq_lengths = [64, 128, 256, 512]
        
        # 过滤超出限制的配置
        batch_sizes = [b for b in batch_sizes if b <= max_batch_size]
        seq_lengths = [s for s in seq_lengths if s <= max_seq_length]
        
        attention = StandardAttention(embed_dim).to(self.device)
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"Testing memory scaling: batch_size={batch_size}, seq_len={seq_len}")
                
                try:
                    # 创建测试数据
                    x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
                    
                    # 记录开始状态
                    self._clear_memory()
                    start_memory_cpu, start_memory_gpu = self._get_memory_usage()
                    
                    # 前向传播
                    with torch.no_grad():
                        output = attention(x)
                    
                    # 记录峰值内存
                    peak_memory_cpu, peak_memory_gpu = self._get_memory_usage()
                    
                    memory_used = peak_memory_cpu - start_memory_cpu
                    gpu_memory_used = peak_memory_gpu - start_memory_gpu
                    
                    result = BenchmarkResult(
                        test_name=f"MemoryScaling_B{batch_size}_S{seq_len}",
                        duration=0.0,  # 不关心时间
                        memory_peak=memory_used,
                        memory_allocated=peak_memory_cpu,
                        gpu_memory_peak=gpu_memory_used,
                        throughput=0.0,  # 不关心吞吐量
                        additional_metrics={
                            'batch_size': batch_size,
                            'seq_length': seq_len,
                            'embed_dim': embed_dim,
                            'total_elements': batch_size * seq_len * embed_dim
                        }
                    )
                    
                    results.append(result)
                    self.results.append(result)
                    
                    print(f"  Memory used: CPU={memory_used:.1f}MB, GPU={gpu_memory_used:.1f}MB")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  OOM at batch_size={batch_size}, seq_len={seq_len}")
                        break
                    else:
                        raise e
        
        return results
    
    def generate_report(self, output_dir: Path):
        """生成性能报告"""
        print("\n=== Generating Performance Report ===")
        
        output_dir.mkdir(exist_ok=True)
        
        # 保存原始数据
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'total_tests': len(self.results),
            'results': [{
                'test_name': r.test_name,
                'duration': r.duration,
                'memory_peak': r.memory_peak,
                'memory_allocated': r.memory_allocated,
                'gpu_memory_peak': r.gpu_memory_peak,
                'throughput': r.throughput,
                'accuracy': r.accuracy,
                'additional_metrics': r.additional_metrics
            } for r in self.results]
        }
        
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # 生成可视化图表
        self._generate_plots(output_dir)
        
        # 生成文本报告
        self._generate_text_report(output_dir)
        
        print(f"Report generated in: {output_dir}")
    
    def _generate_plots(self, output_dir: Path):
        """生成可视化图表"""
        plt.style.use('seaborn-v0_8')
        
        # 1. 吞吐量对比图
        attention_results = [r for r in self.results if 'Attention' in r.test_name and r.throughput > 0]
        if attention_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            test_names = [r.test_name.split('_')[0] for r in attention_results]
            throughputs = [r.throughput for r in attention_results]
            
            ax.bar(range(len(test_names)), throughputs)
            ax.set_xlabel('Test Configuration')
            ax.set_ylabel('Throughput (samples/sec)')
            ax.set_title('Attention Mechanism Throughput Comparison')
            ax.set_xticks(range(len(test_names)))
            ax.set_xticklabels(test_names, rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 内存使用图
        memory_results = [r for r in self.results if 'MemoryScaling' in r.test_name]
        if memory_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            batch_sizes = [r.additional_metrics['batch_size'] for r in memory_results]
            cpu_memory = [r.memory_peak for r in memory_results]
            gpu_memory = [r.gpu_memory_peak for r in memory_results]
            
            ax1.plot(batch_sizes, cpu_memory, 'o-', label='CPU Memory')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Memory Usage (MB)')
            ax1.set_title('CPU Memory Scaling')
            ax1.legend()
            
            ax2.plot(batch_sizes, gpu_memory, 'o-', label='GPU Memory', color='orange')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('GPU Memory Scaling')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / 'memory_scaling.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 执行时间热力图
        attention_forward = [r for r in self.results if 'StandardAttention_B' in r.test_name and 'Backward' not in r.test_name]
        if attention_forward:
            # 创建热力图数据
            batch_sizes = sorted(list(set([r.additional_metrics['batch_size'] for r in attention_forward])))
            seq_lengths = sorted(list(set([r.additional_metrics['seq_length'] for r in attention_forward])))
            
            heatmap_data = np.zeros((len(batch_sizes), len(seq_lengths)))
            
            for r in attention_forward:
                b_idx = batch_sizes.index(r.additional_metrics['batch_size'])
                s_idx = seq_lengths.index(r.additional_metrics['seq_length'])
                heatmap_data[b_idx, s_idx] = r.duration * 1000  # 转换为毫秒
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_data, 
                       xticklabels=seq_lengths, 
                       yticklabels=batch_sizes,
                       annot=True, 
                       fmt='.2f',
                       cmap='YlOrRd',
                       ax=ax)
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Batch Size')
            ax.set_title('Attention Forward Pass Duration (ms)')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'duration_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_text_report(self, output_dir: Path):
        """生成文本报告"""
        with open(output_dir / 'benchmark_summary.txt', 'w') as f:
            f.write("VIVTransformer Performance Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Total Tests: {len(self.results)}\n\n")
            
            # 按类别分组结果
            categories = {}
            for result in self.results:
                category = result.test_name.split('_')[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)
            
            for category, results in categories.items():
                f.write(f"{category} Results:\n")
                f.write("-" * 30 + "\n")
                
                for result in results:
                    f.write(f"  {result.test_name}:\n")
                    f.write(f"    Duration: {result.duration*1000:.2f}ms\n")
                    f.write(f"    Throughput: {result.throughput:.1f} samples/s\n")
                    f.write(f"    Memory Peak: {result.memory_peak:.1f}MB\n")
                    if result.gpu_memory_peak > 0:
                        f.write(f"    GPU Memory Peak: {result.gpu_memory_peak:.1f}MB\n")
                    f.write("\n")
                
                f.write("\n")
            
            # 性能总结
            f.write("Performance Summary:\n")
            f.write("-" * 30 + "\n")
            
            if self.results:
                avg_duration = np.mean([r.duration for r in self.results if r.duration > 0])
                avg_throughput = np.mean([r.throughput for r in self.results if r.throughput > 0])
                max_memory = max([r.memory_peak for r in self.results])
                
                f.write(f"Average Duration: {avg_duration*1000:.2f}ms\n")
                f.write(f"Average Throughput: {avg_throughput:.1f} samples/s\n")
                f.write(f"Peak Memory Usage: {max_memory:.1f}MB\n")
                
                if self.device.type == 'cuda':
                    max_gpu_memory = max([r.gpu_memory_peak for r in self.results])
                    f.write(f"Peak GPU Memory Usage: {max_gpu_memory:.1f}MB\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run benchmarks on'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='benchmark_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark with smaller configurations'
    )
    
    parser.add_argument(
        '--tests',
        nargs='+',
        choices=['attention_forward', 'attention_backward', 'multihead', 'loss', 'memory'],
        default=['attention_forward', 'multihead', 'loss', 'memory'],
        help='Specific tests to run'
    )
    
    args = parser.parse_args()
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark(args.device)
    
    # 配置测试参数
    if args.quick:
        batch_sizes = [1, 4, 16]
        seq_lengths = [64, 128]
        embed_dims = [128, 256]
        feature_dims = [64, 128]
        num_heads_list = [4, 8]
    else:
        batch_sizes = [1, 2, 4, 8, 16, 32]
        seq_lengths = [64, 128, 256, 512]
        embed_dims = [128, 256, 512]
        feature_dims = [64, 128, 256, 512]
        num_heads_list = [4, 8, 16]
    
    print(f"Starting benchmark with device: {benchmark.device}")
    print(f"Quick mode: {args.quick}")
    print(f"Tests to run: {args.tests}")
    
    try:
        # 运行选定的测试
        if 'attention_forward' in args.tests:
            benchmark.benchmark_attention_forward(batch_sizes, seq_lengths, embed_dims)
        
        if 'attention_backward' in args.tests:
            benchmark.benchmark_attention_backward(batch_sizes, seq_lengths, embed_dims)
        
        if 'multihead' in args.tests:
            benchmark.benchmark_multihead_attention(batch_sizes, seq_lengths, embed_dims, num_heads_list)
        
        if 'loss' in args.tests:
            benchmark.benchmark_loss_functions(batch_sizes, feature_dims)
        
        if 'memory' in args.tests:
            max_batch = max(batch_sizes) if batch_sizes else 32
            max_seq = max(seq_lengths) if seq_lengths else 256
            benchmark.benchmark_memory_scaling(max_batch, max_seq)
        
        # 生成报告
        benchmark.generate_report(args.output_dir)
        
        print("\n🎉 Benchmark completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()