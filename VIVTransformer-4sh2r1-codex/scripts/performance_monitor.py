#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIVTransformer 性能监控和分析工具

功能:
- 代码性能分析和瓶颈检测
- 内存使用监控和泄漏检测
- 性能优化建议生成
- 性能基准测试
- 资源使用统计

作者: VIVTransformer Team
日期: 2024
"""

import os
import sys
import ast
import time
import json
import psutil
import logging
import argparse
import tracemalloc
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, deque
import cProfile
import pstats
import io
import gc
import weakref

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    memory_peak: float
    execution_time: float
    function_calls: int
    io_operations: int
    gc_collections: int
    thread_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryLeak:
    """内存泄漏信息"""
    object_type: str
    count: int
    size_bytes: int
    growth_rate: float
    location: str
    severity: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceBottleneck:
    """性能瓶颈信息"""
    function_name: str
    file_path: str
    line_number: int
    execution_time: float
    call_count: int
    time_per_call: float
    cumulative_time: float
    severity: str
    optimization_suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history = deque(maxlen=1000)
        self.monitor_thread = None
        self.process = psutil.Process()
        
    def start_monitoring(self, interval: float = 1.0):
        """开始监控系统资源"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("系统资源监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("系统资源监控已停止")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"监控过程中出错: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        memory_info = self.process.memory_info()
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_usage=self.process.cpu_percent(),
            memory_usage=memory_info.rss / 1024 / 1024,  # MB
            memory_peak=memory_info.peak_wss / 1024 / 1024 if hasattr(memory_info, 'peak_wss') else 0,
            execution_time=0.0,  # 将在具体分析中设置
            function_calls=0,
            io_operations=self.process.io_counters().read_count + self.process.io_counters().write_count,
            gc_collections=sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats()))),
            thread_count=self.process.num_threads()
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m.cpu_usage for m in self.metrics_history]
        memory_values = [m.memory_usage for m in self.metrics_history]
        
        return {
            'monitoring_duration': len(self.metrics_history),
            'cpu_usage': {
                'average': sum(cpu_values) / len(cpu_values),
                'peak': max(cpu_values),
                'minimum': min(cpu_values)
            },
            'memory_usage': {
                'average': sum(memory_values) / len(memory_values),
                'peak': max(memory_values),
                'minimum': min(memory_values)
            },
            'total_samples': len(self.metrics_history)
        }


class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self):
        self.snapshots = []
        self.tracking_enabled = False
        
    def start_tracking(self):
        """开始内存跟踪"""
        tracemalloc.start()
        self.tracking_enabled = True
        logger.info("内存跟踪已启动")
    
    def stop_tracking(self):
        """停止内存跟踪"""
        if self.tracking_enabled:
            tracemalloc.stop()
            self.tracking_enabled = False
            logger.info("内存跟踪已停止")
    
    def take_snapshot(self, label: str = None) -> int:
        """拍摄内存快照"""
        if not self.tracking_enabled:
            self.start_tracking()
        
        snapshot = tracemalloc.take_snapshot()
        snapshot_info = {
            'snapshot': snapshot,
            'timestamp': datetime.now().isoformat(),
            'label': label or f"snapshot_{len(self.snapshots)}"
        }
        self.snapshots.append(snapshot_info)
        
        logger.info(f"内存快照已保存: {snapshot_info['label']}")
        return len(self.snapshots) - 1
    
    def detect_leaks(self, baseline_index: int = 0, current_index: int = -1) -> List[MemoryLeak]:
        """检测内存泄漏"""
        if len(self.snapshots) < 2:
            logger.warning("需要至少两个快照来检测内存泄漏")
            return []
        
        baseline = self.snapshots[baseline_index]['snapshot']
        current = self.snapshots[current_index]['snapshot']
        
        # 比较快照
        top_stats = current.compare_to(baseline, 'lineno')
        
        leaks = []
        for stat in top_stats[:20]:  # 只检查前20个最大的差异
            if stat.size_diff > 1024 * 1024:  # 大于1MB的增长
                severity = self._calculate_leak_severity(stat.size_diff, stat.count_diff)
                
                leak = MemoryLeak(
                    object_type=str(stat.traceback.format()[-1]) if stat.traceback else "Unknown",
                    count=stat.count_diff,
                    size_bytes=stat.size_diff,
                    growth_rate=stat.size_diff / (stat.size or 1),
                    location=str(stat.traceback.format()[0]) if stat.traceback else "Unknown",
                    severity=severity,
                    description=f"内存增长 {stat.size_diff / 1024 / 1024:.2f}MB, 对象数量增加 {stat.count_diff}"
                )
                leaks.append(leak)
        
        return leaks
    
    def _calculate_leak_severity(self, size_diff: int, count_diff: int) -> str:
        """计算泄漏严重程度"""
        size_mb = size_diff / 1024 / 1024
        
        if size_mb > 100 or count_diff > 10000:
            return "critical"
        elif size_mb > 50 or count_diff > 5000:
            return "high"
        elif size_mb > 10 or count_diff > 1000:
            return "medium"
        else:
            return "low"
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        if not self.snapshots:
            return {}
        
        current_snapshot = self.snapshots[-1]['snapshot']
        top_stats = current_snapshot.statistics('lineno')
        
        total_size = sum(stat.size for stat in top_stats)
        total_count = sum(stat.count for stat in top_stats)
        
        return {
            'total_memory_usage': total_size,
            'total_objects': total_count,
            'top_memory_consumers': [
                {
                    'location': str(stat.traceback.format()[0]) if stat.traceback else "Unknown",
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                }
                for stat in top_stats[:10]
            ],
            'snapshots_count': len(self.snapshots)
        }


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiler = None
        self.profiling_active = False
        
    def start_profiling(self):
        """开始性能分析"""
        if self.profiling_active:
            return
            
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.profiling_active = True
        logger.info("性能分析已启动")
    
    def stop_profiling(self) -> pstats.Stats:
        """停止性能分析并返回统计信息"""
        if not self.profiling_active:
            return None
            
        self.profiler.disable()
        self.profiling_active = False
        
        # 创建统计对象
        stats_stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)
        
        logger.info("性能分析已停止")
        return stats
    
    def analyze_performance(self, stats: pstats.Stats) -> List[PerformanceBottleneck]:
        """分析性能瓶颈"""
        if not stats:
            return []
        
        # 按累计时间排序
        stats.sort_stats('cumulative')
        
        bottlenecks = []
        
        # 获取统计信息
        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line_number, function_name = func_info
            
            # 过滤掉系统函数和库函数
            if self._should_analyze_function(filename, function_name):
                time_per_call = tt / cc if cc > 0 else 0
                
                severity = self._calculate_bottleneck_severity(ct, cc, time_per_call)
                suggestions = self._generate_optimization_suggestions(ct, cc, time_per_call, function_name)
                
                bottleneck = PerformanceBottleneck(
                    function_name=function_name,
                    file_path=filename,
                    line_number=line_number,
                    execution_time=tt,
                    call_count=cc,
                    time_per_call=time_per_call,
                    cumulative_time=ct,
                    severity=severity,
                    optimization_suggestions=suggestions
                )
                
                bottlenecks.append(bottleneck)
        
        # 按累计时间排序，返回前20个
        return sorted(bottlenecks, key=lambda x: x.cumulative_time, reverse=True)[:20]
    
    def _should_analyze_function(self, filename: str, function_name: str) -> bool:
        """判断是否应该分析该函数"""
        # 过滤系统文件和库文件
        if any(path in filename for path in ['<built-in>', '<frozen>', 'site-packages']):
            return False
        
        # 过滤特殊函数
        if function_name.startswith('_') and not function_name.startswith('__'):
            return False
            
        return True
    
    def _calculate_bottleneck_severity(self, cumulative_time: float, call_count: int, time_per_call: float) -> str:
        """计算瓶颈严重程度"""
        if cumulative_time > 1.0 or time_per_call > 0.1:
            return "critical"
        elif cumulative_time > 0.5 or time_per_call > 0.05:
            return "high"
        elif cumulative_time > 0.1 or time_per_call > 0.01:
            return "medium"
        else:
            return "low"
    
    def _generate_optimization_suggestions(self, cumulative_time: float, call_count: int, 
                                         time_per_call: float, function_name: str) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        if call_count > 1000:
            suggestions.append("考虑缓存函数结果以减少重复计算")
            suggestions.append("检查是否存在不必要的重复调用")
        
        if time_per_call > 0.1:
            suggestions.append("函数执行时间较长，考虑算法优化")
            suggestions.append("检查是否有阻塞操作可以异步化")
        
        if cumulative_time > 1.0:
            suggestions.append("该函数是主要性能瓶颈，优先优化")
        
        if 'loop' in function_name.lower() or 'iter' in function_name.lower():
            suggestions.append("优化循环逻辑，考虑向量化操作")
        
        if 'io' in function_name.lower() or 'read' in function_name.lower() or 'write' in function_name.lower():
            suggestions.append("考虑批量I/O操作以提高效率")
            suggestions.append("使用异步I/O减少阻塞时间")
        
        return suggestions or ["监控函数性能变化"]


class PerformanceMonitor:
    """性能监控主控制器"""
    
    def __init__(self):
        self.resource_monitor = SystemResourceMonitor()
        self.memory_detector = MemoryLeakDetector()
        self.profiler = PerformanceProfiler()
        
    def start_comprehensive_monitoring(self):
        """开始综合监控"""
        self.resource_monitor.start_monitoring()
        self.memory_detector.start_tracking()
        self.profiler.start_profiling()
        
        # 拍摄初始内存快照
        self.memory_detector.take_snapshot("baseline")
        
        logger.info("综合性能监控已启动")
    
    def stop_comprehensive_monitoring(self) -> Dict[str, Any]:
        """停止综合监控并生成报告"""
        # 拍摄最终内存快照
        self.memory_detector.take_snapshot("final")
        
        # 停止各种监控
        stats = self.profiler.stop_profiling()
        self.resource_monitor.stop_monitoring()
        
        # 分析结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'resource_summary': self.resource_monitor.get_metrics_summary(),
            'memory_statistics': self.memory_detector.get_memory_statistics(),
            'memory_leaks': [leak.to_dict() for leak in self.memory_detector.detect_leaks()],
            'performance_bottlenecks': [b.to_dict() for b in self.profiler.analyze_performance(stats)],
            'recommendations': self._generate_performance_recommendations()
        }
        
        self.memory_detector.stop_tracking()
        logger.info("综合性能监控已停止")
        
        return results
    
    def _generate_performance_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = [
            "定期监控应用性能指标",
            "使用性能分析工具识别瓶颈",
            "优化数据库查询和I/O操作",
            "考虑使用缓存减少重复计算",
            "监控内存使用避免泄漏",
            "使用异步编程提高并发性能",
            "定期进行性能基准测试",
            "优化算法复杂度",
            "合理使用多线程和多进程",
            "监控第三方库的性能影响"
        ]
        return recommendations
    
    def analyze_code_performance(self, target_path: str) -> Dict[str, Any]:
        """分析代码性能"""
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"路径不存在: {target_path}")
        
        logger.info(f"开始分析代码性能: {target_path}")
        
        # 开始监控
        self.start_comprehensive_monitoring()
        
        try:
            # 如果是Python文件，尝试执行分析
            if target_path.endswith('.py'):
                self._analyze_python_file(target_path)
            else:
                # 分析目录中的所有Python文件
                self._analyze_python_directory(target_path)
            
            # 等待一段时间收集数据
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"性能分析过程中出错: {e}")
        
        # 停止监控并获取结果
        results = self.stop_comprehensive_monitoring()
        
        logger.info("代码性能分析完成")
        return results
    
    def _analyze_python_file(self, file_path: str):
        """分析单个Python文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 解析AST进行静态分析
            tree = ast.parse(source_code)
            
            # 分析函数复杂度
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_function_complexity(node)
                    if complexity > 10:
                        logger.warning(f"函数 {node.name} 复杂度较高: {complexity}")
        
        except Exception as e:
            logger.error(f"分析文件 {file_path} 时出错: {e}")
    
    def _analyze_python_directory(self, dir_path: str):
        """分析目录中的Python文件"""
        python_files = list(Path(dir_path).rglob('*.py'))
        
        for file_path in python_files[:10]:  # 限制分析文件数量
            self._analyze_python_file(str(file_path))
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数复杂度"""
        complexity = 1  # 基础复杂度
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """生成性能报告"""
        report_lines = []
        
        # 报告标题
        report_lines.append("# 性能监控分析报告")
        report_lines.append(f"\n**生成时间**: {results['timestamp']}")
        
        # 资源使用摘要
        resource_summary = results.get('resource_summary', {})
        if resource_summary:
            report_lines.append("\n## 🖥️ 系统资源使用")
            
            cpu_info = resource_summary.get('cpu_usage', {})
            if cpu_info:
                report_lines.append(f"\n**CPU使用率**:")
                report_lines.append(f"- 平均: {cpu_info.get('average', 0):.2f}%")
                report_lines.append(f"- 峰值: {cpu_info.get('peak', 0):.2f}%")
            
            memory_info = resource_summary.get('memory_usage', {})
            if memory_info:
                report_lines.append(f"\n**内存使用**:")
                report_lines.append(f"- 平均: {memory_info.get('average', 0):.2f}MB")
                report_lines.append(f"- 峰值: {memory_info.get('peak', 0):.2f}MB")
        
        # 内存泄漏检测
        memory_leaks = results.get('memory_leaks', [])
        if memory_leaks:
            report_lines.append("\n## 🚨 内存泄漏检测")
            
            critical_leaks = [leak for leak in memory_leaks if leak['severity'] == 'critical']
            if critical_leaks:
                report_lines.append(f"\n**严重泄漏**: {len(critical_leaks)}")
                for leak in critical_leaks[:5]:
                    report_lines.append(f"- {leak['description']}")
                    report_lines.append(f"  位置: {leak['location']}")
        
        # 性能瓶颈
        bottlenecks = results.get('performance_bottlenecks', [])
        if bottlenecks:
            report_lines.append("\n## ⚡ 性能瓶颈分析")
            
            critical_bottlenecks = [b for b in bottlenecks if b['severity'] in ['critical', 'high']]
            if critical_bottlenecks:
                report_lines.append(f"\n**关键瓶颈**: {len(critical_bottlenecks)}")
                
                for bottleneck in critical_bottlenecks[:5]:
                    report_lines.append(f"\n### {bottleneck['function_name']}")
                    report_lines.append(f"- **文件**: {bottleneck['file_path']}:{bottleneck['line_number']}")
                    report_lines.append(f"- **累计时间**: {bottleneck['cumulative_time']:.4f}s")
                    report_lines.append(f"- **调用次数**: {bottleneck['call_count']}")
                    report_lines.append(f"- **平均耗时**: {bottleneck['time_per_call']:.6f}s")
                    
                    suggestions = bottleneck.get('optimization_suggestions', [])
                    if suggestions:
                        report_lines.append("- **优化建议**:")
                        for suggestion in suggestions[:3]:
                            report_lines.append(f"  - {suggestion}")
        
        # 优化建议
        recommendations = results.get('recommendations', [])
        if recommendations:
            report_lines.append("\n## 💡 性能优化建议")
            
            for i, rec in enumerate(recommendations[:8], 1):
                report_lines.append(f"\n{i}. {rec}")
        
        # 下一步行动
        report_lines.append("\n## 🎯 下一步行动")
        
        if memory_leaks:
            report_lines.append("\n1. **内存优化**: 修复检测到的内存泄漏")
        
        if bottlenecks:
            report_lines.append("\n2. **性能优化**: 优化关键性能瓶颈")
        
        report_lines.append("\n3. **持续监控**: 建立性能监控体系")
        report_lines.append("\n4. **基准测试**: 定期进行性能基准测试")
        
        return "\n".join(report_lines)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="VIVTransformer 性能监控和分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --analyze src/
  %(prog)s --monitor --duration 60
  %(prog)s --profile script.py
        """
    )
    
    parser.add_argument(
        '--analyze',
        metavar='PATH',
        help='分析指定路径的性能'
    )
    
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='启动实时性能监控'
    )
    
    parser.add_argument(
        '--profile',
        metavar='SCRIPT',
        help='分析指定脚本的性能'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=30,
        help='监控持续时间（秒）'
    )
    
    parser.add_argument(
        '--output',
        default='reports/performance_report.md',
        help='输出报告文件路径'
    )
    
    parser.add_argument(
        '--format',
        choices=['markdown', 'json'],
        default='markdown',
        help='输出格式'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 初始化性能监控器
    monitor = PerformanceMonitor()
    
    try:
        results = None
        
        if args.analyze:
            if not os.path.exists(args.analyze):
                print(f"❌ 路径不存在: {args.analyze}")
                sys.exit(1)
            
            print(f"🔍 分析性能: {args.analyze}")
            results = monitor.analyze_code_performance(args.analyze)
            
        elif args.profile:
            if not os.path.exists(args.profile):
                print(f"❌ 脚本不存在: {args.profile}")
                sys.exit(1)
            
            print(f"📊 性能分析: {args.profile}")
            
            # 启动监控
            monitor.start_comprehensive_monitoring()
            
            # 执行脚本
            try:
                subprocess.run([sys.executable, args.profile], check=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"脚本执行出错: {e}")
            
            results = monitor.stop_comprehensive_monitoring()
            
        elif args.monitor:
            print(f"📈 启动性能监控 ({args.duration}秒)")
            
            monitor.start_comprehensive_monitoring()
            time.sleep(args.duration)
            results = monitor.stop_comprehensive_monitoring()
            
        else:
            # 默认分析当前目录
            print("🔍 分析当前目录性能")
            results = monitor.analyze_code_performance('.')
        
        if results:
            # 显示结果摘要
            print("\n" + "="*60)
            print("📊 性能分析结果")
            
            resource_summary = results.get('resource_summary', {})
            if resource_summary:
                cpu_info = resource_summary.get('cpu_usage', {})
                memory_info = resource_summary.get('memory_usage', {})
                print(f"🖥️  CPU峰值: {cpu_info.get('peak', 0):.2f}%")
                print(f"💾 内存峰值: {memory_info.get('peak', 0):.2f}MB")
            
            memory_leaks = results.get('memory_leaks', [])
            print(f"🚨 内存泄漏: {len(memory_leaks)}")
            
            bottlenecks = results.get('performance_bottlenecks', [])
            critical_bottlenecks = [b for b in bottlenecks if b['severity'] in ['critical', 'high']]
            print(f"⚡ 性能瓶颈: {len(critical_bottlenecks)}/{len(bottlenecks)}")
            
            # 生成报告
            if args.format == 'json':
                output_content = json.dumps(results, indent=2, ensure_ascii=False)
            else:
                output_content = monitor.generate_performance_report(results)
            
            # 保存报告
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            print(f"\n📋 详细报告已保存: {args.output}")
            
            # 显示关键建议
            if critical_bottlenecks:
                print("\n⚡ 关键性能瓶颈:")
                for bottleneck in critical_bottlenecks[:3]:
                    print(f"   - {bottleneck['function_name']}: {bottleneck['cumulative_time']:.4f}s")
            
            if memory_leaks:
                critical_leaks = [leak for leak in memory_leaks if leak['severity'] == 'critical']
                if critical_leaks:
                    print("\n🚨 严重内存泄漏:")
                    for leak in critical_leaks[:3]:
                        print(f"   - {leak['description']}")
            
            print("="*60)
        
    except KeyboardInterrupt:
        print("\n⏹️  操作被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"性能分析失败: {e}")
        print(f"❌ 性能分析失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()