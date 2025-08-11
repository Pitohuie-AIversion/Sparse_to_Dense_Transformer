"""训练过程可视化工具

该模块提供了训练过程中各种指标的可视化功能，包括：
- 损失曲线绘制
- 硬件使用情况监控
- 学习率变化曲线
- 梯度范数统计
- 训练仪表板生成
- HTML报告导出
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, log_dir: str):
        """初始化可视化器
        
        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = Path(log_dir)
        self.output_dir = self.log_dir / 'visualizations'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载数据
        self.hardware_metrics = self._load_hardware_metrics()
        self.epoch_summaries = self._load_epoch_summaries()
        self.training_metrics = self._load_training_metrics()
    
    def _load_hardware_metrics(self) -> Optional[Dict]:
        """加载硬件监控数据"""
        hardware_file = self.log_dir / 'hardware_metrics.json'
        if hardware_file.exists():
            try:
                with open(hardware_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载硬件监控数据失败: {e}")
        return None
    
    def _load_epoch_summaries(self) -> List[Dict]:
        """加载epoch摘要数据"""
        epoch_file = self.log_dir / 'epoch_summaries.json'
        if epoch_file.exists():
            try:
                with open(epoch_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载epoch摘要数据失败: {e}")
        return []
    
    def _load_training_metrics(self) -> List[Dict]:
        """加载训练指标数据"""
        metrics_file = self.log_dir / 'training_metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载训练指标数据失败: {e}")
        return []
    
    def plot_loss_curves(self, save_path: Optional[str] = None, show: bool = False) -> str:
        """绘制损失曲线"""
        if not self.epoch_summaries:
            print("没有找到epoch摘要数据")
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = list(range(1, len(self.epoch_summaries) + 1))
        train_losses = [epoch.get('train_loss', 0) for epoch in self.epoch_summaries]
        val_losses = [epoch.get('valid_loss', 0) for epoch in self.epoch_summaries]
        test_losses = [epoch.get('test_loss', 0) for epoch in self.epoch_summaries]
        
        # 训练和验证损失
        ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失值')
        ax1.set_title('训练和验证损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 测试损失
        if any(loss > 0 for loss in test_losses):
            ax2.plot(epochs, test_losses, 'g-', label='测试损失', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('损失值')
            ax2.set_title('测试损失曲线')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '暂无测试损失数据', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('测试损失曲线')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'loss_curves.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def plot_hardware_usage(self, save_path: Optional[str] = None, show: bool = False) -> str:
        """绘制硬件使用情况"""
        if not self.hardware_metrics or "epoch_metrics" not in self.hardware_metrics:
            print("没有找到硬件监控数据")
            return ""
        
        epoch_metrics = self.hardware_metrics["epoch_metrics"]
        end_metrics = [m for m in epoch_metrics if m.get("event") == "epoch_end"]
        
        if not end_metrics:
            print("没有找到epoch结束时的硬件数据")
            return ""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = list(range(1, len(end_metrics) + 1))
        
        # CPU使用率
        cpu_usage = [m['cpu_percent'] for m in end_metrics]
        ax1.plot(epochs, cpu_usage, 'b-', linewidth=2, marker='o')
        ax1.set_title('CPU使用率')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('使用率 (%)')
        ax1.grid(True, alpha=0.3)
        
        # 内存使用率
        memory_usage = [m['memory_percent'] for m in end_metrics]
        ax2.plot(epochs, memory_usage, 'g-', linewidth=2, marker='s')
        ax2.set_title('内存使用率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('使用率 (%)')
        ax2.grid(True, alpha=0.3)
        
        # GPU使用率
        if end_metrics[0].get('gpu_metrics'):
            gpu_usage = [m['gpu_metrics'][0]['gpu_utilization_percent'] for m in end_metrics if m.get('gpu_metrics')]
            ax3.plot(epochs[:len(gpu_usage)], gpu_usage, 'r-', linewidth=2, marker='^')
            ax3.set_title('GPU使用率')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('使用率 (%)')
            ax3.grid(True, alpha=0.3)
            
            # GPU显存使用
            gpu_memory = [m['gpu_metrics'][0]['memory_used_mb'] for m in end_metrics if m.get('gpu_metrics')]
            ax4.plot(epochs[:len(gpu_memory)], gpu_memory, 'm-', linewidth=2, marker='d')
            ax4.set_title('GPU显存使用')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('显存 (MB)')
            ax4.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '暂无GPU数据', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('GPU使用率')
            ax4.text(0.5, 0.5, '暂无GPU显存数据', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('GPU显存使用')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / 'hardware_usage.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def create_training_dashboard(self, save_path: Optional[str] = None, show: bool = False) -> str:
        """创建训练仪表板"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 损失曲线 (占据左上角2x2)
        ax_loss = fig.add_subplot(gs[0:2, 0:2])
        if self.epoch_summaries:
            epochs = list(range(1, len(self.epoch_summaries) + 1))
            train_losses = [epoch.get('train_loss', 0) for epoch in self.epoch_summaries]
            val_losses = [epoch.get('valid_loss', 0) for epoch in self.epoch_summaries]
            
            ax_loss.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
            ax_loss.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
            ax_loss.set_title('损失曲线', fontsize=14, fontweight='bold')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('损失值')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)
        
        # 硬件使用情况 (右上角)
        ax_hardware = fig.add_subplot(gs[0, 2:4])
        if self.hardware_metrics and "epoch_metrics" in self.hardware_metrics:
            end_metrics = [m for m in self.hardware_metrics["epoch_metrics"] if m.get("event") == "epoch_end"]
            if end_metrics:
                epochs_hw = list(range(1, len(end_metrics) + 1))
                cpu_usage = [m['cpu_percent'] for m in end_metrics]
                memory_usage = [m['memory_percent'] for m in end_metrics]
                
                ax_hardware.plot(epochs_hw, cpu_usage, 'b-', label='CPU', linewidth=2)
                ax_hardware.plot(epochs_hw, memory_usage, 'g-', label='内存', linewidth=2)
                
                if end_metrics[0].get('gpu_metrics'):
                    gpu_usage = [m['gpu_metrics'][0]['gpu_utilization_percent'] for m in end_metrics if m.get('gpu_metrics')]
                    ax_hardware.plot(epochs_hw[:len(gpu_usage)], gpu_usage, 'r-', label='GPU', linewidth=2)
                
                ax_hardware.set_title('硬件使用率', fontsize=14, fontweight='bold')
                ax_hardware.set_xlabel('Epoch')
                ax_hardware.set_ylabel('使用率 (%)')
                ax_hardware.legend()
                ax_hardware.grid(True, alpha=0.3)
        
        # 训练时间统计 (右中)
        ax_time = fig.add_subplot(gs[1, 2:4])
        if self.epoch_summaries:
            epoch_times = [epoch.get('epoch_duration_seconds', 0) for epoch in self.epoch_summaries]
            if any(t > 0 for t in epoch_times):
                epochs_time = list(range(1, len(epoch_times) + 1))
                ax_time.bar(epochs_time, epoch_times, alpha=0.7, color='orange')
                ax_time.set_title('每轮训练时间', fontsize=14, fontweight='bold')
                ax_time.set_xlabel('Epoch')
                ax_time.set_ylabel('时间 (秒)')
                ax_time.grid(True, alpha=0.3)
        
        # 统计摘要 (底部)
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('off')
        
        stats_text = self._generate_dashboard_stats()
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                     fontsize=12, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        if save_path is None:
            save_path = self.output_dir / 'training_dashboard.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        return str(save_path)
    
    def _generate_dashboard_stats(self) -> str:
        """生成仪表板统计信息"""
        stats = []
        
        if self.epoch_summaries:
            total_epochs = len(self.epoch_summaries)
            best_val_loss = min(epoch.get('valid_loss', float('inf')) for epoch in self.epoch_summaries)
            best_epoch = next((i for i, epoch in enumerate(self.epoch_summaries) 
                             if epoch.get('valid_loss') == best_val_loss), 0)
            
            total_time = sum(epoch.get('epoch_duration_seconds', 0) for epoch in self.epoch_summaries)
            avg_time = total_time / total_epochs if total_epochs > 0 else 0
            
            stats.extend([
                f"总训练轮数: {total_epochs}",
                f"最佳验证损失: {best_val_loss:.6f} (第 {best_epoch + 1} 轮)",
                f"总训练时间: {total_time:.2f}秒",
                f"平均每轮时间: {avg_time:.2f}秒"
            ])
        
        if self.hardware_metrics and "system_info" in self.hardware_metrics:
            sys_info = self.hardware_metrics["system_info"]
            stats.extend([
                f"CPU: {sys_info.get('cpu_count', 'N/A')} 核心",
                f"内存: {sys_info.get('memory_total_gb', 'N/A'):.1f} GB",
                f"GPU: {sys_info.get('gpu_name', 'N/A')}"
            ])
        
        return "\n".join(stats)
    
    def generate_stats_text(self, output_dir: Optional[str] = None) -> str:
        """生成训练统计文本摘要"""
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        stats_text = []
        stats_text.append("=" * 60)
        stats_text.append("训练统计摘要")
        stats_text.append("=" * 60)
        
        # 基本信息
        if self.epoch_summaries:
            total_epochs = len(self.epoch_summaries)
            stats_text.append(f"总训练轮数: {total_epochs}")
            
            # 最佳性能
            best_val_loss = min(epoch.get('valid_loss', float('inf')) for epoch in self.epoch_summaries)
            best_epoch = next((i for i, epoch in enumerate(self.epoch_summaries) 
                             if epoch.get('valid_loss') == best_val_loss), 0)
            stats_text.append(f"最佳验证损失: {best_val_loss:.6f} (第 {best_epoch + 1} 轮)")
            
            # 训练时间
            total_time = sum(epoch.get('epoch_duration_seconds', 0) for epoch in self.epoch_summaries)
            avg_time = total_time / total_epochs if total_epochs > 0 else 0
            stats_text.append(f"总训练时间: {total_time:.2f}秒")
            stats_text.append(f"平均每轮时间: {avg_time:.2f}秒")
        
        # 硬件使用情况
        if self.hardware_metrics and "epoch_metrics" in self.hardware_metrics:
            end_metrics = [m for m in self.hardware_metrics["epoch_metrics"] if m.get("event") == "epoch_end"]
            if end_metrics:
                stats_text.append("\n硬件使用统计:")
                cpu_usage = [m['cpu_percent'] for m in end_metrics]
                memory_usage = [m['memory_percent'] for m in end_metrics]
                
                if cpu_usage:
                    stats_text.append(f"平均CPU使用率: {np.mean(cpu_usage):.1f}%")
                    stats_text.append(f"最大CPU使用率: {np.max(cpu_usage):.1f}%")
                
                if memory_usage:
                    stats_text.append(f"平均内存使用率: {np.mean(memory_usage):.1f}%")
                    stats_text.append(f"最大内存使用率: {np.max(memory_usage):.1f}%")
                
                # GPU统计
                if end_metrics[0].get('gpu_metrics'):
                    gpu_utilization = [m['gpu_metrics'][0]['gpu_utilization_percent'] for m in end_metrics if m.get('gpu_metrics')]
                    if gpu_utilization:
                        stats_text.append(f"平均GPU使用率: {np.mean(gpu_utilization):.1f}%")
                        stats_text.append(f"最大GPU使用率: {np.max(gpu_utilization):.1f}%")
        
        stats_text.append("=" * 60)
        
        # 保存到文件
        stats_file = output_dir / 'training_stats.txt'
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(stats_text))
        
        return '\n'.join(stats_text)
    
    def generate_html_report(self, output_dir: Optional[str] = None) -> str:
        """生成HTML格式的训练报告"""
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>训练报告</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; color: #333; }}
        .section {{ margin: 20px 0; }}
        .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .image {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>VIVTransformer 训练报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>训练统计</h2>
        <div class="stats">
            <pre>{self.generate_stats_text(str(output_dir))}</pre>
        </div>
    </div>
    
    <div class="section">
        <h2>可视化图表</h2>
        <div class="image">
            <h3>损失曲线</h3>
            <img src="loss_curves.png" alt="损失曲线">
        </div>
        <div class="image">
            <h3>硬件使用情况</h3>
            <img src="hardware_usage.png" alt="硬件使用情况">
        </div>
        <div class="image">
            <h3>训练仪表板</h3>
            <img src="training_dashboard.png" alt="训练仪表板">
        </div>
    </div>
</body>
</html>
        """
        
        html_file = output_dir / 'training_report.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_file)
    
    def export_to_csv(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """导出数据到CSV文件"""
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        csv_files = {}
        
        # 导出epoch摘要
        if self.epoch_summaries:
            epoch_file = output_dir / 'epoch_summaries.csv'
            df_epochs = pd.DataFrame(self.epoch_summaries)
            df_epochs.to_csv(epoch_file, index=False, encoding='utf-8')
            csv_files['epochs'] = str(epoch_file)
        
        # 导出硬件数据
        if self.hardware_metrics and "epoch_metrics" in self.hardware_metrics:
            hardware_file = output_dir / 'hardware_metrics.csv'
            df_hardware = pd.DataFrame(self.hardware_metrics["epoch_metrics"])
            df_hardware.to_csv(hardware_file, index=False, encoding='utf-8')
            csv_files['hardware'] = str(hardware_file)
        
        return csv_files
    
    def generate_complete_report(self, output_dir: Optional[str] = None, 
                               include_plots: bool = True,
                               include_html: bool = True,
                               include_csv: bool = True) -> Dict[str, str]:
        """生成完整的训练报告"""
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        report_files = {}
        
        # 生成图表
        if include_plots:
            try:
                # 损失曲线
                loss_plot = output_dir / 'loss_curves.png'
                self.plot_loss_curves(save_path=str(loss_plot))
                report_files['loss_plot'] = str(loss_plot)
            except Exception as e:
                print(f"生成损失曲线失败: {e}")
            
            try:
                # 硬件使用情况
                hardware_plot = output_dir / 'hardware_usage.png'
                self.plot_hardware_usage(save_path=str(hardware_plot))
                report_files['hardware_plot'] = str(hardware_plot)
            except Exception as e:
                print(f"生成硬件使用图表失败: {e}")
            
            try:
                # 训练仪表板
                dashboard_plot = output_dir / 'training_dashboard.png'
                self.create_training_dashboard(save_path=str(dashboard_plot))
                report_files['dashboard_plot'] = str(dashboard_plot)
            except Exception as e:
                print(f"生成训练仪表板失败: {e}")
        
        # 生成文本统计
        try:
            stats_text = self.generate_stats_text(str(output_dir))
            report_files['stats_text'] = str(output_dir / 'training_stats.txt')
        except Exception as e:
            print(f"生成统计文本失败: {e}")
        
        # 生成HTML报告
        if include_html:
            try:
                html_file = self.generate_html_report(str(output_dir))
                report_files['html_report'] = html_file
            except Exception as e:
                print(f"生成HTML报告失败: {e}")
        
        # 导出CSV
        if include_csv:
            try:
                csv_files = self.export_to_csv(str(output_dir))
                report_files.update(csv_files)
            except Exception as e:
                print(f"导出CSV文件失败: {e}")
        
        return report_files


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练过程可视化工具')
    parser.add_argument('--log-dir', required=True, help='日志目录路径')
    parser.add_argument('--output-dir', required=True, help='输出目录路径')
    parser.add_argument('--no-plots', action='store_true', help='不生成图表')
    parser.add_argument('--no-html', action='store_true', help='不生成HTML报告')
    parser.add_argument('--no-csv', action='store_true', help='不导出CSV文件')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = TrainingVisualizer(args.log_dir)
    
    # 生成报告
    report_files = visualizer.generate_complete_report(
        output_dir=args.output_dir,
        include_plots=not args.no_plots,
        include_html=not args.no_html,
        include_csv=not args.no_csv
    )
    
    print("训练报告生成完成!")
    print("生成的文件:")
    for file_type, file_path in report_files.items():
        print(f"  {file_type}: {file_path}")


if __name__ == '__main__':
    main()