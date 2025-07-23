#!/usr/bin/env python3
"""
固定轮数输入输出预测可视化工具

该脚本提供了专门用于可视化固定轮数的输入、输出和预测结果的功能，
包括：
- 多轮次预测结果对比
- 输入输出预测三联图
- 误差分析可视化
- 时间序列预测对比
- 批量样本可视化
"""

import os
import sys
import yaml
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from datetime import datetime
import seaborn as sns

# 设置日志 - 必须在使用logger之前定义
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置matplotlib字体 - 兼容Linux服务器
try:
    # 尝试设置中文字体
    import matplotlib.font_manager as fm
    # 检查可用的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    available_font = None
    for font in chinese_fonts:
        if any(font in f.name for f in fm.fontManager.ttflist):
            available_font = font
            break
    
    if available_font:
        plt.rcParams['font.sans-serif'] = [available_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        logger.info(f"使用字体: {available_font}")
    else:
        # 如果没有中文字体，使用英文标签
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        logger.warning("未找到中文字体，将使用英文标签")
        # 设置全局标志，用于后续判断是否使用中文
        globals()['USE_CHINESE'] = False
except Exception as e:
    logger.warning(f"字体设置失败: {e}，使用默认字体")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    globals()['USE_CHINESE'] = False

sns.set_style("whitegrid")

# 检查是否使用中文标签
USE_CHINESE = globals().get('USE_CHINESE', True)


class PredictionVisualizer:
    """
    固定轮数预测结果可视化器
    
    提供多种可视化方式来展示模型在固定轮数下的预测效果
    """
    
    def __init__(self, output_dir: str = "prediction_visualizations"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建子目录
        (self.output_dir / "single_samples").mkdir(exist_ok=True)
        (self.output_dir / "batch_comparisons").mkdir(exist_ok=True)
        (self.output_dir / "error_analysis").mkdir(exist_ok=True)
        (self.output_dir / "time_series").mkdir(exist_ok=True)
        
        # 设置标签语言
        self.use_chinese = USE_CHINESE
        
        logger.info(f"预测可视化器初始化完成，输出目录: {self.output_dir}")
    
    def _get_label(self, chinese_text: str, english_text: str) -> str:
        """
        根据字体支持情况返回适当的标签文本
        
        Args:
            chinese_text: 中文文本
            english_text: 英文文本
        
        Returns:
            适当的标签文本
        """
        return chinese_text if self.use_chinese else english_text
    
    def visualize_single_prediction(
        self, 
        input_data: np.ndarray, 
        target_data: np.ndarray, 
        prediction_data: np.ndarray,
        epoch: int,
        sample_idx: int = 0,
        time_step: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        可视化单个样本的输入、目标和预测结果
        
        Args:
            input_data: 输入数据 (可以是1D或2D)
            target_data: 目标数据 (可以是1D或2D)
            prediction_data: 预测数据 (可以是1D或2D)
            epoch: 当前轮数
            sample_idx: 样本索引
            time_step: 时间步（可选）
            save_path: 保存路径（可选）
        
        Returns:
            保存的图片路径
        """
        # 数据预处理
        input_2d = self._reshape_to_2d(input_data)
        target_2d = self._reshape_to_2d(target_data)
        prediction_2d = self._reshape_to_2d(prediction_data)
        
        # 创建图形
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.4)
        
        # 输入数据
        ax1 = fig.add_subplot(gs[:, 0:2])
        im1 = ax1.imshow(input_2d, cmap='viridis', interpolation='nearest')
        ax1.set_title(f'{self._get_label("输入数据", "Input Data")}\n{self._get_label("形状", "Shape")}: {input_2d.shape}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # 目标数据
        ax2 = fig.add_subplot(gs[:, 2:4])
        im2 = ax2.imshow(target_2d, cmap='plasma', interpolation='nearest')
        ax2.set_title(f'{self._get_label("真实输出", "Ground Truth")}\n{self._get_label("形状", "Shape")}: {target_2d.shape}', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # 预测数据
        ax3 = fig.add_subplot(gs[:, 4:6])
        im3 = ax3.imshow(prediction_2d, cmap='plasma', interpolation='nearest')
        ax3.set_title(f'{self._get_label("模型预测", "Model Prediction")}\n{self._get_label("形状", "Shape")}: {prediction_2d.shape}', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 添加总标题
        time_info = f", t={time_step:.3f}" if time_step is not None else ""
        title_text = self._get_label(
            f'第 {epoch} 轮预测结果对比 - 样本 {sample_idx}{time_info}',
            f'Epoch {epoch} Prediction Comparison - Sample {sample_idx}{time_info}'
        )
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        # 保存图片
        if save_path is None:
            save_path = self.output_dir / "single_samples" / f"epoch_{epoch}_sample_{sample_idx}_prediction.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_msg = self._get_label(f"单样本预测可视化已保存: {save_path}", f"Single sample prediction visualization saved: {save_path}")
        logger.info(log_msg)
        return str(save_path)
    
    def visualize_prediction_with_error(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray,
        prediction_data: np.ndarray,
        epoch: int,
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> str:
        """
        可视化预测结果并包含误差分析
        
        Args:
            input_data: 输入数据
            target_data: 目标数据
            prediction_data: 预测数据
            epoch: 当前轮数
            sample_idx: 样本索引
            save_path: 保存路径（可选）
        
        Returns:
            保存的图片路径
        """
        # 数据预处理
        input_2d = self._reshape_to_2d(input_data)
        target_2d = self._reshape_to_2d(target_data)
        prediction_2d = self._reshape_to_2d(prediction_data)
        
        # 计算误差
        absolute_error = np.abs(target_2d - prediction_2d)
        relative_error = np.abs(target_2d - prediction_2d) / (np.abs(target_2d) + 1e-8)
        
        # 创建图形
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 第一行：输入、目标、预测
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(input_2d, cmap='viridis')
        ax1.set_title(self._get_label('输入数据', 'Input Data'), fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(target_2d, cmap='plasma')
        ax2.set_title(self._get_label('真实输出', 'Ground Truth'), fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(prediction_2d, cmap='plasma')
        ax3.set_title(self._get_label('模型预测', 'Model Prediction'), fontweight='bold')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 第一行第四列：统计信息
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        # 计算统计指标
        mse = np.mean((target_2d - prediction_2d) ** 2)
        mae = np.mean(np.abs(target_2d - prediction_2d))
        max_error = np.max(absolute_error)
        mean_rel_error = np.mean(relative_error)
        
        if self.use_chinese:
            stats_text = f"""
统计指标:

MSE: {mse:.6f}
MAE: {mae:.6f}
最大绝对误差: {max_error:.6f}
平均相对误差: {mean_rel_error:.4f}

数据范围:
目标值: [{np.min(target_2d):.3f}, {np.max(target_2d):.3f}]
预测值: [{np.min(prediction_2d):.3f}, {np.max(prediction_2d):.3f}]
            """
        else:
            stats_text = f"""
Statistics:

MSE: {mse:.6f}
MAE: {mae:.6f}
Max Abs Error: {max_error:.6f}
Mean Rel Error: {mean_rel_error:.4f}

Data Range:
Target: [{np.min(target_2d):.3f}, {np.max(target_2d):.3f}]
Prediction: [{np.min(prediction_2d):.3f}, {np.max(prediction_2d):.3f}]
            """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 第二行：误差分析
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(absolute_error, cmap='hot')
        ax5.set_title(self._get_label('绝对误差', 'Absolute Error'), fontweight='bold')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        ax6 = fig.add_subplot(gs[1, 1])
        im6 = ax6.imshow(relative_error, cmap='hot')
        ax6.set_title(self._get_label('相对误差', 'Relative Error'), fontweight='bold')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        # 误差分布直方图
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.hist(absolute_error.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        ax7.set_title(self._get_label('绝对误差分布', 'Absolute Error Distribution'), fontweight='bold')
        ax7.set_xlabel(self._get_label('绝对误差', 'Absolute Error'))
        ax7.set_ylabel(self._get_label('频次', 'Frequency'))
        ax7.grid(True, alpha=0.3)
        
        # 散点图：真实值 vs 预测值
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.scatter(target_2d.flatten(), prediction_2d.flatten(), alpha=0.5, s=1)
        min_val = min(np.min(target_2d), np.min(prediction_2d))
        max_val = max(np.max(target_2d), np.max(prediction_2d))
        ax8.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax8.set_xlabel(self._get_label('真实值', 'Ground Truth'))
        ax8.set_ylabel(self._get_label('预测值', 'Prediction'))
        ax8.set_title(self._get_label('真实值 vs 预测值', 'Ground Truth vs Prediction'), fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # 第三行：剖面线对比
        ax9 = fig.add_subplot(gs[2, :])
        
        # 选择中间行和中间列进行剖面分析
        mid_row = target_2d.shape[0] // 2
        mid_col = target_2d.shape[1] // 2
        
        x_profile = np.arange(target_2d.shape[1])
        y_profile = np.arange(target_2d.shape[0])
        
        # 水平剖面
        gt_label = self._get_label('真实值 (水平剖面)', 'Ground Truth (Horizontal)')
        pred_label = self._get_label('预测值 (水平剖面)', 'Prediction (Horizontal)')
        ax9.plot(x_profile, target_2d[mid_row, :], 'b-', linewidth=2, label=gt_label)
        ax9.plot(x_profile, prediction_2d[mid_row, :], 'r--', linewidth=2, label=pred_label)
        
        ax9.set_xlabel(self._get_label('位置', 'Position'))
        ax9.set_ylabel(self._get_label('数值', 'Value'))
        profile_title = self._get_label(f'中间行剖面对比 (行 {mid_row})', f'Middle Row Profile Comparison (Row {mid_row})')
        ax9.set_title(profile_title, fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # 添加总标题
        main_title = self._get_label(
            f'第 {epoch} 轮预测结果详细分析 - 样本 {sample_idx}',
            f'Epoch {epoch} Detailed Prediction Analysis - Sample {sample_idx}'
        )
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
        # 保存图片
        if save_path is None:
            save_path = self.output_dir / "error_analysis" / f"epoch_{epoch}_sample_{sample_idx}_error_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_msg = self._get_label(f"误差分析可视化已保存: {save_path}", f"Error analysis visualization saved: {save_path}")
        logger.info(log_msg)
        return str(save_path)
    
    def visualize_batch_predictions(
        self,
        inputs_batch: np.ndarray,
        targets_batch: np.ndarray,
        predictions_batch: np.ndarray,
        epoch: int,
        num_samples: int = 6,
        save_path: Optional[str] = None
    ) -> str:
        """
        可视化一个批次中多个样本的预测结果
        
        Args:
            inputs_batch: 输入批次数据 [batch_size, ...]
            targets_batch: 目标批次数据 [batch_size, ...]
            predictions_batch: 预测批次数据 [batch_size, ...]
            epoch: 当前轮数
            num_samples: 要可视化的样本数量
            save_path: 保存路径（可选）
        
        Returns:
            保存的图片路径
        """
        batch_size = min(inputs_batch.shape[0], num_samples)
        
        # 创建图形
        fig = plt.figure(figsize=(18, 4 * batch_size))
        gs = gridspec.GridSpec(batch_size, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        for i in range(batch_size):
            # 获取单个样本
            input_2d = self._reshape_to_2d(inputs_batch[i])
            target_2d = self._reshape_to_2d(targets_batch[i])
            prediction_2d = self._reshape_to_2d(predictions_batch[i])
            
            # 输入
            ax1 = fig.add_subplot(gs[i, 0])
            im1 = ax1.imshow(input_2d, cmap='viridis')
            input_title = self._get_label(f'样本 {i+1} - 输入', f'Sample {i+1} - Input')
            ax1.set_title(input_title, fontweight='bold')
            if i == 0:
                input_ylabel = self._get_label('输入数据', 'Input Data')
                ax1.set_ylabel(input_ylabel, fontsize=12, fontweight='bold')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # 目标
            ax2 = fig.add_subplot(gs[i, 1])
            im2 = ax2.imshow(target_2d, cmap='plasma')
            target_title = self._get_label(f'样本 {i+1} - 真实输出', f'Sample {i+1} - Ground Truth')
            ax2.set_title(target_title, fontweight='bold')
            if i == 0:
                target_ylabel = self._get_label('真实输出', 'Ground Truth')
                ax2.set_ylabel(target_ylabel, fontsize=12, fontweight='bold')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            # 预测
            ax3 = fig.add_subplot(gs[i, 2])
            im3 = ax3.imshow(prediction_2d, cmap='plasma')
            pred_title = self._get_label(f'样本 {i+1} - 模型预测', f'Sample {i+1} - Model Prediction')
            ax3.set_title(pred_title, fontweight='bold')
            if i == 0:
                pred_ylabel = self._get_label('模型预测', 'Model Prediction')
                ax3.set_ylabel(pred_ylabel, fontsize=12, fontweight='bold')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            
            # 计算并显示MSE
            mse = np.mean((target_2d - prediction_2d) ** 2)
            ax3.text(0.02, 0.98, f'MSE: {mse:.6f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=10)
        
        # 添加总标题
        batch_title = self._get_label(
            f'第 {epoch} 轮批次预测结果对比 ({batch_size} 个样本)',
            f'Epoch {epoch} Batch Prediction Comparison ({batch_size} samples)'
        )
        fig.suptitle(batch_title, fontsize=16, fontweight='bold')
        
        # 保存图片
        if save_path is None:
            save_path = self.output_dir / "batch_comparisons" / f"epoch_{epoch}_batch_predictions.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log_msg = self._get_label(f"批次预测可视化已保存: {save_path}", f"Batch prediction visualization saved: {save_path}")
        logger.info(log_msg)
        return str(save_path)
    
    def visualize_training_progress(
        self,
        prediction_history: List[Dict[str, np.ndarray]],
        epochs: List[int],
        sample_idx: int = 0,
        save_path: Optional[str] = None
    ) -> str:
        """
        可视化训练过程中预测结果的变化
        
        Args:
            prediction_history: 预测历史记录，每个元素包含 {'input', 'target', 'prediction'}
            epochs: 对应的轮数列表
            sample_idx: 要跟踪的样本索引
            save_path: 保存路径（可选）
        
        Returns:
            保存的图片路径
        """
        num_epochs = len(prediction_history)
        
        # 创建输出目录
        progress_dir = self.output_dir / "time_series" / f"sample_{sample_idx}_progress"
        progress_dir.mkdir(parents=True, exist_ok=True)
        
        mse_history = []
        saved_files = []
        
        # 分别保存每个epoch的预测结果
        for i, (epoch, pred_data) in enumerate(zip(epochs, prediction_history)):
            input_2d = self._reshape_to_2d(pred_data['input'])
            target_2d = self._reshape_to_2d(pred_data['target'])
            prediction_2d = self._reshape_to_2d(pred_data['prediction'])
            
            # 计算MSE
            mse = np.mean((target_2d - prediction_2d) ** 2)
            mse_history.append(mse)
            
            # 为每个epoch创建单独的图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 输入数据
            im1 = axes[0].imshow(input_2d, cmap='viridis')
            input_title = self._get_label('输入数据', 'Input Data')
            axes[0].set_title(f'{input_title} - Epoch {epoch}', fontweight='bold')
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            
            # 真实输出
            im2 = axes[1].imshow(target_2d, cmap='plasma')
            target_title = self._get_label('真实输出', 'Ground Truth')
            axes[1].set_title(f'{target_title} - Epoch {epoch}', fontweight='bold')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 模型预测
            im3 = axes[2].imshow(prediction_2d, cmap='plasma')
            pred_title = self._get_label('模型预测', 'Model Prediction')
            axes[2].set_title(f'{pred_title} - Epoch {epoch}\nMSE: {mse:.6f}', fontweight='bold')
            plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            
            # 保存单个epoch的图
            epoch_save_path = progress_dir / f"epoch_{epoch:04d}_prediction.png"
            plt.tight_layout()
            try:
                plt.savefig(epoch_save_path, dpi=150, bbox_inches='tight')
                logger.info(f"已保存epoch {epoch}图片: {epoch_save_path}")
                saved_files.append(str(epoch_save_path))
            except Exception as e:
                logger.error(f"保存epoch {epoch}图片失败: {e}")
            plt.close()
        
        # 单独创建MSE趋势图
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, mse_history, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        trend_title = self._get_label('预测误差变化趋势', 'Prediction Error Trend')
        ax.set_title(f'{trend_title} - Sample {sample_idx}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 保存MSE趋势图
        mse_save_path = progress_dir / "mse_trend.png"
        plt.tight_layout()
        try:
            plt.savefig(mse_save_path, dpi=150, bbox_inches='tight')
            logger.info(f"已保存MSE趋势图: {mse_save_path}")
            saved_files.append(str(mse_save_path))
        except Exception as e:
            logger.error(f"保存MSE趋势图失败: {e}")
        plt.close()
        
        # 创建汇总图（较小尺寸）
        max_epochs_in_summary = min(10, num_epochs)  # 最多显示10个epoch
        step = max(1, num_epochs // max_epochs_in_summary)
        selected_indices = list(range(0, num_epochs, step))[:max_epochs_in_summary]
        
        fig = plt.figure(figsize=(16, 4 * len(selected_indices)))
        gs = gridspec.GridSpec(len(selected_indices), 4, figure=fig, hspace=0.3, wspace=0.3)
        
        for plot_idx, i in enumerate(selected_indices):
            epoch = epochs[i]
            pred_data = prediction_history[i]
            
            input_2d = self._reshape_to_2d(pred_data['input'])
            target_2d = self._reshape_to_2d(pred_data['target'])
            prediction_2d = self._reshape_to_2d(pred_data['prediction'])
            mse = mse_history[i]
            
            # 输入（只在第一行显示）
            if plot_idx == 0:
                ax1 = fig.add_subplot(gs[plot_idx, 0])
                im1 = ax1.imshow(input_2d, cmap='viridis')
                input_title = self._get_label('输入数据', 'Input Data')
                ax1.set_title(input_title, fontweight='bold')
                ax1.set_ylabel(f'Epoch {epoch}', fontsize=12, fontweight='bold')
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            else:
                ax1 = fig.add_subplot(gs[plot_idx, 0])
                ax1.axis('off')
                ax1.set_ylabel(f'Epoch {epoch}', fontsize=12, fontweight='bold')
            
            # 目标（只在第一行显示）
            if plot_idx == 0:
                ax2 = fig.add_subplot(gs[plot_idx, 1])
                im2 = ax2.imshow(target_2d, cmap='plasma')
                target_title = self._get_label('真实输出', 'Ground Truth')
                ax2.set_title(target_title, fontweight='bold')
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            else:
                ax2 = fig.add_subplot(gs[plot_idx, 1])
                ax2.axis('off')
            
            # 预测
            ax3 = fig.add_subplot(gs[plot_idx, 2])
            im3 = ax3.imshow(prediction_2d, cmap='plasma')
            if plot_idx == 0:
                pred_title = self._get_label('模型预测', 'Model Prediction')
                ax3.set_title(pred_title, fontweight='bold')
            ax3.text(0.02, 0.98, f'MSE: {mse:.6f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=10)
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # MSE趋势图（在最后一列）
        ax4 = fig.add_subplot(gs[:, 3])
        ax4.plot(epochs, mse_history, 'b-o', linewidth=2, markersize=4)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('MSE')
        trend_title = self._get_label('预测误差变化趋势', 'Prediction Error Trend')
        ax4.set_title(trend_title, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        # 添加总标题
        progress_title = self._get_label(
            f'训练过程预测结果变化 - 样本 {sample_idx} (精选)',
            f'Training Progress Prediction Changes - Sample {sample_idx} (Selected)'
        )
        fig.suptitle(progress_title, fontsize=16, fontweight='bold')
        
        # 保存汇总图
        if save_path is None:
            save_path = progress_dir / "training_progress_summary.png"
        
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"已保存汇总图: {save_path}")
            saved_files.append(str(save_path))
        except Exception as e:
            logger.error(f"保存汇总图失败: {e}")
        plt.close()
        
        log_msg = self._get_label(
            f"训练进度可视化已保存: {len(saved_files)} 个文件到 {progress_dir}", 
            f"Training progress visualization saved: {len(saved_files)} files to {progress_dir}"
        )
        logger.info(log_msg)
        return str(save_path)
    
    def _reshape_to_2d(self, data: np.ndarray) -> np.ndarray:
        """
        将数据重塑为2D格式用于可视化
        
        Args:
            data: 输入数据
        
        Returns:
            2D格式的数据
        """
        if len(data.shape) == 1:
            # 1D数据，尝试重塑为正方形
            size = int(np.sqrt(len(data)))
            if size * size == len(data):
                return data.reshape(size, size)
            else:
                # 如果不是完全平方数，重塑为接近正方形的矩形
                height = int(np.sqrt(len(data)))
                width = len(data) // height
                return data[:height*width].reshape(height, width)
        elif len(data.shape) == 2:
            return data
        elif len(data.shape) == 3:
            # 3D数据，取第一个通道或平均
            if data.shape[0] == 1:
                return data[0]
            else:
                return np.mean(data, axis=0)
        else:
            # 高维数据，展平后重塑
            flattened = data.flatten()
            size = int(np.sqrt(len(flattened)))
            return flattened[:size*size].reshape(size, size)
    
    def generate_summary_report(self, epoch: int, num_samples_visualized: int) -> str:
        """
        生成可视化摘要报告
        
        Args:
            epoch: 当前轮数
            num_samples_visualized: 已可视化的样本数量
        
        Returns:
            报告文件路径
        """
        report_content = f"""
# 第 {epoch} 轮预测可视化报告

## 📊 可视化摘要

- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **轮数**: {epoch}
- **可视化样本数**: {num_samples_visualized}
- **输出目录**: {self.output_dir}

## 📁 生成的文件

### 单样本预测对比
- 位置: `single_samples/`
- 内容: 输入、真实输出、模型预测的三联图
- 用途: 详细查看单个样本的预测效果

### 批次预测对比
- 位置: `batch_comparisons/`
- 内容: 多个样本的预测结果并排对比
- 用途: 快速浏览多个样本的预测质量

### 误差分析
- 位置: `error_analysis/`
- 内容: 详细的误差分析，包括绝对误差、相对误差、统计指标
- 用途: 深入分析模型的预测误差特征

### 训练进度跟踪
- 位置: `time_series/`
- 内容: 训练过程中预测结果的变化趋势
- 用途: 观察模型学习过程和收敛情况

## 🎯 使用建议

1. **快速评估**: 查看批次预测对比图，快速了解整体预测质量
2. **详细分析**: 查看误差分析图，了解误差分布和特征
3. **训练监控**: 查看训练进度图，监控模型学习效果
4. **问题诊断**: 如果发现预测质量问题，重点查看误差分析结果

## 📈 下一步行动

- 如果预测质量满意，可以继续训练或进行模型保存
- 如果发现系统性误差，考虑调整模型结构或训练参数
- 如果误差过大，检查数据预处理和模型配置

---

**生成工具**: PredictionVisualizer v1.0
**项目**: VIVTransformer 压力场重建
        """
        
        report_path = self.output_dir / f"epoch_{epoch}_visualization_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        log_msg = self._get_label(f"可视化报告已生成: {report_path}", f"Visualization report generated: {report_path}")
        logger.info(log_msg)
        return str(report_path)


def demo_visualization():
    """
    演示可视化功能
    """
    logger.info("Starting prediction visualization demo...")
    
    # 创建可视化器
    visualizer = PredictionVisualizer("demo_prediction_visualizations")
    
    # 生成模拟数据
    np.random.seed(42)
    
    # 模拟压力场数据
    input_data = np.random.randn(20, 20)  # 20x20 输入
    target_data = np.random.randn(200, 200)  # 200x200 输出
    
    # 模拟预测数据（添加一些噪声）
    prediction_data = target_data + np.random.normal(0, 0.1, target_data.shape)
    
    # 演示单样本可视化
    logger.info("Generating single sample prediction visualization...")
    visualizer.visualize_single_prediction(
        input_data, target_data, prediction_data, 
        epoch=10, sample_idx=0, time_step=0.5
    )
    
    # 演示误差分析
    logger.info("Generating error analysis visualization...")
    visualizer.visualize_prediction_with_error(
        input_data, target_data, prediction_data,
        epoch=10, sample_idx=0
    )
    
    # 演示批次可视化
    logger.info("Generating batch prediction visualization...")
    batch_size = 4
    inputs_batch = np.random.randn(batch_size, 20, 20)
    targets_batch = np.random.randn(batch_size, 200, 200)
    predictions_batch = targets_batch + np.random.normal(0, 0.1, targets_batch.shape)
    
    visualizer.visualize_batch_predictions(
        inputs_batch, targets_batch, predictions_batch,
        epoch=10, num_samples=4
    )
    
    # 演示训练进度可视化
    logger.info("Generating training progress visualization...")
    prediction_history = []
    epochs = [1, 5, 10, 15, 20]
    
    for i, epoch in enumerate(epochs):
        # 模拟预测质量逐渐提升
        noise_level = 0.5 * (1 - i / len(epochs))  # 噪声逐渐减少
        pred = target_data + np.random.normal(0, noise_level, target_data.shape)
        
        prediction_history.append({
            'input': input_data,
            'target': target_data,
            'prediction': pred
        })
    
    visualizer.visualize_training_progress(
        prediction_history, epochs, sample_idx=0
    )
    
    # 生成摘要报告
    logger.info("Generating visualization summary report...")
    visualizer.generate_summary_report(epoch=10, num_samples_visualized=4)
    
    logger.info(f"Demo completed! All visualization results saved to: {visualizer.output_dir}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='固定轮数预测可视化工具')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    parser.add_argument('--output-dir', type=str, default='prediction_visualizations', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_visualization()
    else:
        logger.info("请使用 --demo 参数运行演示，或在代码中调用相应的可视化函数")
        logger.info("可用的可视化功能:")
        logger.info("  - visualize_single_prediction: 单样本预测对比")
        logger.info("  - visualize_prediction_with_error: 预测误差分析")
        logger.info("  - visualize_batch_predictions: 批次预测对比")
        logger.info("  - visualize_training_progress: 训练进度跟踪")


if __name__ == '__main__':
    main()