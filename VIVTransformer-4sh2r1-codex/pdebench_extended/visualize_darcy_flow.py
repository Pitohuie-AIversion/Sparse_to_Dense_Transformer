#!/usr/bin/env python3
"""
2D Darcy Flow 数据可视化脚本

这个脚本用于可视化 2D_DarcyFlow_beta0.1_Train.hdf5 数据集
包含静态图像展示、动态动画生成和数据统计分析
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse
from tqdm import tqdm
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_darcy_data(file_path, sample_idx=0):
    """
    加载 Darcy Flow 数据
    
    Args:
        file_path: HDF5文件路径
        sample_idx: 样本索引
    
    Returns:
        data: 解数据 (t, x, y, channel)
        nu: 扩散系数 (x, y, channel)
    """
    with h5py.File(file_path, 'r') as h5_file:
        print(f"文件中的键: {list(h5_file.keys())}")
        
        # 获取数据形状信息
        if 'tensor' in h5_file:
            tensor_shape = h5_file['tensor'].shape
            print(f"tensor 形状: {tensor_shape}")
            
            # 加载数据
            data = np.array(h5_file['tensor'][sample_idx], dtype=np.float32)
            print(f"加载的数据形状: {data.shape}")
        else:
            raise KeyError("未找到 'tensor' 键")
            
        # 加载扩散系数（如果存在）
        nu = None
        if 'nu' in h5_file:
            nu = np.array(h5_file['nu'][sample_idx], dtype=np.float32)
            print(f"扩散系数形状: {nu.shape}")
        
        return data, nu

def analyze_data_statistics(data, nu=None):
    """
    分析数据统计信息
    """
    print("\n=== 数据统计分析 ===")
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"数据范围: [{data.min():.6f}, {data.max():.6f}]")
    print(f"数据均值: {data.mean():.6f}")
    print(f"数据标准差: {data.std():.6f}")
    
    if nu is not None:
        print(f"\n扩散系数形状: {nu.shape}")
        print(f"扩散系数范围: [{nu.min():.6f}, {nu.max():.6f}]")
        print(f"扩散系数均值: {nu.mean():.6f}")

def create_static_visualization(data, nu=None, save_path="darcy_flow_static.png"):
    """
    创建静态可视化图像
    """
    # 处理不同的数据形状
    if len(data.shape) == 4:  # (t, x, y, channel)
        # 显示第一个时间步和最后一个时间步
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 第一个时间步
        im1 = axes[0, 0].imshow(data[0, :, :, 0], cmap='viridis')
        axes[0, 0].set_title('t=0 时刻的解')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 最后一个时间步
        im2 = axes[0, 1].imshow(data[-1, :, :, 0], cmap='viridis')
        axes[0, 1].set_title('最终时刻的解')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 差值图
        diff = data[-1, :, :, 0] - data[0, :, :, 0]
        im3 = axes[0, 2].imshow(diff, cmap='RdBu_r')
        axes[0, 2].set_title('变化量 (最终-初始)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # 时间序列中心点变化
        center_x, center_y = data.shape[1]//2, data.shape[2]//2
        time_series = data[:, center_x, center_y, 0]
        axes[1, 0].plot(time_series)
        axes[1, 0].set_title(f'中心点 ({center_x}, {center_y}) 时间序列')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('数值')
        
        # 数据分布直方图
        axes[1, 1].hist(data.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title('数据分布直方图')
        axes[1, 1].set_xlabel('数值')
        axes[1, 1].set_ylabel('频次')
        
        # 扩散系数（如果存在）
        if nu is not None:
            if len(nu.shape) == 3:
                im4 = axes[1, 2].imshow(nu[:, :, 0], cmap='plasma')
            else:
                im4 = axes[1, 2].imshow(nu, cmap='plasma')
            axes[1, 2].set_title('扩散系数 ν')
            plt.colorbar(im4, ax=axes[1, 2])
        else:
            axes[1, 2].text(0.5, 0.5, '无扩散系数数据', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('扩散系数')
            
    elif len(data.shape) == 3:  # (1, x, y) or (x, y, channel)
        if data.shape[0] == 1:  # (1, x, y)
            data_2d = data[0, :, :]
        else:  # (x, y, channel)
            data_2d = data[:, :, 0]
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 主要数据
        im1 = axes[0].imshow(data_2d, cmap='viridis')
        axes[0].set_title('Darcy Flow 解')
        plt.colorbar(im1, ax=axes[0])
        
        # 扩散系数
        if nu is not None:
            if len(nu.shape) == 3:
                nu_2d = nu[:, :, 0]
            else:
                nu_2d = nu
            im2 = axes[1].imshow(nu_2d, cmap='plasma')
            axes[1].set_title('扩散系数 ν')
            plt.colorbar(im2, ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, '无扩散系数数据', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('扩散系数')
        
        # 数据分布直方图
        axes[2].hist(data_2d.flatten(), bins=50, alpha=0.7, color='blue')
        if nu is not None:
            axes[2].hist(nu_2d.flatten(), bins=50, alpha=0.7, color='red', label='扩散系数')
            axes[2].legend(['解', '扩散系数'])
        axes[2].set_title('数据分布直方图')
        axes[2].set_xlabel('数值')
        axes[2].set_ylabel('频次')
        
    else:  # 其他情况
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data.squeeze(), cmap='viridis')
        ax.set_title('Darcy Flow 数据')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"静态可视化已保存到: {save_path}")
    plt.show()

def create_animation(data, save_path="darcy_flow_animation.gif"):
    """
    创建动态动画
    """
    if len(data.shape) != 4:
        print("数据不是时间序列，无法创建动画")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 设置颜色范围
    vmin, vmax = data.min(), data.max()
    
    # 初始化图像
    im = ax.imshow(data[0, :, :, 0], cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title('2D Darcy Flow 演化')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('数值')
    
    # 添加时间步显示
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        im.set_array(data[frame, :, :, 0])
        time_text.set_text(f'时间步: {frame}/{data.shape[0]-1}')
        return [im, time_text]
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=data.shape[0], 
                                 interval=100, blit=True, repeat=True)
    
    # 保存动画
    writer = animation.PillowWriter(fps=10, bitrate=1800)
    anim.save(save_path, writer=writer)
    print(f"动画已保存到: {save_path}")
    
    plt.show()
    return anim

def create_detailed_analysis(data, nu=None, save_path="darcy_flow_analysis.png"):
    """
    创建详细分析图
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 处理数据形状
    if len(data.shape) == 3 and data.shape[0] == 1:
        # 将 (1, x, y) 转换为 (x, y)
        data_2d = data[0, :, :]
        is_time_series = False
    elif len(data.shape) == 4:  # 时间序列数据
        is_time_series = True
    else:
        data_2d = data.squeeze()
        is_time_series = False
    
    if is_time_series:
        # 1. 时空演化热图
        ax1 = plt.subplot(3, 3, 1)
        # 取中间一行的时空演化
        middle_row = data.shape[1] // 2
        space_time = data[:, middle_row, :, 0].T
        im1 = ax1.imshow(space_time, aspect='auto', cmap='viridis')
        ax1.set_title('时空演化 (中间行)')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('空间位置')
        plt.colorbar(im1, ax=ax1)
        
        # 2. 能量演化
        ax2 = plt.subplot(3, 3, 2)
        energy = np.sum(data**2, axis=(1, 2, 3))
        ax2.plot(energy)
        ax2.set_title('总能量演化')
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('能量')
        
        # 3. 最大值演化
        ax3 = plt.subplot(3, 3, 3)
        max_vals = np.max(data, axis=(1, 2, 3))
        min_vals = np.min(data, axis=(1, 2, 3))
        ax3.plot(max_vals, label='最大值')
        ax3.plot(min_vals, label='最小值')
        ax3.set_title('极值演化')
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('数值')
        ax3.legend()
        
        # 4-6. 不同时间步的空间分布
        time_steps = [0, data.shape[0]//2, data.shape[0]-1]
        titles = ['初始时刻', '中间时刻', '最终时刻']
        
        for i, (t, title) in enumerate(zip(time_steps, titles)):
            ax = plt.subplot(3, 3, 4+i)
            im = ax.imshow(data[t, :, :, 0], cmap='viridis')
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        
        # 7. 梯度分析
        ax7 = plt.subplot(3, 3, 7)
        final_data = data[-1, :, :, 0]
        grad_y, grad_x = np.gradient(final_data)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        im7 = ax7.imshow(grad_magnitude, cmap='hot')
        ax7.set_title('最终时刻梯度幅值')
        plt.colorbar(im7, ax=ax7)
        
        # 8. 统计分布
        ax8 = plt.subplot(3, 3, 8)
        ax8.hist(data[0].flatten(), bins=30, alpha=0.5, label='初始', density=True)
        ax8.hist(data[-1].flatten(), bins=30, alpha=0.5, label='最终', density=True)
        ax8.set_title('数值分布对比')
        ax8.set_xlabel('数值')
        ax8.set_ylabel('密度')
        ax8.legend()
        
        # 9. 扩散系数或相关性分析
        ax9 = plt.subplot(3, 3, 9)
        if nu is not None:
            if len(nu.shape) == 3:
                im9 = ax9.imshow(nu[:, :, 0], cmap='plasma')
            else:
                im9 = ax9.imshow(nu, cmap='plasma')
            ax9.set_title('扩散系数 ν')
            plt.colorbar(im9, ax=ax9)
        else:
            # 计算时间相关性
            correlation = np.corrcoef(data.reshape(data.shape[0], -1))
            im9 = ax9.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)
            ax9.set_title('时间步相关性')
            plt.colorbar(im9, ax=ax9)
    
    else:  # 非时间序列数据
        # 1. 主要数据显示
        ax1 = plt.subplot(3, 3, 1)
        im1 = ax1.imshow(data_2d, cmap='viridis')
        ax1.set_title('Darcy Flow 解')
        plt.colorbar(im1, ax=ax1)
        
        # 2. 扩散系数
        ax2 = plt.subplot(3, 3, 2)
        if nu is not None:
            if len(nu.shape) == 3:
                nu_2d = nu[:, :, 0]
            else:
                nu_2d = nu
            im2 = ax2.imshow(nu_2d, cmap='plasma')
            ax2.set_title('扩散系数 ν')
            plt.colorbar(im2, ax=ax2)
        else:
            ax2.text(0.5, 0.5, '无扩散系数数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('扩散系数')
        
        # 3. 梯度分析
        ax3 = plt.subplot(3, 3, 3)
        grad_y, grad_x = np.gradient(data_2d)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        im3 = ax3.imshow(grad_magnitude, cmap='hot')
        ax3.set_title('梯度幅值')
        plt.colorbar(im3, ax=ax3)
        
        # 4. 数据分布
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist(data_2d.flatten(), bins=50, alpha=0.7, color='blue', label='解')
        if nu is not None:
            ax4.hist(nu_2d.flatten(), bins=50, alpha=0.7, color='red', label='扩散系数')
            ax4.legend()
        ax4.set_title('数据分布')
        ax4.set_xlabel('数值')
        ax4.set_ylabel('频次')
        
        # 5. 水平剖面
        ax5 = plt.subplot(3, 3, 5)
        middle_row = data_2d.shape[0] // 2
        ax5.plot(data_2d[middle_row, :], label='解')
        if nu is not None:
            ax5.plot(nu_2d[middle_row, :], label='扩散系数')
            ax5.legend()
        ax5.set_title(f'水平剖面 (行 {middle_row})')
        ax5.set_xlabel('列索引')
        ax5.set_ylabel('数值')
        
        # 6. 垂直剖面
        ax6 = plt.subplot(3, 3, 6)
        middle_col = data_2d.shape[1] // 2
        ax6.plot(data_2d[:, middle_col], label='解')
        if nu is not None:
            ax6.plot(nu_2d[:, middle_col], label='扩散系数')
            ax6.legend()
        ax6.set_title(f'垂直剖面 (列 {middle_col})')
        ax6.set_xlabel('行索引')
        ax6.set_ylabel('数值')
        
        # 7. 等高线图
        ax7 = plt.subplot(3, 3, 7)
        contour = ax7.contour(data_2d, levels=10)
        ax7.clabel(contour, inline=True, fontsize=8)
        ax7.set_title('等高线图')
        
        # 8. 3D表面图（投影）
        ax8 = plt.subplot(3, 3, 8)
        x = np.arange(data_2d.shape[1])
        y = np.arange(data_2d.shape[0])
        X, Y = np.meshgrid(x, y)
        ax8.contourf(X, Y, data_2d, levels=20, cmap='viridis')
        ax8.set_title('填充等高线图')
        
        # 9. 统计信息
        ax9 = plt.subplot(3, 3, 9)
        stats_text = f"解统计:\n"
        stats_text += f"最小值: {data_2d.min():.6f}\n"
        stats_text += f"最大值: {data_2d.max():.6f}\n"
        stats_text += f"均值: {data_2d.mean():.6f}\n"
        stats_text += f"标准差: {data_2d.std():.6f}\n"
        if nu is not None:
            stats_text += f"\n扩散系数统计:\n"
            stats_text += f"最小值: {nu_2d.min():.6f}\n"
            stats_text += f"最大值: {nu_2d.max():.6f}\n"
            stats_text += f"均值: {nu_2d.mean():.6f}\n"
            stats_text += f"标准差: {nu_2d.std():.6f}"
        ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax9.set_title('统计信息')
        ax9.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"详细分析图已保存到: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='2D Darcy Flow 数据可视化')
    parser.add_argument('--file_path', type=str, 
                       default='PDEBench/pdebench/data_download/2D_DarcyFlow_beta0.1_Train.hdf5',
                       help='HDF5文件路径')
    parser.add_argument('--sample_idx', type=int, default=0, help='样本索引')
    parser.add_argument('--output_dir', type=str, default='.', help='输出目录')
    parser.add_argument('--create_animation', action='store_true', help='创建动画')
    parser.add_argument('--detailed_analysis', action='store_true', help='创建详细分析')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    print(f"正在加载文件: {file_path}")
    
    try:
        # 加载数据
        data, nu = load_darcy_data(file_path, args.sample_idx)
        
        # 分析数据统计
        analyze_data_statistics(data, nu)
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 创建静态可视化
        static_path = output_dir / "darcy_flow_static.png"
        create_static_visualization(data, nu, static_path)
        
        # 创建动画（如果请求）
        if args.create_animation:
            anim_path = output_dir / "darcy_flow_animation.gif"
            create_animation(data, anim_path)
        
        # 创建详细分析（如果请求）
        if args.detailed_analysis:
            analysis_path = output_dir / "darcy_flow_analysis.png"
            create_detailed_analysis(data, nu, analysis_path)
        
        print("\n可视化完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()