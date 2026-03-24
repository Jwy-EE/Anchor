"""
绘图工具模块
包含三个实验的可视化函数
"""
import matplotlib
# 设置中文字体，避免中文显示为方块
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except:
    pass  # 如果字体不存在，使用默认字体

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_sampling_grid(
    original_signal,
    base_grid,
    sampling_locations,
    save_path=None,
    figsize=(12, 6),
    dpi=150,
    ax=None,
):
    """
    绘制采样点动态补偿轨迹图（实验一）
    
    Args:
        original_signal: 原始信号，形状 (L,) 或 (C, L)
        base_grid: 基准采样网格，形状 (1, L, 1, kernel_size)
        sampling_locations: 实际采样位置，形状 (1, L, group, kernel_size)
        save_path: 保存路径，如果为 None 则显示图像
        figsize: 图像大小
        dpi: 分辨率
        ax: 可选，传入已有坐标轴时绘制到该子图上
    """
    # 处理输入信号
    if original_signal.ndim == 2:
        # 多通道，取第一个通道
        signal = original_signal[0] if original_signal.shape[0] == 1 else original_signal[:, 0]
    else:
        signal = original_signal
    
    L = len(signal)
    
    # 提取数据
    base_grid = base_grid.squeeze()  # (L, kernel_size) 或 (L, 1, kernel_size)
    sampling_locations = sampling_locations.squeeze()  # (L, group, kernel_size)
    
    # 如果还有多余的维度，取第一个 group 和第一个 kernel 位置
    if base_grid.ndim == 3:
        base_grid = base_grid[:, 0, :]  # (L, kernel_size)
    if sampling_locations.ndim == 3:
        sampling_locations = sampling_locations[:, 0, :]  # (L, kernel_size)
    
    # 取中心 kernel 位置（通常是第 (kernel_size//2) 个）
    kernel_size = base_grid.shape[1]
    center_kernel_idx = kernel_size // 2
    
    base_points = base_grid[:, center_kernel_idx]  # (L,)
    sampling_points = sampling_locations[:, center_kernel_idx]  # (L,)
    
    # 创建图形
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 绘制原始信号
    time_axis = np.arange(L)
    ax.plot(time_axis, signal, 'b-', linewidth=1.5, alpha=0.7, label='原始信号')
    
    # 绘制基准采样点（红色方块）
    base_y = np.interp(base_points, time_axis, signal)
    ax.scatter(base_points, base_y, c='red', marker='s', s=50, 
               label='FFT基准采样点', zorder=5)
    
    # 绘制实际采样点（绿色星形）
    sampling_y = np.interp(sampling_points, time_axis, signal)
    ax.scatter(sampling_points, sampling_y, c='green', marker='*', s=100, 
               label='DCN实际采样点', zorder=6)
    
    # 绘制箭头连接基准点和实际点
    for i in range(min(20, L)):  # 只画前20个点避免过于拥挤
        idx = i * (L // 20)
        if idx < L:
            ax.annotate('', xy=(sampling_points[idx], sampling_y[idx]), 
                       xytext=(base_points[idx], base_y[idx]),
                       arrowprops=dict(arrowstyle='->', color='black', 
                                      linestyle='--', alpha=0.7, linewidth=1))
    
    # 设置图形属性
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('信号值', fontsize=12)
    ax.set_title('采样点动态补偿轨迹图\n(红色: FFT离散基准点, 绿色: DCN连续补偿点)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 保存或显示
    if own_figure:
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"图像已保存到: {save_path}")
        else:
            plt.show()
        plt.close(fig)


def plot_multi_scale_offset(
    offset_data_list,
    kernel_sizes,
    save_path=None,
    figsize=(15, 5),
    dpi=150,
    axes=None,
):
    """
    绘制多尺度频段的自适应感受野解耦图（实验二）
    
    Args:
        offset_data_list: 偏移量数据列表，每个元素形状为 (1, L, group, kernel_size)
        kernel_sizes: 对应的核大小列表
        save_path: 保存路径
        figsize: 图像大小
        dpi: 分辨率
        axes: 可选，传入已有坐标轴列表时绘制到这些子图上
    """
    num_plots = len(offset_data_list)
    
    own_figure = axes is None
    if own_figure:
        fig, axes = plt.subplots(1, num_plots, figsize=figsize, dpi=dpi, sharey=True)
        if num_plots == 1:
            axes = [axes]
    
    for idx, (offset_data, kernel_size) in enumerate(zip(offset_data_list, kernel_sizes)):
        ax = axes[idx]
        
        # 提取偏移量数据
        offset = offset_data.squeeze()  # (L, group, kernel_size)
        
        # 展平所有偏移量
        if offset.ndim == 3:
            offset_flat = offset.reshape(-1)  # (L * group * kernel_size,)
        elif offset.ndim == 2:
            offset_flat = offset.reshape(-1)
        else:
            offset_flat = offset.flatten()
        
        # 计算统计信息
        mean_val = np.mean(offset_flat)
        std_val = np.std(offset_flat)
        abs_mean = np.mean(np.abs(offset_flat))
        
        # 绘制小提琴图
        parts = ax.violinplot(offset_flat, showmeans=True, showmedians=True)
        
        # 设置颜色
        for pc in parts['bodies']:
            pc.set_facecolor(plt.cm.viridis(idx / num_plots))
            pc.set_alpha(0.7)
        
        # 添加统计信息文本
        stats_text = f'均值: {mean_val:.3f}\n标准差: {std_val:.3f}\n绝对均值: {abs_mean:.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 设置子图属性
        ax.set_title(f'Kernel Size = {kernel_size}', fontsize=12, fontweight='bold')
        ax.set_ylabel('偏移量 (Offset)', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    if own_figure:
        # 设置整体标题
        fig.suptitle('多尺度频段的自适应感受野解耦图\n(不同 Kernel 大小的 Offset 分布对比)', 
                    fontsize=14, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"图像已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)


def plot_offset_evolution(
    offset_matrix,
    epochs,
    save_path=None,
    figsize=(12, 8),
    dpi=150,
    plot_3d=False,
    ax=None,
):
    """
    绘制训练收敛过程中的相位"寻找"瀑布图（实验三）
    
    Args:
        offset_matrix: 偏移量矩阵，形状 (num_epochs, L)
        epochs: epoch 列表或数组
        save_path: 保存路径
        figsize: 图像大小
        dpi: 分辨率
        plot_3d: 是否绘制3D瀑布图（True）或2D热力图（False）
        ax: 可选，传入已有坐标轴时绘制到该子图上
    """
    num_epochs, L = offset_matrix.shape
    
    own_figure = ax is None
    
    if plot_3d:
        # 3D 瀑布图
        if own_figure:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # 创建网格
        X = np.arange(L)
        Y = np.array(epochs)
        X, Y = np.meshgrid(X, Y)
        Z = offset_matrix
        
        # 绘制3D曲面
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # 添加等高线
        ax.contour(X, Y, Z, 10, offset=Z.min(), cmap=cm.viridis, alpha=0.5)
        
        # 设置坐标轴标签
        ax.set_xlabel('时间步', fontsize=11)
        ax.set_ylabel('训练 Epoch', fontsize=11)
        ax.set_zlabel('偏移量 (Offset)', fontsize=11)
        ax.set_title('训练收敛过程中的相位"寻找"瀑布图 (3D)', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='偏移量')
        
    else:
        # 2D 热力图
        if own_figure:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure
        
        # 绘制热力图
        im = ax.imshow(offset_matrix, aspect='auto', cmap='viridis', 
                      extent=[0, L, epochs[-1], epochs[0]])
        
        # 设置坐标轴
        ax.set_xlabel('时间步', fontsize=12)
        ax.set_ylabel('训练 Epoch', fontsize=12)
        ax.set_title('训练收敛过程中的相位"寻找"热力图', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('偏移量 (Offset)', fontsize=11)
        
        # 添加等高线
        if num_epochs > 1 and L > 1:
            X = np.arange(L)
            Y = np.array(epochs)
            X, Y = np.meshgrid(X, Y)
            ax.contour(X, Y, offset_matrix, 5, colors='white', alpha=0.5, linewidths=0.5)
    
    if own_figure:
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"图像已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)


def create_summary_figure(exp1_data, exp2_data, exp3_data, save_path=None, 
                         figsize=(20, 12), dpi=150):
    """
    创建汇总图，将三个实验的结果放在一张图中
    
    Args:
        exp1_data: 实验一数据 (original_signal, base_grid, sampling_locations)
        exp2_data: 实验二数据 (offset_data_list, kernel_sizes)
        exp3_data: 实验三数据 (offset_matrix, epochs)
        save_path: 保存路径
        figsize: 图像大小
        dpi: 分辨率
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # 子图1：采样点动态补偿轨迹图
    ax1 = plt.subplot(2, 2, 1)
    original_signal, base_grid, sampling_locations = exp1_data
    plot_sampling_grid(
        original_signal,
        base_grid,
        sampling_locations,
        save_path=None,
        figsize=(8, 4),
        dpi=dpi,
        ax=ax1,
    )
    
    # 子图2：多尺度频段的自适应感受野解耦图
    ax2 = plt.subplot(2, 2, 2)
    offset_data_list, kernel_sizes = exp2_data
    plot_multi_scale_offset(
        offset_data_list,
        kernel_sizes,
        save_path=None,
        figsize=(8, 4),
        dpi=dpi,
        axes=[ax2] if len(offset_data_list) == 1 else [ax2] + [fig.add_subplot(2, 2, 2) for _ in range(len(offset_data_list) - 1)],
    )
    
    # 子图3：训练收敛过程中的相位"寻找"瀑布图
    ax3 = plt.subplot(2, 2, (3, 4))
    offset_matrix, epochs = exp3_data
    plot_offset_evolution(
        offset_matrix,
        epochs,
        save_path=None,
        figsize=(10, 6),
        dpi=dpi,
        plot_3d=False,
        ax=ax3,
    )
    
    # 设置整体标题
    fig.suptitle('ANCHOR 模型核心机制表征实验汇总图', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"汇总图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


if __name__ == "__main__":
    # 测试绘图函数
    print("绘图工具模块已加载")
    print("包含以下函数:")
    print("1. plot_sampling_grid() - 实验一：采样点动态补偿轨迹图")
    print("2. plot_multi_scale_offset() - 实验二：多尺度频段解耦图")
    print("3. plot_offset_evolution() - 实验三：训练收敛瀑布图")
    print("4. create_summary_figure() - 三个实验汇总图")