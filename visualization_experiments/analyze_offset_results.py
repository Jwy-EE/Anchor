"""
分析相位补偿实验的 DCN offset 结果
验证 FGDM 是否成功打破了"零偏移"的诅咒
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加 Time-Series-Library 到路径
LIB_ROOT = os.path.join(PROJECT_ROOT, "Time-Series-Library")
if LIB_ROOT not in sys.path:
    sys.path.insert(0, LIB_ROOT)

import importlib.util

def load_model_module(module_path, class_name):
    spec = importlib.util.spec_from_file_location("model_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def load_checkpoint(checkpoint_path, device='cuda'):
    """加载训练好的模型checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def analyze_offset_evolution(checkpoints_dir, output_dir="results"):
    """
    分析不同epoch的offset变化，验证相位补偿效果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型类
    model_module_path = os.path.join(
        PROJECT_ROOT,
        "Time-Series-Library",
        "models",
        "uni_fft_1D_forecast_ascending_order.py",
    )
    ModelClass = load_model_module(model_module_path, "Model")
    
    # 模型配置
    model_config = {
        "task_name": "short_term_forecast",
        "seq_len": 96,
        "pred_len": 48,
        "enc_in": 7,
        "dropout": 0.1,
        "depths": [2, 2, 4, 2],
        "dims": [32, 64, 128, 256],
        "drop_path": 0.0,
    }
    
    # 修补模型（使用带Hook的版本）
    from model_hook import patch_model_with_hooks
    model = ModelClass(SimpleNamespace(**model_config)).to(device)
    model = patch_model_with_hooks(model)
    
    # 启用特定层的Hook（我们关注stage 0的第一个DCN层）
    from model_hook import enable_hook_for_layer
    target_layer = "stages.0.0.rfa.a1.2"  # stage 0, block 0, rfa, a1[2] (第一个DCN层)
    enable_hook_for_layer(model, target_layer, enable=True)
    
    # 获取checkpoint文件
    checkpoint_files = []
    for fname in os.listdir(checkpoints_dir):
        # 只处理以checkpoint_epoch开头且以.pth结尾的文件
        if fname.startswith("checkpoint_epoch") and fname.endswith(".pth"):
            # 移除可能的额外后缀（如.pthX）
            clean_fname = fname.replace(".pthX", ".pth").replace(".pth", "")
            # 提取epoch数字
            try:
                # 格式: checkpoint_epoch{数字}
                epoch_str = clean_fname.split("_")[1]  # "epoch{数字}"
                epoch = int(epoch_str.replace("epoch", ""))
                checkpoint_files.append((epoch, os.path.join(checkpoints_dir, fname)))
            except (IndexError, ValueError) as e:
                print(f"警告: 无法解析文件名 {fname}: {e}")
                continue
    
    # 按epoch排序
    checkpoint_files.sort(key=lambda x: x[0])
    
    print(f"找到 {len(checkpoint_files)} 个checkpoint文件")
    
    # 准备测试数据
    from simulate_data import SimulatedDataset
    dataset = SimulatedDataset(
        seq_len=96,
        pred_len=48,
        num_samples=1,  # 只需要一个样本
        num_channels=7,
        use_phase_shift=True  # 使用T=5.2的数据
    )
    
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
    seq_x = seq_x.unsqueeze(0).to(device)  # 添加batch维度
    
    # 存储每个epoch的offset数据
    all_offsets = []
    epochs = []
    
    for epoch, checkpoint_path in checkpoint_files:
        print(f"分析 epoch {epoch}...")
        
        # 加载checkpoint
        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        
        # 前向传播（会触发Hook保存数据）
        with torch.no_grad():
            _ = model(seq_x, None, None, None)
        
        # 获取可视化数据
        from model_hook import get_viz_data_from_layer
        viz_data = get_viz_data_from_layer(model, target_layer)
        
        if viz_data is not None:
            offset = viz_data['offset']  # shape: (1, L, group, kernel_size)
            # 取第一个batch，第一个group，第一个kernel的offset
            offset_1d = offset[0, :, 0, 0].numpy()  # shape: (L,)
            
            all_offsets.append(offset_1d)
            epochs.append(epoch)
            
            print(f"  Epoch {epoch}: offset范围 [{offset_1d.min():.4f}, {offset_1d.max():.4f}], 均值: {offset_1d.mean():.4f}")
        else:
            print(f"  Epoch {epoch}: 未获取到可视化数据")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制offset随epoch的变化
    if len(all_offsets) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 不同epoch的offset对比
        ax1 = axes[0, 0]
        for i, (epoch, offset) in enumerate(zip(epochs, all_offsets)):
            ax1.plot(offset, label=f'Epoch {epoch}', alpha=0.7)
        ax1.set_xlabel('时间步 (Time Step)')
        ax1.set_ylabel('Offset 值')
        ax1.set_title('不同训练阶段的 DCN Offset 对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. offset的统计特性随epoch的变化
        ax2 = axes[0, 1]
        offset_means = [o.mean() for o in all_offsets]
        offset_stds = [o.std() for o in all_offsets]
        offset_maxs = [o.max() for o in all_offsets]
        offset_mins = [o.min() for o in all_offsets]
        
        ax2.plot(epochs, offset_means, 'o-', label='均值', linewidth=2)
        ax2.fill_between(epochs, 
                        [m - s for m, s in zip(offset_means, offset_stds)],
                        [m + s for m, s in zip(offset_means, offset_stds)],
                        alpha=0.3, label='±标准差')
        ax2.plot(epochs, offset_maxs, 's--', label='最大值', alpha=0.7)
        ax2.plot(epochs, offset_mins, '^--', label='最小值', alpha=0.7)
        ax2.set_xlabel('训练轮数 (Epoch)')
        ax2.set_ylabel('Offset 统计值')
        ax2.set_title('Offset 统计特性随训练轮数的变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 最终epoch的offset详细分析
        ax3 = axes[1, 0]
        final_offset = all_offsets[-1]
        time_steps = np.arange(len(final_offset))
        
        ax3.plot(time_steps, final_offset, 'b-', linewidth=2, label='Offset轨迹')
        
        # 计算理论相位补偿线（每5.2个时间步补偿0.2）
        theoretical_phase = []
        for t in time_steps:
            # 每5.2个时间步累积0.2的相位差
            phase_accum = (t % 5.2) * (0.2 / 5.2)
            theoretical_phase.append(phase_accum)
        
        ax3.plot(time_steps, theoretical_phase, 'r--', linewidth=1.5, 
                label='理论相位补偿 (每5.2步补偿0.2)')
        
        ax3.set_xlabel('时间步 (Time Step)')
        ax3.set_ylabel('Offset 值')
        ax3.set_title(f'最终模型 (Epoch {epochs[-1]}) 的 Offset 分析')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. offset的直方图
        ax4 = axes[1, 1]
        ax4.hist(final_offset, bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--', label='零线')
        ax4.axvline(x=final_offset.mean(), color='g', linestyle='-', 
                   label=f'均值: {final_offset.mean():.4f}')
        ax4.set_xlabel('Offset 值')
        ax4.set_ylabel('频数')
        ax4.set_title(f'最终模型 Offset 分布 (Epoch {epochs[-1]})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "offset_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"分析图表已保存到: {plot_path}")
        
        # 保存数据
        data_path = os.path.join(output_dir, "offset_data.npz")
        np.savez(data_path, 
                epochs=epochs,
                offsets=all_offsets,
                final_offset=final_offset,
                theoretical_phase=theoretical_phase)
        print(f"数据已保存到: {data_path}")
        
        # 打印关键结论
        print("\n" + "="*60)
        print("相位补偿控制变量实验 - 关键结论")
        print("="*60)
        
        # 检查是否打破了零偏移诅咒
        offset_abs_mean = np.abs(final_offset).mean()
        if offset_abs_mean > 0.01:  # 如果平均绝对值大于0.01，认为打破了零偏移
            print(f"✅ 成功打破了'零偏移'诅咒！")
            print(f"   - Offset平均绝对值: {offset_abs_mean:.4f}")
            print(f"   - Offset范围: [{final_offset.min():.4f}, {final_offset.max():.4f}]")
        else:
            print(f"❌ Offset仍然接近零，可能未成功学习")
            print(f"   - Offset平均绝对值: {offset_abs_mean:.4f}")
        
        # 检查是否显示出周期性模式
        autocorr = np.correlate(final_offset, final_offset, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        # 寻找第一个显著峰值（排除零滞后）
        peak_indices = np.where(autocorr[1:20] > 0.5 * autocorr.max())[0]
        if len(peak_indices) > 0:
            first_peak = peak_indices[0] + 1
            print(f"✅ Offset显示出周期性模式！")
            print(f"   - 自相关第一个显著峰值在滞后 {first_peak} 处")
            print(f"   - 这可能对应着对 5.2 vs 5 相位差的补偿")
        
        # 检查与理论相位补偿的相关性
        correlation = np.corrcoef(final_offset, theoretical_phase)[0, 1]
        print(f"📊 Offset与理论相位补偿的相关性: {correlation:.4f}")
        if correlation > 0.3:
            print(f"   ✅ 显著正相关，模型可能在学习预期的相位补偿")
        elif correlation < -0.3:
            print(f"   ⚠️  显著负相关，模型在学习相反的补偿模式")
        else:
            print(f"   ⚠️  相关性较弱，模型可能在学习其他模式")
        
        print("="*60)
        
        plt.show()
    else:
        print("未获取到有效的offset数据")

if __name__ == "__main__":
    checkpoints_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    output_dir = os.path.join(SCRIPT_DIR, "results")
    
    analyze_offset_evolution(checkpoints_dir, output_dir)