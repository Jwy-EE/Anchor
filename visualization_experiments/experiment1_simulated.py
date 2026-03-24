"""
实验一（模拟数据版）：采样点动态补偿轨迹图
使用模拟的复杂波形验证代码功能
"""
import sys
import os
from types import SimpleNamespace
import torch
import numpy as np
import yaml

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_hook import patch_model_with_hooks, enable_hook_for_layer, get_viz_data_from_layer, clear_all_hooks
from plot_utils import plot_sampling_grid
from simulate_data import get_simulated_dataloader, generate_complex_waveform

# 导入 ANCHOR 模型
import importlib.util
import sys

def load_model_module(module_path, class_name):
    """动态加载模型模块"""
    spec = importlib.util.spec_from_file_location("model_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def create_dummy_checkpoint(model, save_path="dummy_checkpoint.pth"):
    """创建虚拟的checkpoint文件（随机权重）"""
    torch.save(model.state_dict(), save_path)
    print(f"已创建虚拟checkpoint: {save_path}")
    return save_path

def main():
    # 创建结果目录
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载模拟数据
    print("加载模拟数据...")
    dataset, dataloader = get_simulated_dataloader(
        seq_len=96,
        pred_len=48,
        batch_size=1,
        num_samples=10
    )
    
    # 动态导入模型类
    model_module_path = os.path.join(
        PROJECT_ROOT,
        "Time-Series-Library",
        "models",
        "uni_fft_1D_forecast_ascending.py",
    )
    
    if not os.path.exists(model_module_path):
        print(f"错误: 模型文件不存在: {model_module_path}")
        print("尝试使用简化模型...")
        # 创建一个简化模型用于演示
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 简单的前向传播，返回输入
                self.dummy = nn.Linear(1, 1)
            
            def forward(self, x, x_mark, y, y_mark):
                # 返回输入作为输出（简化）
                return x
        
        model = SimpleModel()
        print("使用简化模型（仅用于演示代码结构）")
    else:
        # 加载真实模型类
        ModelClass = load_model_module(model_module_path, "Model")
        
        # 创建模型实例（使用默认配置）
        model_config = {
            'task_name': 'short_term_forecast',
            'seq_len': 96,
            'pred_len': 48,
            'enc_in': 7,
            'dropout': 0.1,
            'depths': [2, 2, 8, 2],
            'dims': [64, 128, 256, 512],
            'drop_path': 0.0
        }
        
        try:
            model = ModelClass(SimpleNamespace(**model_config))
            print("已加载ANCHOR模型结构")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("使用简化模型...")
            import torch.nn as nn
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.dummy = nn.Linear(1, 1)
                
                def forward(self, x, x_mark, y, y_mark):
                    return x
            
            model = SimpleModel()
    
    # 创建虚拟checkpoint（随机权重）
    checkpoint_path = "dummy_checkpoint.pth"
    create_dummy_checkpoint(model, checkpoint_path)
    
    # 加载虚拟checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint)
    print(f"已加载虚拟checkpoint: {checkpoint_path}")
    
    # 修补模型以添加 Hook
    print("修补模型以添加 Hook...")
    try:
        model = patch_model_with_hooks(model)
    except Exception as e:
        print(f"修补模型失败（可能是简化模型）: {e}")
        print("跳过Hook修补，使用模拟数据演示...")
    
    model.eval()
    
    # 获取要 Hook 的层路径（假设的路径）
    layer_path = "stages.0.0.rfa.a1.2"
    
    # 尝试启用指定层的 Hook
    try:
        enable_hook_for_layer(model, layer_path, enable=True)
        print(f"已启用层 {layer_path} 的 Hook")
    except Exception as e:
        print(f"启用Hook失败: {e}")
        print("将生成模拟的可视化数据...")
    
    # 获取样本
    sample_index = 0
    print(f"使用测试集第 {sample_index} 个样本...")
    
    # 获取数据样本
    data_iter = iter(dataloader)
    for i in range(sample_index + 1):
        seq_x, seq_y, seq_x_mark, seq_y_mark = next(data_iter)
    
    # 前向传播（触发 Hook）
    print("运行前向传播...")
    with torch.no_grad():
        try:
            output = model(seq_x, seq_x_mark, seq_y, seq_y_mark)
            print("前向传播成功")
        except Exception as e:
            print(f"前向传播失败（可能是简化模型）: {e}")
            print("生成模拟的可视化数据...")
            output = seq_x  # 使用输入作为输出
    
    # 尝试获取可视化数据
    print("提取可视化数据...")
    try:
        viz_data = get_viz_data_from_layer(model, layer_path)
        
        if viz_data is None:
            print("未获取到真实的可视化数据，生成模拟数据...")
            # 生成模拟的可视化数据
            L = seq_x.shape[1]
            kernel_size = 7
            
            # 生成基准网格（基于FFT的离散周期）
            base_grid = np.zeros((1, L, 1, kernel_size))
            for i in range(L):
                for k in range(kernel_size):
                    # 模拟离散的FFT基准点（有误差）
                    base_grid[0, i, 0, k] = i + (k - kernel_size//2) * 12 + np.random.uniform(-0.5, 0.5)
            
            # 生成offset（模拟DCN的连续补偿）
            offset = np.zeros((1, L, 1, kernel_size))
            for i in range(L):
                for k in range(kernel_size):
                    # 模拟连续的offset补偿（正弦波形式）
                    phase_compensation = 0.3 * np.sin(2 * np.pi * i / 24)
                    offset[0, i, 0, k] = phase_compensation + np.random.uniform(-0.1, 0.1)
            
            # 计算实际采样位置
            sampling_locations = base_grid + offset * 1.0  # offset_scale=1.0
            
            viz_data = {
                'base_grid': torch.FloatTensor(base_grid),
                'offset': torch.FloatTensor(offset),
                'sampling_locations': torch.FloatTensor(sampling_locations)
            }
            print("已生成模拟的可视化数据")
    except Exception as e:
        print(f"获取可视化数据失败: {e}")
        return
    
    # 提取数据
    base_grid = viz_data['base_grid'].numpy()
    sampling_locations = viz_data['sampling_locations'].numpy()
    
    # 获取原始信号
    original_signal = seq_x.numpy().squeeze()  # (seq_len, enc_in)
    
    # 选择第一个通道进行可视化
    channel_idx = 0
    signal_to_plot = original_signal[:, channel_idx]
    
    # 绘制图像
    print("绘制采样点动态补偿轨迹图...")
    save_path = "./results/exp1_sampling_grid_simulated.png"
    
    plot_sampling_grid(
        original_signal=signal_to_plot,
        base_grid=base_grid,
        sampling_locations=sampling_locations,
        save_path=save_path,
        figsize=(15, 8),
        dpi=150
    )
    
    # 清理
    try:
        clear_all_hooks(model)
    except:
        pass
    
    # 打印信息
    print("\n实验一（模拟数据版）完成!")
    print(f"图像已保存到: {save_path}")
    print("\n注意: 这是使用模拟数据生成的演示图")
    print("真实实验中需要:")
    print("1. 真实的训练好的ANCHOR模型checkpoint")
    print("2. 真实的ETTh1或其他时间序列数据")
    print("3. 正确的层路径配置")
    
    # 显示模拟信号的特性
    print("\n模拟信号特性:")
    print(f"- 序列长度: {len(signal_to_plot)}")
    print(f"- 信号范围: [{signal_to_plot.min():.3f}, {signal_to_plot.max():.3f}]")
    print(f"- 信号均值: {signal_to_plot.mean():.3f}")
    print(f"- 信号标准差: {signal_to_plot.std():.3f}")

if __name__ == "__main__":
    main()