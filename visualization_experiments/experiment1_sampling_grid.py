"""
实验一：采样点动态补偿轨迹图
展示 DCN 的 offset 如何补偿 FFT 离散周期的相位误差
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

from data_loader import load_config, get_dataloader
from model_hook import patch_model_with_hooks, enable_hook_for_layer, get_viz_data_from_layer, clear_all_hooks
from plot_utils import plot_sampling_grid
from simulate_data import get_simulated_dataloader

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

def main():
    # 加载配置
    config_path = os.path.join(SCRIPT_DIR, "config.yaml")
    config = load_config(config_path)
    
    # 创建结果目录
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    data_path = config['data']['data_path']
    if not os.path.isabs(data_path):
        data_path = os.path.join(SCRIPT_DIR, data_path)

    use_simulated_data = (
        config['data']['data_path'].startswith("path/to/")
        or not os.path.exists(data_path)
    )

    if use_simulated_data:
        print("警告: 未找到真实数据文件，回退到模拟数据")
        dataset, dataloader = get_simulated_dataloader(
            seq_len=config['model']['configs']['seq_len'],
            pred_len=config['model']['configs']['pred_len'],
            batch_size=1,
            num_samples=max(config['experiment']['exp1']['sample_index'] + 1, 10),
        )
    else:
        config['data']['data_path'] = data_path
        dataset, dataloader = get_dataloader(config, flag='test')
    
    # 加载模型
    print("加载模型...")
    
    # 动态导入模型类
    model_class_name = config['model']['model_class'].split('.')[-1]
    model_module_path = os.path.join(
        PROJECT_ROOT,
        "Time-Series-Library",
        "models",
        "uni_fft_1D_forecast_ascending.py",
    )
    
    if not os.path.exists(model_module_path):
        print(f"错误: 模型文件不存在: {model_module_path}")
        print("请确保 config.yaml 中的 model_class 配置正确")
        return
    
    ModelClass = load_model_module(model_module_path, model_class_name)
    
    # 创建模型实例
    model_config = config['model']['configs']
    model = ModelClass(SimpleNamespace(**model_config))
    
    # 加载预训练权重
    checkpoint_path = config['model']['checkpoint_path']
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(SCRIPT_DIR, checkpoint_path)
    if os.path.exists(checkpoint_path) and "path/to/" not in checkpoint_path.replace("\\", "/"):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"已加载 checkpoint: {checkpoint_path}")
    else:
        print(f"警告: checkpoint 不存在: {checkpoint_path}")
        print("将使用随机初始化的模型")
    
    # 修补模型以添加 Hook
    print("修补模型以添加 Hook...")
    model = patch_model_with_hooks(model)
    model.eval()
    
    # 获取要 Hook 的层路径
    layer_path = config['experiment']['exp1']['layer_to_hook']
    
    # 启用指定层的 Hook
    enable_hook_for_layer(model, layer_path, enable=True)
    
    # 获取样本
    sample_index = config['experiment']['exp1']['sample_index']
    print(f"使用测试集第 {sample_index} 个样本...")
    
    # 获取数据样本
    data_iter = iter(dataloader)
    for i in range(sample_index + 1):
        seq_x, seq_y, seq_x_mark, seq_y_mark = next(data_iter)
    
    # 前向传播（触发 Hook）
    print("运行前向传播...")
    with torch.no_grad():
        output = model(seq_x, seq_x_mark, seq_y, seq_y_mark)
    
    # 获取可视化数据
    print("提取可视化数据...")
    viz_data = get_viz_data_from_layer(model, layer_path)
    
    if viz_data is None:
        print("错误: 未获取到可视化数据")
        print("请检查: 1) 层路径是否正确 2) Hook 是否启用 3) 模型是否已修补")
        return
    
    # 提取数据
    base_grid = viz_data['base_grid'].numpy()
    sampling_locations = viz_data['sampling_locations'].numpy()
    
    # 获取原始信号（逆标准化）
    original_signal = seq_x.numpy().squeeze()  # (seq_len, enc_in)
    
    # 选择第一个通道进行可视化
    channel_idx = 0
    signal_to_plot = original_signal[:, channel_idx]
    
    # 绘制图像
    print("绘制采样点动态补偿轨迹图...")
    save_path = config['experiment']['exp1']['save_path']
    if not os.path.isabs(save_path):
        save_path = os.path.join(SCRIPT_DIR, save_path)
    
    plot_sampling_grid(
        original_signal=signal_to_plot,
        base_grid=base_grid,
        sampling_locations=sampling_locations,
        save_path=save_path,
        figsize=tuple(config['visualization']['figure_size']),
        dpi=config['visualization']['dpi']
    )
    
    # 清理 Hook
    clear_all_hooks(model)
    
    # 打印信息
    print("\n实验一完成!")
    print(f"图像已保存到: {save_path}")
    print("\n图像说明:")
    print("1. 蓝色曲线: 原始时间序列信号")
    print("2. 红色方块: FFT 提取的离散周期基准采样点")
    print("3. 绿色星形: DCN 经过 offset 补偿后的实际采样点")
    print("4. 黑色箭头: 从基准点到实际点的偏移量 (Δp_n)")
    print("\n预期现象:")
    print("- 红色方块可能偏离波峰/波谷（FFT 离散误差）")
    print("- 绿色星形应更精准地对齐波峰/波谷（DCN 连续补偿）")
    print("- 箭头方向显示 offset 的补偿方向")

if __name__ == "__main__":
    main()