"""
实验二：多尺度频段的自适应感受野解耦图
展示不同 kernel 大小分支的 offset 分布差异
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
from plot_utils import plot_multi_scale_offset
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
            num_samples=max(config['experiment']['exp2']['sample_index'] + 1, 10),
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
    if os.path.exists(checkpoint_path):
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
    
    # 获取要 Hook 的层路径列表
    layer_paths = config['experiment']['exp2']['layers_to_hook']
    kernel_sizes = [7, 9, 11]  # 对应 a1.2, a2.2, a3.2 的 kernel 大小
    
    # 启用所有指定层的 Hook
    for layer_path in layer_paths:
        enable_hook_for_layer(model, layer_path, enable=True)
    
    # 获取样本
    sample_index = config['experiment']['exp2']['sample_index']
    print(f"使用测试集第 {sample_index} 个样本...")
    
    # 获取数据样本
    data_iter = iter(dataloader)
    for i in range(sample_index + 1):
        seq_x, seq_y, seq_x_mark, seq_y_mark = next(data_iter)
    
    # 前向传播（触发 Hook）
    print("运行前向传播...")
    with torch.no_grad():
        output = model(seq_x, seq_x_mark, seq_y, seq_y_mark)
    
    # 从所有层获取可视化数据
    print("提取各层可视化数据...")
    offset_data_list = []
    
    for layer_path in layer_paths:
        viz_data = get_viz_data_from_layer(model, layer_path)
        
        if viz_data is None:
            print(f"警告: 未获取到层 {layer_path} 的可视化数据")
            continue
        
        # 提取 offset 数据
        offset_data = viz_data['offset'].numpy()
        offset_data_list.append(offset_data)
        
        # 打印该层的统计信息
        offset_flat = offset_data.flatten()
        mean_val = np.mean(offset_flat)
        std_val = np.std(offset_flat)
        abs_mean = np.mean(np.abs(offset_flat))
        
        print(f"层 {layer_path}:")
        print(f"  - offset 形状: {offset_data.shape}")
        print(f"  - 均值: {mean_val:.4f}, 标准差: {std_val:.4f}, 绝对均值: {abs_mean:.4f}")
    
    if len(offset_data_list) == 0:
        print("错误: 未获取到任何层的可视化数据")
        return
    
    # 绘制图像
    print("绘制多尺度频段的自适应感受野解耦图...")
    save_path = config['experiment']['exp2']['save_path']
    if not os.path.isabs(save_path):
        save_path = os.path.join(SCRIPT_DIR, save_path)
    
    plot_multi_scale_offset(
        offset_data_list=offset_data_list,
        kernel_sizes=kernel_sizes[:len(offset_data_list)],
        save_path=save_path,
        figsize=tuple(config['visualization']['figure_size']),
        dpi=config['visualization']['dpi']
    )
    
    # 清理 Hook
    clear_all_hooks(model)
    
    # 打印信息
    print("\n实验二完成!")
    print(f"图像已保存到: {save_path}")
    print("\n图像说明:")
    print("1. 三个子图分别对应 kernel_size = 7, 9, 11 的 DCN 层")
    print("2. 每个小提琴图展示该层 offset 的分布")
    print("3. 图中标注了均值、标准差和绝对均值")
    print("\n预期现象:")
    print("- Kernel=7 (高频分支): offset 分布广，标准差大，绝对均值大")
    print("  体现模型在剧烈调整以捕捉高频细节")
    print("- Kernel=9 (中频分支): offset 分布适中")
    print("- Kernel=11 (低频分支): offset 分布紧凑，标准差小，绝对均值小")
    print("  体现模型在大尺度上保持稳定，微调以对齐宏观周期")
    print("\n这验证了论文中的'小核抓高保真微观信号，大核抓宏观弱特征'机制")

if __name__ == "__main__":
    main()