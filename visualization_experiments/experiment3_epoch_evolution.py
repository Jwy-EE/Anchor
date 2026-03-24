"""
实验三：训练收敛过程中的相位"寻找"瀑布图
展示 offset 随训练 epoch 的演化过程
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
from plot_utils import plot_offset_evolution
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

def load_checkpoint_at_epoch(model, checkpoint_path, epoch_key):
    """加载指定 epoch 的 checkpoint"""
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(SCRIPT_DIR, checkpoint_path)

    if os.path.exists(checkpoint_path) and "path/to/" not in checkpoint_path.replace("\\", "/"):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"  已加载 {epoch_key}: {checkpoint_path}")
        return True
    else:
        print(f"  警告: {epoch_key} checkpoint 不存在，使用当前随机初始化权重: {checkpoint_path}")
        return False

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
            num_samples=max(config['experiment']['exp3']['fixed_sample_index'] + 1, 10),
        )
    else:
        config['data']['data_path'] = data_path
        dataset, dataloader = get_dataloader(config, flag='test')
    
    # 获取固定样本
    sample_index = config['experiment']['exp3']['fixed_sample_index']
    print(f"使用测试集第 {sample_index} 个样本作为固定观察样本...")
    
    # 获取数据样本
    data_iter = iter(dataloader)
    for i in range(sample_index + 1):
        seq_x, seq_y, seq_x_mark, seq_y_mark = next(data_iter)
    
    # 固定样本数据
    fixed_seq_x = seq_x
    fixed_seq_x_mark = seq_x_mark
    fixed_seq_y = seq_y
    fixed_seq_y_mark = seq_y_mark
    
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
    
    # 修补模型以添加 Hook
    model = patch_model_with_hooks(model)
    model.eval()
    
    # 获取要 Hook 的层路径
    layer_path = config['experiment']['exp3']['layer_to_hook']
    
    # 获取 checkpoint 配置
    checkpoint_configs = config['experiment']['exp3']['checkpoints']
    
    # 收集不同 epoch 的 offset 数据
    print("\n收集不同 epoch 的 offset 数据...")
    offset_matrix = []
    epochs = []
    epoch_keys = []
    
    for epoch_key, checkpoint_path in checkpoint_configs.items():
        print(f"\n处理 {epoch_key}...")
        
        # 加载该 epoch 的 checkpoint
        load_checkpoint_at_epoch(model, checkpoint_path, epoch_key)
        
        # 启用 Hook
        enable_hook_for_layer(model, layer_path, enable=True)
        
        # 前向传播（使用固定样本）
        with torch.no_grad():
            output = model(fixed_seq_x, fixed_seq_x_mark, fixed_seq_y, fixed_seq_y_mark)
        
        # 获取可视化数据
        viz_data = get_viz_data_from_layer(model, layer_path)
        
        if viz_data is None:
            print(f"  警告: 未获取到 {epoch_key} 的可视化数据")
            continue
        
        # 提取 offset 数据
        offset_data = viz_data['offset'].numpy()  # (1, L, group, kernel_size)
        
        # 取第一个样本、第一个 group、第一个 kernel 位置的 offset
        offset_slice = offset_data[0, :, 0, 0]  # (L,)
        
        # 添加到矩阵
        offset_matrix.append(offset_slice)
        
        # 解析 epoch 编号
        if epoch_key.startswith('epoch'):
            try:
                epoch_num = int(epoch_key[5:])  # 去掉 'epoch' 前缀
                epochs.append(epoch_num)
                epoch_keys.append(epoch_key)
            except ValueError:
                epochs.append(len(epochs))
                epoch_keys.append(epoch_key)
        else:
            epochs.append(len(epochs))
            epoch_keys.append(epoch_key)
        
        # 清理 Hook 为下一个 epoch 准备
        enable_hook_for_layer(model, layer_path, enable=False)
        
        # 打印该 epoch 的统计信息
        mean_val = np.mean(offset_slice)
        std_val = np.std(offset_slice)
        print(f"  offset 形状: {offset_slice.shape}, 均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
    
    if len(offset_matrix) == 0:
        print("错误: 未获取到任何 epoch 的 offset 数据")
        return
    
    # 转换为 numpy 数组
    offset_matrix = np.array(offset_matrix)  # (num_epochs, L)
    
    print(f"\n成功收集 {len(offset_matrix)} 个 epoch 的 offset 数据")
    print(f"offset 矩阵形状: {offset_matrix.shape}")
    print(f"epochs: {epochs}")
    
    # 绘制图像
    print("\n绘制训练收敛过程中的相位'寻找'瀑布图...")
    save_path = config['experiment']['exp3']['save_path']
    if not os.path.isabs(save_path):
        save_path = os.path.join(SCRIPT_DIR, save_path)
    
    plot_offset_evolution(
        offset_matrix=offset_matrix,
        epochs=epochs,
        save_path=save_path,
        figsize=tuple(config['visualization']['figure_size']),
        dpi=config['visualization']['dpi'],
        plot_3d=False  # 设置为 True 可绘制 3D 瀑布图
    )
    
    # 清理 Hook
    clear_all_hooks(model)
    
    # 打印信息
    print("\n实验三完成!")
    print(f"图像已保存到: {save_path}")
    print("\n图像说明:")
    print("1. 热力图的 X 轴: 时间步 (0~L-1)")
    print("2. 热力图的 Y 轴: 训练 Epoch (从上到下递增)")
    print("3. 颜色: offset 值 (蓝色=负偏移, 红色=正偏移)")
    print("4. 白色等高线: offset 的等高线，显示模式形成过程")
    print("\n预期现象:")
    print("- Epoch 0: offset 呈现随机噪声模式，无明显规律")
    print("- Epoch 5~15: offset 开始出现周期性模式，但仍有噪声")
    print("- Epoch 30: offset 形成清晰的周期性模式，与信号周期对齐")
    print("\n这验证了模型'慢慢找到'真实相位偏移的学习过程")

if __name__ == "__main__":
    main()