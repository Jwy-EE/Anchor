"""
使用weather数据集（或ETTh1作为替代）训练ANCHOR模型
"""
import os
import sys
import copy
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加Time-Series-Library到路径
lib_root = os.path.join(PROJECT_ROOT, "Time-Series-Library")
if lib_root not in sys.path:
    sys.path.insert(0, lib_root)

from data_provider.data_factory import data_provider
import importlib.util

def load_model_module(module_path, class_name):
    spec = importlib.util.spec_from_file_location("model_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def save_checkpoint(model, save_path, epoch, train_loss=None, val_loss=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        save_path,
    )

def evaluate(model, loader, device, criterion, pred_len):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in loader:
            seq_x = seq_x.float().to(device)  # 转换为float32
            seq_y = seq_y.float().to(device)
            seq_x_mark = seq_x_mark.float().to(device)
            seq_y_mark = seq_y_mark.float().to(device)

            pred = model(seq_x, seq_x_mark, seq_y, seq_y_mark)
            # seq_y的形状是 [batch_size, label_len + pred_len, features]
            # pred的形状是 [batch_size, pred_len, features]
            # 只比较pred和seq_y的后pred_len部分
            loss = criterion(pred, seq_y[:, -pred_len:, :])

            batch_size = seq_x.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ETTh1', help='数据集类型')
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--use_hooks', action='store_true', help='使用hook修补模型')
    parser.add_argument('--experiment', action='store_true', help='启用相位补偿实验')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model_module_path = os.path.join(
        PROJECT_ROOT,
        "Time-Series-Library",
        "models",
        "uni_fft_1D_forecast_ascending_order.py",
    )
    ModelClass = load_model_module(model_module_path, "Model")
    
    # 根据数据集设置参数
    if args.data == 'ETTh1':
        enc_in = 7
        data_path = 'ETTh1.csv'
        root_path = './dataset/ETT-small/'
    elif args.data == 'custom':  # weather
        enc_in = 21
        data_path = 'weather.csv'
        root_path = './dataset/weather/'
    else:
        raise ValueError(f"不支持的数据集: {args.data}")
    
    model_config = {
        "task_name": "long_term_forecast",
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "enc_in": enc_in,
        "dropout": 0.1,
        "depths": [2, 2, 4, 2],
        "dims": [32, 64, 128, 256],
        "drop_path": 0.0,
    }
    
    model = ModelClass(SimpleNamespace(**model_config)).to(device)
    
    # 使用hook修补模型
    if args.use_hooks:
        from model_hook import patch_model_with_hooks
        model = patch_model_with_hooks(model)
        print("[关键修复] 已调用 patch_model_with_hooks()，确保训练和可视化使用相同的 DCN 实现")
    
    # 实验配置：启用相位补偿实验
    if args.experiment:
        print("[实验配置] 启用相位补偿控制变量实验")
        # 强制锚定第一阶段的 DCN 膨胀系数为 5
        model.anchor_dilation_for_experiment(stage_idx=0, dilation_values=[5, 5, 5])
    
    # 创建args对象用于数据加载
    data_args = SimpleNamespace()
    data_args.task_name = 'long_term_forecast'
    data_args.data = args.data
    data_args.root_path = root_path
    data_args.data_path = data_path
    data_args.features = 'M'  # 多变量预测多变量
    data_args.target = 'OT'
    data_args.freq = 'h'
    data_args.seq_len = args.seq_len
    data_args.label_len = args.seq_len // 2  # 使用一半作为label_len
    data_args.pred_len = args.pred_len
    data_args.enc_in = enc_in
    data_args.dec_in = enc_in
    data_args.c_out = enc_in
    data_args.batch_size = args.batch_size
    data_args.num_workers = 0
    data_args.embed = 'timeF'
    data_args.augmentation_ratio = 0
    data_args.seasonal_patterns = None
    
    print(f"使用数据集: {args.data}")
    print(f"特征数量: {enc_in}")
    print(f"序列长度: {args.seq_len}, 预测长度: {args.pred_len}")
    
    # 加载数据
    train_dataset, train_loader = data_provider(data_args, flag='train')
    val_dataset, val_loader = data_provider(data_args, flag='val')
    
    print(f"训练集大小: {len(train_dataset)}, 批次: {len(train_loader)}")
    print(f"验证集大小: {len(val_dataset)}, 批次: {len(val_loader)}")
    
    # 训练设置
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    checkpoints_dir = os.path.join(SCRIPT_DIR, "checkpoints_weather")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    save_epochs = {0, 5, 10, 15, 30}
    
    # 保存初始checkpoint
    initial_ckpt = os.path.join(checkpoints_dir, f"checkpoint_epoch0_{args.data}.pth")
    save_checkpoint(model, initial_ckpt, epoch=0, train_loss=None, val_loss=None)
    print(f"已保存初始 checkpoint: {initial_ckpt}")
    
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        
        for seq_x, seq_y, seq_x_mark, seq_y_mark in train_loader:
            seq_x = seq_x.float().to(device)  # 转换为float32
            seq_y = seq_y.float().to(device)
            seq_x_mark = seq_x_mark.float().to(device)
            seq_y_mark = seq_y_mark.float().to(device)
            
            optimizer.zero_grad()
            pred = model(seq_x, seq_x_mark, seq_y, seq_y_mark)
            # seq_y的形状是 [batch_size, label_len + pred_len, features]
            # pred的形状是 [batch_size, pred_len, features]
            # 只比较pred和seq_y的后pred_len部分
            loss = criterion(pred, seq_y[:, -args.pred_len:, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_size = seq_x.shape[0]
            running_loss += loss.item() * batch_size
            sample_count += batch_size
        
        train_loss = running_loss / max(sample_count, 1)
        val_loss = evaluate(model, val_loader, device, criterion, args.pred_len)
        
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        
        if epoch in save_epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_epoch{epoch}_{args.data}.pth")
            save_checkpoint(model, ckpt_path, epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            print(f"已保存 checkpoint: {ckpt_path}")
    
    # 保存最佳模型
    best_path = os.path.join(checkpoints_dir, f"checkpoint_best_{args.data}.pth")
    model.load_state_dict(best_state)
    save_checkpoint(model, best_path, epoch=args.epochs, train_loss=None, val_loss=best_val)
    print(f"已保存最佳 checkpoint: {best_path}")
    print("训练完成。")
    
    # 分析offset结果
    if args.use_hooks and args.experiment:
        print("\n分析offset结果...")
        from analyze_offset_results import analyze_offset_evolution
        # 创建结果目录
        results_dir = os.path.join(SCRIPT_DIR, "results_weather")
        analyze_offset_evolution(checkpoints_dir, results_dir)

if __name__ == "__main__":
    main()