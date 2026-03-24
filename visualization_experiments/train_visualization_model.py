"""
为可视化实验训练一个轻量版 ANCHOR 模型，并保存多个 epoch checkpoint。
默认使用 simulate_data.py 中的模拟数据，目标是让 DCN offset 学到非零形态，
从而支撑 experiment1/2/3 的可视化结果。
"""
import os
import sys
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulate_data import SimulatedDataset
import importlib.util


def load_model_module(module_path, class_name):
    # 确保 Time-Series-Library 在路径中，以便模型能导入 layers
    lib_root = os.path.join(PROJECT_ROOT, "Time-Series-Library")
    if lib_root not in sys.path:
        sys.path.insert(0, lib_root)
    
    spec = importlib.util.spec_from_file_location("model_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def build_dataloaders(seq_len=96, pred_len=48, num_samples=320, batch_size=16):
    dataset = SimulatedDataset(
        seq_len=seq_len,
        pred_len=pred_len,
        num_samples=num_samples,
        num_channels=7,
    )

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


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


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in loader:
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)
            seq_x_mark = seq_x_mark.to(device)
            seq_y_mark = seq_y_mark.to(device)

            pred = model(seq_x, seq_x_mark, seq_y, seq_y_mark)
            loss = criterion(pred, seq_y)

            batch_size = seq_x.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_module_path = os.path.join(
        PROJECT_ROOT,
        "Time-Series-Library",
        "models",
        "uni_fft_1D_forecast_ascending.py",
    )
    ModelClass = load_model_module(model_module_path, "Model")

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

    model = ModelClass(SimpleNamespace(**model_config)).to(device)

    train_loader, val_loader = build_dataloaders(
        seq_len=model_config["seq_len"],
        pred_len=model_config["pred_len"],
        num_samples=320,
        batch_size=16,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    checkpoints_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    save_epochs = {0, 5, 10, 15, 30}

    initial_ckpt = os.path.join(checkpoints_dir, "checkpoint_epoch0.pth")
    save_checkpoint(model, initial_ckpt, epoch=0, train_loss=None, val_loss=None)
    print(f"已保存初始 checkpoint: {initial_ckpt}")

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    total_epochs = 30
    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for seq_x, seq_y, seq_x_mark, seq_y_mark in train_loader:
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)
            seq_x_mark = seq_x_mark.to(device)
            seq_y_mark = seq_y_mark.to(device)

            optimizer.zero_grad()
            pred = model(seq_x, seq_x_mark, seq_y, seq_y_mark)
            loss = criterion(pred, seq_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = seq_x.shape[0]
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        train_loss = running_loss / max(sample_count, 1)
        val_loss = evaluate(model, val_loader, device, criterion)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch in save_epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"checkpoint_epoch{epoch}.pth")
            save_checkpoint(model, ckpt_path, epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            print(f"已保存 checkpoint: {ckpt_path}")

    best_path = os.path.join(checkpoints_dir, "checkpoint_best.pth")
    model.load_state_dict(best_state)
    save_checkpoint(model, best_path, epoch=total_epochs, train_loss=None, val_loss=best_val)
    print(f"已保存最佳 checkpoint: {best_path}")
    print("训练完成。")


if __name__ == "__main__":
    main()