# ANCHOR 模型表征可视化实验代码

本文件夹包含用于验证 ANCHOR 模型核心机制（频域先验引导时域自适应）的三个可视化实验代码。

## 实验概述

基于技术文档 `ANCHOR_Representation_Visualization_Design.txt`，本代码实现以下三个实验：

1. **实验一：采样点动态补偿轨迹图** - 展示 DCN 的 offset 如何补偿 FFT 离散周期的相位误差
2. **实验二：多尺度频段的自适应感受野解耦图** - 展示不同 kernel 大小分支的 offset 分布差异
3. **实验三：训练收敛过程中的相位"寻找"瀑布图** - 展示 offset 随训练 epoch 的演化过程

## 文件结构

```
visualization_experiments/
├── README.md                    # 本文件
├── requirements.txt             # 依赖包
├── config.yaml                  # 配置文件
├── data_loader.py               # 数据加载与预处理
├── model_hook.py                # 模型 Hook 实现
├── experiment1_sampling_grid.py # 实验一主脚本
├── experiment2_multi_scale.py   # 实验二主脚本
├── experiment3_epoch_evolution.py # 实验三主脚本
├── plot_utils.py                # 绘图工具函数
└── results/                     # 输出图像目录
```

## 使用方法

1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `config.yaml`：设置模型路径、数据路径等
3. 运行实验：
   ```bash
   python experiment1_sampling_grid.py
   python experiment2_multi_scale.py
   python experiment3_epoch_evolution.py
   ```

## 依赖

- PyTorch >= 1.10
- matplotlib >= 3.5
- numpy >= 1.21
- yaml

## 注意事项

- 需要已训练好的 ANCHOR 模型 checkpoint
- 需要 ETTh1 或其他时间序列数据集
- Hook 机制会临时修改模型 forward 函数，但不会影响原始权重
