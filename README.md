# ANCHOR: 频域先验与时域自适应白盒表征模型

## 项目概述

ANCHOR（Adaptive Neural Compensation for Harmonic Oscillations and Noise Reduction）是一个创新的时间序列分析框架，结合频域先验知识和时域自适应机制，实现白盒可解释的时间序列建模与预测。

## 核心创新

### 1. 频域先验与时域自适应融合
- **FFT离散周期提取**：通过快速傅里叶变换提取物理周期作为先验知识
- **连续梯度优化**：利用双线性插值的连续可导特性，通过反向传播优化采样点位置
- **相位漂移补偿**：动态补偿FFT整数周期带来的相位误差

### 2. 多尺度解耦感受野
- **正交通道分区**：不同核大小的卷积分支分别处理不同频率成分
- **自适应感受野**：小核捕捉高频微观信号，大核维持低频宏观稳定性
- **解耦特征学习**：各分支独立学习不同时间尺度的模式

## 项目结构

```
Anchor/
├── Time-Series-Library/          # 基础时间序列分析库 (Git子模块)
├── visualization_experiments/    # 可视化实验代码
│   ├── experiment1_sampling_grid.py      # 采样点动态补偿轨迹实验
│   ├── experiment2_multi_scale.py        # 多尺度感受野解耦实验
│   ├── experiment3_epoch_evolution.py    # 训练过程相位演化实验
│   ├── model_hook.py                     # 模型钩子函数
│   ├── data_loader.py                    # 数据加载器
│   ├── plot_utils.py                     # 绘图工具
│   └── results/                          # 实验结果图像
├── ANCHOR_Representation_Visualization_Design.txt  # 技术设计文档
├── README.md                           # 项目说明文档
├── .gitignore                         # Git忽略文件
└── .gitmodules                        # Git子模块配置
```

**注意**: Time-Series-Library 是一个 Git 子模块，指向 https://github.com/Jwy-EE/Time-Series-Library.git

## 关键技术组件

### 1. FFT周期提取 (`fft_seek.py`)
```python
# 提取时间序列的物理周期
periods = extract_periods_fft(time_series, seq_len)
```

### 2. 可变形卷积网络 (`dcnv4_1D.py`)
```python
# 基准采样网格 + 自适应偏移
base_grid = compute_base_grid(dilation, kernel_size)
offset = self.offset_linear(x_feat_reduced)
sampling_locations = base_grid + offset * self.offset_scale
```

### 3. 多尺度架构 (`uni_fft_1D_forecast_ascending.py`)
```python
# 三个并行分支处理不同频率成分
a1 = DCNv3_1D(kernel=7, dilation=d1)  # 高频分支
a2 = DCNv3_1D(kernel=9, dilation=d2)  # 中频分支  
a3 = DCNv3_1D(kernel=11, dilation=d3) # 低频分支
```

## 可视化实验

### 实验1: 采样点动态补偿轨迹
展示FFT整数周期采样点（红点）如何通过自适应偏移（绿星）补偿相位漂移。

### 实验2: 多尺度感受野解耦
通过小提琴图展示不同核大小分支的偏移量分布，验证多尺度解耦效果。

### 实验3: 训练过程相位演化
3D瀑布图展示训练过程中偏移量的演化，直观显示模型"学习"规律的过程。

## 快速开始

### 1. 克隆仓库并初始化子模块
```bash
# 克隆ANCHOR仓库
git clone https://github.com/yourusername/anchor.git
cd anchor

# 初始化并更新子模块
git submodule init
git submodule update
```

### 2. 环境要求
```bash
# 安装基础依赖
pip install -r visualization_experiments/requirements.txt

# 如果需要使用Time-Series-Library
cd Time-Series-Library
pip install -r requirements.txt
```

### 3. 运行可视化实验
```bash
# 实验1: 采样点动态补偿
python visualization_experiments/experiment1_sampling_grid.py

# 实验2: 多尺度感受野解耦  
python visualization_experiments/experiment2_multi_scale.py

# 实验3: 训练过程相位演化
python visualization_experiments/experiment3_epoch_evolution.py
```

## 实验结果

实验结果保存在 `visualization_experiments/results/` 目录中：
- `exp1_sampling_grid.png` - 采样点补偿轨迹
- `exp2_multi_scale.png` - 多尺度偏移量分布
- `exp3_epoch_evolution.png` - 训练过程相位演化

## 技术优势

1. **白盒可解释性**：所有计算过程透明可追溯
2. **物理意义明确**：基于FFT的周期提取具有明确的物理含义
3. **自适应能力强**：通过连续梯度优化实现精准相位补偿
4. **多尺度分析**：同时捕捉微观和宏观时间模式
5. **可视化支持**：完整的实验验证和可视化方案

## 应用场景

- 时间序列预测（电力负荷、股票价格、气象数据）
- 异常检测（工业设备监控、网络安全）
- 信号处理（生物医学信号、通信信号）
- 周期性模式分析（销售数据、用户行为）

## 引用

如果您使用ANCHOR项目，请引用：
```
@software{anchor2024,
  title = {ANCHOR: Adaptive Neural Compensation for Harmonic Oscillations and Noise Reduction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/anchor}
}
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目Issues页面](https://github.com/yourusername/anchor/issues)
- Email: your.email@example.com