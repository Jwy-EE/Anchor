"""
模拟数据生成模块
生成复杂的多频率波形用于验证实验
"""
import numpy as np
import pandas as pd
import torch
import os

def generate_complex_waveform(length=96, num_channels=7, seed=42):
    """
    生成复杂的多频率波形
    
    Args:
        length: 序列长度
        num_channels: 通道数
        seed: 随机种子
    
    Returns:
        data: 形状为 (length, num_channels) 的 numpy 数组
    """
    np.random.seed(seed)
    
    # 时间轴
    t = np.linspace(0, 10, length)
    
    # 生成每个通道的复杂波形
    data = np.zeros((length, num_channels))
    
    for ch in range(num_channels):
        # 基础频率（主周期）
        f_base = 0.5 + ch * 0.1  # 不同通道有不同的基础频率
        
        # 添加多个谐波成分
        waveform = 0
        for harmonic in range(1, 6):  # 5个谐波
            amplitude = 1.0 / harmonic  # 振幅随谐波次数衰减
            frequency = f_base * harmonic
            phase = np.random.uniform(0, 2*np.pi)  # 随机相位
            
            # 添加谐波
            waveform += amplitude * np.sin(2 * np.pi * frequency * t + phase)
        
        # 添加一些噪声
        noise = 0.1 * np.random.randn(length)
        
        # 添加趋势项
        trend = 0.05 * t
        
        # 组合所有成分
        data[:, ch] = waveform + noise + trend
    
    # 归一化
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    return data

def generate_phase_shift_data(length=96, num_channels=7, seed=42):
    """
    生成用于相位补偿控制变量实验的数据
    核心设计：真实物理周期 T=5.2，模型锚点周期 D=5
    
    Args:
        length: 序列长度
        num_channels: 通道数
        seed: 随机种子
    
    Returns:
        data: 形状为 (length, num_channels) 的 numpy 数组
    """
    np.random.seed(seed)
    
    # 时间轴
    t = np.linspace(0, 10, length)
    
    # 生成每个通道的波形
    data = np.zeros((length, num_channels))
    
    for ch in range(num_channels):
        # 真实物理周期：5.2
        T_real = 5.2
        
        # 生成主波形：严格遵循 T=5.2 的正弦波
        main_wave = np.sin(2 * np.pi * t / T_real)
        
        # 添加微弱的二次谐波（保持主周期不变）
        harmonic = 0.2 * np.sin(4 * np.pi * t / T_real + np.pi/4)
        
        # 添加极小的白噪声（信噪比高，确保梯度清晰）
        noise = 0.02 * np.random.randn(length)
        
        # 组合：主波 + 谐波 + 噪声
        waveform = main_wave + harmonic + noise
        
        # 轻微调整不同通道的振幅，但保持相同的周期
        amplitude_scale = 0.8 + 0.4 * (ch / num_channels)
        data[:, ch] = amplitude_scale * waveform
    
    # 归一化
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    print(f"[Phase Shift Data] 生成完成，真实周期 T={T_real}，序列长度={length}")
    print(f"[Phase Shift Data] 数据形状: {data.shape}")
    
    return data

def create_simulated_etth1_csv(save_path="simulated_ETTh1.csv", length=10000):
    """
    创建模拟的ETTh1 CSV文件
    
    Args:
        save_path: 保存路径
        length: 总数据长度
    """
    # 生成数据
    data = generate_complex_waveform(length=length, num_channels=7)
    
    # 创建时间戳
    dates = pd.date_range(start='2016-01-01', periods=length, freq='H')
    
    # 创建列名（模拟ETTh1的7个特征）
    columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, 'date', dates)
    
    # 保存为CSV
    df.to_csv(save_path, index=False)
    print(f"模拟数据已保存到: {save_path}")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
    
    return save_path

class SimulatedDataset:
    """模拟数据集类，用于替代真实数据"""
    
    def __init__(self, seq_len=96, pred_len=48, num_samples=100, num_channels=7, use_phase_shift=False):
        """
        初始化模拟数据集
        
        Args:
            seq_len: 输入序列长度
            pred_len: 预测长度
            num_samples: 样本数量
            num_channels: 通道数
            use_phase_shift: 是否使用相位偏移实验数据（T=5.2）
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.use_phase_shift = use_phase_shift
        
        # 生成所有数据
        self.total_length = (seq_len + pred_len) * num_samples
        
        if use_phase_shift:
            print(f"[SimulatedDataset] 使用相位偏移实验数据 (T=5.2)")
            self.data = generate_phase_shift_data(
                length=self.total_length, 
                num_channels=num_channels,
                seed=42
            )
        else:
            self.data = generate_complex_waveform(
                length=self.total_length, 
                num_channels=num_channels,
                seed=42
            )
        
        # 标准化
        self.scaler_mean = self.data.mean(axis=0)
        self.scaler_std = self.data.std(axis=0) + 1e-8
        self.data = (self.data - self.scaler_mean) / self.scaler_std
        
        # 创建时间特征（模拟）
        self.data_stamp = np.zeros((self.total_length, 6))  # 6个时间特征
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        """获取一个样本"""
        s_begin = index * (self.seq_len + self.pred_len)
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # 转换为torch tensor
        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)
        seq_x_mark = torch.FloatTensor(seq_x_mark)
        seq_y_mark = torch.FloatTensor(seq_y_mark)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def inverse_transform(self, data):
        """逆标准化"""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        return data * self.scaler_std + self.scaler_mean

def get_simulated_dataloader(seq_len=96, pred_len=48, batch_size=1, num_samples=100):
    """
    获取模拟数据加载器
    
    Returns:
        dataset, dataloader
    """
    from torch.utils.data import DataLoader
    
    dataset = SimulatedDataset(
        seq_len=seq_len,
        pred_len=pred_len,
        num_samples=num_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    return dataset, dataloader

if __name__ == "__main__":
    # 测试数据生成
    print("测试模拟数据生成...")
    
    # 生成一小段数据查看
    test_data = generate_complex_waveform(length=200, num_channels=7)
    print(f"生成数据形状: {test_data.shape}")
    print(f"通道1的前10个值: {test_data[:10, 0]}")
    
    # 创建模拟CSV文件
    csv_path = create_simulated_etth1_csv(length=1000)
    
    # 测试数据集
    dataset, dataloader = get_simulated_dataloader(num_samples=10)
    print(f"\n模拟数据集大小: {len(dataset)}")
    
    # 获取一个样本
    seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
    print(f"输入序列形状: {seq_x.shape}")
    print(f"输出序列形状: {seq_y.shape}")