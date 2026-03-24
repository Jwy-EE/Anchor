"""
数据加载模块
用于加载ETTh1等时间序列数据集，并进行预处理
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import yaml

class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    
    def __init__(self, data_path, seq_len, pred_len, flag='test', 
                 scale=True, freq='h', target='OT', features='M'):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            seq_len: 输入序列长度
            pred_len: 预测长度
            flag: 数据集类型 ('train', 'val', 'test')
            scale: 是否标准化
            freq: 数据频率
            target: 预测目标列
            features: 特征类型 ('M'=多变量预测多变量, 'S'=单变量预测单变量, 'MS'=多变量预测单变量)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.target = target
        self.features = features
        self.flag = flag
        
        # 加载数据
        self.data, self.data_stamp = self._load_data(data_path, freq)
        
        # 划分数据集
        self._split_data()
        
        # 标准化
        if self.scale:
            self._standardize()
    
    def _load_data(self, data_path, freq):
        """加载CSV数据"""
        df_raw = pd.read_csv(data_path)
        
        # 处理时间戳
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        data_stamp = self._time_features(df_raw['date'], freq=freq)
        
        # 选择特征
        cols = list(df_raw.columns)
        cols.remove('date')
        cols.remove(self.target) if self.target in cols else None
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # 数据值
        data = df_raw[cols + [self.target]].values
        
        return data, data_stamp
    
    def _time_features(self, dates, freq='h'):
        """提取时间特征"""
        dates = pd.to_datetime(dates)
        if freq == 'h':
            time_features = np.stack([
                dates.year, dates.month, dates.day, dates.hour,
                dates.dayofweek, dates.dayofyear
            ], axis=1).astype(np.float32)
        elif freq == 't':
            time_features = np.stack([
                dates.year, dates.month, dates.day, dates.hour, dates.minute,
                dates.dayofweek, dates.dayofyear
            ], axis=1).astype(np.float32)
        else:
            time_features = np.stack([
                dates.year, dates.month, dates.day,
                dates.dayofweek, dates.dayofyear
            ], axis=1).astype(np.float32)
        return time_features
    
    def _split_data(self):
        """划分训练集、验证集、测试集"""
        num_data = len(self.data)
        num_test = int(num_data * 0.2)
        num_val = int(num_data * 0.2)
        num_train = num_data - num_test - num_val
        
        if self.flag == 'test':
            self.data_x = self.data[num_train + num_val:]
            self.data_y = self.data[num_train + num_val:]
            self.data_stamp_x = self.data_stamp[num_train + num_val:]
            self.data_stamp_y = self.data_stamp[num_train + num_val:]
        elif self.flag == 'val':
            self.data_x = self.data[num_train:num_train + num_val]
            self.data_y = self.data[num_train:num_train + num_val]
            self.data_stamp_x = self.data_stamp[num_train:num_train + num_val]
            self.data_stamp_y = self.data_stamp[num_train:num_train + num_val]
        else:  # train
            self.data_x = self.data[:num_train]
            self.data_y = self.data[:num_train]
            self.data_stamp_x = self.data_stamp[:num_train]
            self.data_stamp_y = self.data_stamp[:num_train]
    
    def _standardize(self):
        """标准化数据"""
        self.scaler_mean = self.data_x.mean(axis=0)
        self.scaler_std = self.data_x.std(axis=0) + 1e-8
        self.data_x = (self.data_x - self.scaler_mean) / self.scaler_std
        self.data_y = (self.data_y - self.scaler_mean) / self.scaler_std
    
    def inverse_transform(self, data):
        """逆标准化"""
        if self.scale:
            return data * self.scaler_std + self.scaler_mean
        return data
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        """获取一个样本"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        seq_x_mark = self.data_stamp_x[s_begin:s_end]
        seq_y_mark = self.data_stamp_y[r_begin:r_end]
        
        # 转换为torch tensor
        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)
        seq_x_mark = torch.FloatTensor(seq_x_mark)
        seq_y_mark = torch.FloatTensor(seq_y_mark)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark


def get_dataloader(config, flag='test'):
    """
    获取数据加载器
    
    Args:
        config: 配置字典
        flag: 数据集类型 ('train', 'val', 'test')
    
    Returns:
        DataLoader
    """
    data_config = config['data']
    model_config = config['model']['configs']
    
    dataset = TimeSeriesDataset(
        data_path=data_config['data_path'],
        seq_len=model_config['seq_len'],
        pred_len=model_config['pred_len'],
        flag=flag,
        scale=data_config['scale'],
        freq=data_config['freq'],
        target=data_config['target'],
        features=data_config['features']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 可视化实验通常使用batch_size=1
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    return dataset, dataloader


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # 测试数据加载
    config = load_config("config.yaml")
    dataset, dataloader = get_dataloader(config, flag='test')
    
    print(f"数据集大小: {len(dataset)}")
    print(f"输入形状: {dataset[0][0].shape}")
    print(f"输出形状: {dataset[0][1].shape}")
