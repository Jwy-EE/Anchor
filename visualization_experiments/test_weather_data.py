"""
测试weather数据集是否可以正常加载
"""
import os
import sys
from types import SimpleNamespace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 添加Time-Series-Library到路径
lib_root = os.path.join(PROJECT_ROOT, "Time-Series-Library")
if lib_root not in sys.path:
    sys.path.insert(0, lib_root)

from data_provider.data_factory import data_provider

def test_weather_data():
    # 创建模拟的args对象
    args = SimpleNamespace()
    args.task_name = 'long_term_forecast'
    args.data = 'custom'
    args.root_path = './dataset/weather/'
    args.data_path = 'weather.csv'
    args.features = 'MS'  # 多变量预测单变量
    args.target = 'OT'  # 目标特征
    args.freq = 'h'  # 小时频率
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    args.enc_in = 21  # weather数据集有21个特征
    args.dec_in = 21
    args.c_out = 21
    args.batch_size = 32
    args.num_workers = 0
    args.embed = 'timeF'
    args.augmentation_ratio = 0
    args.seasonal_patterns = None  # 添加缺失的属性
    
    print("测试加载weather数据集...")
    
    try:
        # 尝试加载训练数据
        train_dataset, train_loader = data_provider(args, flag='train')
        print(f"训练数据集大小: {len(train_dataset)}")
        print(f"训练数据加载器批次数量: {len(train_loader)}")
        
        # 尝试加载一个批次
        for batch_idx, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(train_loader):
            print(f"批次 {batch_idx}:")
            print(f"  seq_x 形状: {seq_x.shape}")
            print(f"  seq_y 形状: {seq_y.shape}")
            print(f"  seq_x_mark 形状: {seq_x_mark.shape}")
            print(f"  seq_y_mark 形状: {seq_y_mark.shape}")
            break  # 只查看第一个批次
            
        # 测试验证数据
        val_dataset, val_loader = data_provider(args, flag='val')
        print(f"验证数据集大小: {len(val_dataset)}")
        
        # 测试测试数据
        test_dataset, test_loader = data_provider(args, flag='test')
        print(f"测试数据集大小: {len(test_dataset)}")
        
        print("weather数据集加载成功!")
        return True
        
    except Exception as e:
        print(f"加载weather数据集时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_weather_data()