import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib

# =====================================================================
# 1. 复制你 LSTM_baseline.py 中的模型结构
# =====================================================================
class WindLSTM(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, num_layers=2, output_dim=24, lstm_dropout=0.2, fc_dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(p=fc_dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        out = self.dropout(out) 
        return self.fc(out)

# =====================================================================
# 2. 针对 Baseline 的数据预处理逻辑 (严格对应 baseline.py)
# =====================================================================
def preprocess_baseline_test(raw_data):
    """
    针对 Baseline 的特征工程：[U10, V10, U100, V100, Power, WS100]
    """
    print("🧹 正在处理测试集缺失值 (NaN)...")
    # 提取数值位 (Index 1-5: U10, V10, U100, V100, Power)
    numeric_data = raw_data[:, :, 1:].astype(float)
    
    # 插值处理 NaN
    for n in range(numeric_data.shape[1]):
        df = pd.DataFrame(numeric_data[:, n, :])
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        numeric_data[:, n, :] = df.values

    # 合成 WS100 (基于 U100 和 V100)
    print("⚙️ 正在合成 WS100 特征...")
    u100 = numeric_data[:, :, 2]
    v100 = numeric_data[:, :, 3]
    ws100 = np.sqrt(u100**2 + v100**2)[:, :, np.newaxis]
    
    # 组合成 Baseline 所需的 6 维基础特征
    full_data = np.concatenate([numeric_data, ws100], axis=-1).astype(np.float32)
    return full_data

class WindBaselineTestDataset(Dataset):
    def __init__(self, data, scaler, window_size=24):
        self.window_size = window_size
        self.data = data.copy()
        
        # 索引定义 (对应 baseline)
        self.power_idx = 4
        self.weather_idxs = [0, 1, 2, 3, 5] # U10, V10, U100, V100, WS100
        
        # 归一化气象特征 (注意 Baseline 只对这5个气象特征做了归一化)
        segment_weather = self.data[:, :, self.weather_idxs].reshape(-1, 5)
        self.data[:, :, self.weather_idxs] = scaler.transform(segment_weather).reshape(
            self.data.shape[0], 10, 5
        )
        
        # 构建索引 (Time, Node)
        self.indices = []
        for i in range(self.window_size, len(self.data) - self.window_size):
            for f in range(10):
                self.indices.append((i, f))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, f = self.indices[idx]
        # 过去 24 步的 6 维数据
        past_seq = self.data[t - self.window_size : t, f, :] 
        # 未来 24 步的 5 维气象数据
        future_weather = self.data[t : t + self.window_size, f, self.weather_idxs]
        
        # 拼接成 11 维输入
        X = np.concatenate([past_seq, future_weather], axis=-1)
        # 真实的未来出力
        Y = self.data[t : t + self.window_size, f, self.power_idx]
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# =====================================================================
# 3. 评测主函数
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='评测纯 LSTM Baseline 模型')
    parser.add_argument('--test-path', type=str, default='data/wind_test_2013-07-14_to_end.npy')
    parser.add_argument('--model-dir', type=str, required=True, help='包含 baseline scaler.pkl 和 pth 的目录')
    parser.add_argument('--model-file', type=str, required=True, help='具体的 pth 文件名')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载组件
    print(f"📦 正在加载组件...")
    scaler = joblib.load(os.path.join(args.model_dir, 'scaler.pkl'))
    
    # 2. 加载数据并预处理
    raw_test_data = np.load(args.test_path, allow_pickle=True)
    full_data = preprocess_baseline_test(raw_test_data)
    
    test_dataset = WindBaselineTestDataset(full_data, scaler, window_size=24)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 3. 实例化模型
    model = WindLSTM(
        input_dim=11,
        hidden_dim=128,
        num_layers=3,
        output_dim=24
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_file), map_location=device))
    model.eval()

    # 4. 评估
    print(f"✅ 开始评测 LSTM Baseline...")
    mae_sum = 0.0
    mse_sum = 0.0
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            
            mae_sum += torch.sum(torch.abs(pred - by)).item()
            mse_sum += torch.sum(torch.square(pred - by)).item()

    total_elements = len(test_dataset) * 24
    test_mae = mae_sum / total_elements
    test_rmse = np.sqrt(mse_sum / total_elements)

    print("-" * 50)
    print("📊 LSTM Baseline 测试评估结果 📊")
    print("-" * 50)
    print(f"🏆 Test MAE  : {test_mae:.4f}")
    print(f"🏆 Test RMSE : {test_rmse:.4f}")
    print("-" * 50)

if __name__ == '__main__':
    main()