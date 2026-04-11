import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import joblib

# =====================================================================
# 1. 复制你训练脚本中的模型结构 (必须与保存的模型完全一致)
# =====================================================================
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        aggregated = torch.einsum('ij,bslj->bsli', adj, support.transpose(2, 3)).transpose(2, 3)
        return F.relu(aggregated)

class WindGNNLSTM(nn.Module):
    def __init__(self, num_nodes=10, in_dim=11, gnn_dim=32, lstm_dim=128, num_layers=3, output_dim=24, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, gnn_dim)
        )
        self.gcn1 = GraphConvLayer(gnn_dim, gnn_dim)
        self.gcn2 = GraphConvLayer(gnn_dim, gnn_dim)

        self.lstm_input_dim = gnn_dim + in_dim
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 预测头 (请确保与你保存v1模型时的fc结构一致)
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, adj):
        b, s, n, f = x.shape
        x_features = self.feature_extractor(x)
        x_gcn = self.gcn1(x_features, adj)
        x_gcn = self.gcn2(x_gcn, adj)
        x_fused = torch.cat([x_gcn, x], dim=-1)
        
        x_lstm_in = x_fused.permute(0, 2, 1, 3).contiguous().view(b * n, s, -1)
        _, (h_n, _) = self.lstm(x_lstm_in)
        last_hidden = h_n[-1]
        
        out = self.fc(last_hidden)
        out = out.view(b, n, self.output_dim)
        return out

# =====================================================================
# 2. 测试集专属的数据预处理与 Dataset
# =====================================================================
def clean_and_process_test_data(raw_data):
    """
    清洗包含 NaN 的测试集，并合成特征 (11维)
    """
    print("🧹 正在处理测试集缺失值 (NaN)...")
    numeric_data = raw_data[:, :, 1:].astype(float)
    
    # 针对每个风电场，沿时间轴进行线性插值
    for n in range(numeric_data.shape[1]):
        df = pd.DataFrame(numeric_data[:, n, :])
        # 线性插值，如果开头或结尾有NaN则用最邻近的有效值填充(bfill/ffill)
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        numeric_data[:, n, :] = df.values

    # 特征合成
    print("⚙️ 正在合成风速与时间编码特征...")
    timestamps = pd.to_datetime(raw_data[:, 0, 0])
    hours = timestamps.hour.values.astype(np.float32)
    months = timestamps.month.values.astype(np.float32)

    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    month_sin = np.sin(2.0 * np.pi * months / 12.0)
    month_cos = np.cos(2.0 * np.pi * months / 12.0)
    time_feats = np.stack([hour_sin, hour_cos, month_sin, month_cos], axis=-1)
    time_feats = np.repeat(time_feats[:, np.newaxis, :], raw_data.shape[1], axis=1)

    # 计算真实物理风速 WS10, WS100
    u10, v10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
    u100, v100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
    ws10 = np.sqrt(np.square(u10) + np.square(v10))[..., np.newaxis]
    ws100 = np.sqrt(np.square(u100) + np.square(v100))[..., np.newaxis]

    # 取出出力 Power (原 index 4)
    power = numeric_data[:, :, 4:5]
    weather_uv = numeric_data[:, :, 0:4]

    # 拼接: [U10,V10,U100,V100] + [WS10,WS100] + [Power] + [Time*4] = 11 维
    processed = np.concatenate([weather_uv, ws10, ws100, power, time_feats], axis=-1).astype(np.float32)
    
    # 最终检查是否有遗漏的 NaN
    if np.isnan(processed).sum() > 0:
        processed = np.nan_to_num(processed, nan=0.0) # 终极兜底
        
    return processed

class WindTestDataset(Dataset):
    def __init__(self, data, scaler, window_size=24):
        self.window_size = window_size
        self.power_idx = 6  # 11维特征中，Power处于索引6
        self.weather_idxs = [0, 1, 2, 3, 4, 5] # U, V, WS
        self.data = data.copy()

        # 归一化气象特征 (严格使用训练集的 scaler)
        segment_weather = self.data[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
        self.data[:, :, self.weather_idxs] = scaler.transform(segment_weather).reshape(
            self.data.shape[0], self.data.shape[1], len(self.weather_idxs)
        )

        self.indices = list(range(self.window_size, len(self.data) - self.window_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        past_seq = self.data[t - self.window_size:t, :, :]
        future_seq = self.data[t:t + self.window_size, :, :].copy()
        
        future_seq[:, :, self.power_idx] = 0.0 # 未来出力置零，模型需要预测这个位置
        
        x = np.concatenate([past_seq, future_seq], axis=0).astype(np.float32)
        y = self.data[t:t + self.window_size, :, self.power_idx].astype(np.float32)

        return torch.tensor(x), torch.tensor(y)

# =====================================================================
# 3. 推理主函数
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='测试已保存的 GNN-LSTM 模型')
    parser.add_argument('--test-path', type=str, default='data/wind_test_2013-07-14_to_end.npy')
    parser.add_argument('--model-dir', type=str, required=True, help='包含 scaler.pkl, adjacency.pt 和 pth 文件的日志目录路径')
    parser.add_argument('--model-file', type=str, default='best_gnn_lstm_model.pth', help='模型权重文件名')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载组件
    print(f"📦 正在从 {args.model_dir} 加载模型组件...")
    scaler = joblib.load(os.path.join(args.model_dir, 'scaler.pkl'))
    adj = torch.load(os.path.join(args.model_dir, 'adjacency.pt'), weights_only=False).to(device)
    
    # 2. 加载数据
    print("🚀 正在加载测试数据...")
    raw_test_data = np.load(args.test_path, allow_pickle=True)
    processed_test_data = clean_and_process_test_data(raw_test_data)
    
    test_dataset = WindTestDataset(processed_test_data, scaler=scaler)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. 实例化并加载模型
    model = WindGNNLSTM(
        num_nodes=10,
        in_dim=11,  # 确保这里是你保存模型时的维度
        gnn_dim=32,
        lstm_dim=64,
        num_layers=2,
        output_dim=24,
        dropout=0.2
    ).to(device)
    
    model_path = os.path.join(args.model_dir, args.model_file)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. 执行测试
    print(f"✅ 模型加载成功！开始在 {len(test_dataset)} 个样本上进行测试...")
    mae_sum = 0.0
    mse_sum = 0.0

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            # 这里的 target 形状对齐 (B, N, S)，根据你原本训练集定的
            target = by.permute(0, 2, 1).contiguous() 

            pred = model(bx, adj)
            
            mae_sum += torch.sum(torch.abs(pred - target)).item()
            mse_sum += torch.sum(torch.square(pred - target)).item()

    # 5. 计算并打印结果
    total_elements = len(test_dataset) * 10 * 24
    test_mae = mae_sum / total_elements
    test_rmse = np.sqrt(mse_sum / total_elements)

    print("-" * 50)
    print("🎉 测试集评估报告 🎉")
    print("-" * 50)
    print(f"🏆 Test MAE  : {test_mae:.4f}")
    print(f"🏆 Test RMSE : {test_rmse:.4f}")
    print("-" * 50)

    # 6. 保存评估结果到模型目录
    model_stem = os.path.splitext(os.path.basename(args.model_file))[0]
    report_name = f"test_report_{model_stem}.txt"
    report_path = os.path.join(args.model_dir, report_name)
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GNN-LSTM Test Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Time: {now_str}\n")
        f.write(f"Model Directory: {args.model_dir}\n")
        f.write(f"Model File: {args.model_file}\n")
        f.write(f"Test Data: {args.test_path}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Test MAE: {test_mae:.6f}\n")
        f.write(f"Test RMSE: {test_rmse:.6f}\n")
        f.write("=" * 50 + "\n")

    print(f"📝 测试结果已保存到: {report_path}")

if __name__ == '__main__':
    main()