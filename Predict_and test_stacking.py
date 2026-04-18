import os
import time
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================================
# 1. 核心网络结构与 Dataset 声明 (加载 GNN 模型必须的类定义)
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
    def __init__(self, num_nodes=10, in_dim=11, gnn_dim=32, lstm_dim=64, num_layers=2, output_dim=24, dropout=0.1):
        super().__init__()
        self.num_nodes, self.output_dim = num_nodes, output_dim
        self.feature_extractor = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, gnn_dim))
        self.gcn1 = GraphConvLayer(gnn_dim, gnn_dim)
        self.gcn2 = GraphConvLayer(gnn_dim, gnn_dim)
        self.lstm = nn.LSTM(gnn_dim + in_dim, lstm_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Sequential(nn.Linear(lstm_dim, 128), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(128, output_dim))

    def forward(self, x, adj):
        b, s, n, f = x.shape
        x_features = self.feature_extractor(x)
        x_gcn = self.gcn2(self.gcn1(x_features, adj), adj)
        x_fused = torch.cat([x_gcn, x], dim=-1)
        x_lstm_in = x_fused.permute(0, 2, 1, 3).contiguous().view(b * n, s, -1)
        _, (h_n, _) = self.lstm(x_lstm_in)
        out = self.fc(h_n[-1]).view(b, n, self.output_dim)
        return out

class WindGlobalDataset(Dataset):
    """对齐 XGBoost 滑动逻辑的专属数据集"""
    def __init__(self, processed_data, indices, scaler, window_size=24):
        self.window_size = window_size
        self.power_idx = 6
        self.weather_idxs = [0, 1, 2, 3, 4, 5]
        self.indices = indices
        self.full_data = processed_data.copy()
        
        weather_shape = self.full_data[:, :, self.weather_idxs].shape
        self.full_data[:, :, self.weather_idxs] = scaler.transform(
            self.full_data[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
        ).reshape(weather_shape)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx] + self.window_size
        past_seq = self.full_data[t - self.window_size : t, :, :]
        future_seq = self.full_data[t : t + self.window_size, :, :].copy()
        future_seq[:, :, self.power_idx] = 0.0 # 防泄露抹零
        x = np.concatenate([past_seq, future_seq], axis=0).astype(np.float32)
        y = self.full_data[t : t + self.window_size, :, self.power_idx].astype(np.float32)
        return torch.tensor(x), torch.tensor(y)

# =====================================================================
# 2. 数据处理与对齐函数 (包含 XGB 和 GNN)
# =====================================================================
def extract_features_from_raw(data):
    """提取基础气象及时间正余弦特征"""
    timestamps = pd.to_datetime(data[:, 0, 0])
    hours, months = timestamps.hour.values, timestamps.month.values
    sin_hour, cos_hour = np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24)
    sin_month, cos_month = np.sin(2*np.pi*months/12), np.cos(2*np.pi*months/12)
    num_f = data.shape[1]
    
    Sin_H = np.repeat(sin_hour[:, np.newaxis], num_f, axis=1)
    Cos_H = np.repeat(cos_hour[:, np.newaxis], num_f, axis=1)
    Sin_M = np.repeat(sin_month[:, np.newaxis], num_f, axis=1)
    Cos_M = np.repeat(cos_month[:, np.newaxis], num_f, axis=1)

    nd = data[:, :, 1:].astype(np.float32)
    U10, V10, U100, V100, Power = nd[:,:,0], nd[:,:,1], nd[:,:,2], nd[:,:,3], nd[:,:,4]
    WS10, WS100 = np.sqrt(U10**2 + V10**2), np.sqrt(U100**2 + V100**2)
    WS100_Cube = WS100 ** 3
    Wind_Dir = np.arctan2(V100, U100)
    Sin_WDir, Cos_WDir = np.sin(Wind_Dir), np.cos(Wind_Dir)

    # XGB特征: 13维
    xgb_weather = np.stack([U10, V10, U100, V100, WS10, WS100, WS100_Cube, Sin_WDir, Cos_WDir, Sin_H, Cos_H, Sin_M, Cos_M], axis=-1)
    # GNN特征: 11维 (符合 GNN_LSTM_v1.py 定义)
    gnn_weather = np.stack([U10, V10, U100, V100, WS10, WS100, Power, Sin_H, Cos_H, Sin_M, Cos_M], axis=-1)
    
    return xgb_weather, gnn_weather, Power

def build_xgb_sliding_window(weather, power, window=24):
    num_t, num_f = weather.shape[0], weather.shape[1]
    num_samples = num_t - window * 2
    f_dim = window + 4 + (window * 13) * 2
    X = np.zeros((num_samples, num_f, f_dim), dtype=np.float32)
    Y = np.zeros((num_samples, num_f, window), dtype=np.float32)
    
    idx = 0
    for i in range(window, num_t - window):
        for f in range(num_f):
            past_p = power[i-window : i, f]
            stats = np.array([np.mean(past_p), np.std(past_p), np.max(past_p), np.min(past_p)])
            past_w = weather[i-window : i, f, :].flatten()
            future_w = weather[i : i+window, f, :].flatten()
            X[idx, f, :] = np.concatenate([past_p, stats, past_w, future_w])
            Y[idx, f, :] = power[i : i+window, f]
        idx += 1
    return X, Y

def flatten_add_farm(X_3d):
    num_s, num_f, f_dim = X_3d.shape
    X_2d = X_3d.reshape(num_s * num_f, f_dim)
    farm_ids = np.tile(np.arange(num_f), num_s).reshape(-1, 1)
    return np.concatenate([X_2d, farm_ids.astype(np.float32)], axis=1)

# =====================================================================
# 3. 终极预测流水线 (Main)
# =====================================================================
def main():
    test_file = 'data/wind_val_20.npy'
    if not os.path.exists(test_file):
        print(f"❌ 找不到测试集文件 {test_file}")
        return

    print(f"[{time.strftime('%H:%M:%S')}] 🚀 1. 开始读取未知测试集并提取特征...")
    data = np.load(test_file, allow_pickle=True)
    xgb_w, gnn_w, power = extract_features_from_raw(data)

    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ 2. 正在执行 XGBoost 数据对齐...")
    X_xgb_3d, Y_test_3d = build_xgb_sliding_window(xgb_w, power)
    X_xgb_2d = flatten_add_farm(X_xgb_3d)

    print(f"[{time.strftime('%H:%M:%S')}] 🧠 3. 正在召唤 XGBoost 满血大将...")
    xgb_model = joblib.load('saved_models/final_ensemble/layer1_xgb_full.pkl')
    pred_xgb = xgb_model.predict(X_xgb_2d)

    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ 4. 正在执行 GNN_LSTM 数据对齐...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = joblib.load('saved_models/final_ensemble/scaler.pkl') # 注意：请确认 scaler 的真实存放路径
    adj = torch.load('saved_models/final_ensemble/adjacency.pt').to(device)

    indices = np.arange(len(gnn_w) - 48)
    test_dataset = WindGlobalDataset(gnn_w, indices, scaler)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"[{time.strftime('%H:%M:%S')}] 🔮 5. 正在召唤 PyTorch GNN_LSTM 满血大将...")
    gnn_model = WindGNNLSTM().to(device)
    gnn_model.load_state_dict(torch.load('saved_models/final_ensemble/layer1_gnn_full.pth', map_location=device))
    gnn_model.eval()
    
    gnn_preds = []
    with torch.no_grad():
        for bx, _ in test_loader:
            pred = gnn_model(bx.to(device), adj)
            gnn_preds.append(pred.cpu().numpy())
    pred_gnn_2d = np.concatenate(gnn_preds, axis=0).reshape(-1, 24)

    print(f"[{time.strftime('%H:%M:%S')}] 👑 6. 正在移交 Stacking 统帅 (Ridge Meta Model) 执行最终裁决...")
    meta_model = joblib.load('saved_models/final_ensemble/layer2_ridge.pkl')
    Meta_X = np.concatenate([pred_xgb, pred_gnn_2d], axis=1)
    
    # 终极预测结果！(2D形状)
    final_pred_2d = meta_model.predict(Meta_X)
    # 将其还原为业务所需的 3D 形状: (时间步, 10个风场, 未来24小时)
    final_pred_3d = final_pred_2d.reshape(-1, 10, 24)

    # =====================================================================
    # 4. 生成终极业绩报表
    # =====================================================================
    print("\n" + "★" * 60)
    print(" 🏆 风场功率联合预测 - 最终测试集体检报告 🏆")
    print("★" * 60)
    
    total_rmse, total_mae = 0.0, 0.0
    
    # 逐风场计算并打印指标
    for farm_id in range(10):
        # 提取当前风场的真实值和预测值
        y_true = Y_test_3d[:, farm_id, :]
        y_pred = final_pred_3d[:, farm_id, :]
        
        farm_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        farm_mae = mean_absolute_error(y_true, y_pred)
        
        total_rmse += farm_rmse
        total_mae += farm_mae
        
        print(f" 🏭 风电场 {farm_id:02d} | RMSE: {farm_rmse:.6f} | MAE: {farm_mae:.6f}")
        
    print("-" * 60)
    print(f" 🌍 全局平均考核 | RMSE: {total_rmse / 10:.6f} | MAE: {total_mae / 10:.6f}")
    print("★" * 60)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 恭喜！流水线执行完毕。")

if __name__ == "__main__":
    main()