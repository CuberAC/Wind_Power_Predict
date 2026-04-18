import argparse
import os
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# =====================================================================
# 1. 核心网络结构与 Dataset 定义 (推理阶段)
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
    """用于推理的数据集：不再返回真实标签 (y)"""
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
        return torch.tensor(x)

# =====================================================================
# 2. 数据处理与对齐函数
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

    xgb_weather = np.stack([U10, V10, U100, V100, WS10, WS100, WS100_Cube, Sin_WDir, Cos_WDir, Sin_H, Cos_H, Sin_M, Cos_M], axis=-1)
    gnn_weather = np.stack([U10, V10, U100, V100, WS10, WS100, Power, Sin_H, Cos_H, Sin_M, Cos_M], axis=-1)
    
    return xgb_weather, gnn_weather, Power

def build_xgb_sliding_window(weather, power, window=24):
    """仅生成用于推理的特征X，无需输出目标Y"""
    num_t, num_f = weather.shape[0], weather.shape[1]
    num_samples = num_t - window * 2
    f_dim = window + 4 + (window * 13) * 2
    X = np.zeros((num_samples, num_f, f_dim), dtype=np.float32)
    
    idx = 0
    for i in range(window, num_t - window):
        for f in range(num_f):
            past_p = power[i-window : i, f]
            stats = np.array([np.mean(past_p), np.std(past_p), np.max(past_p), np.min(past_p)])
            past_w = weather[i-window : i, f, :].flatten()
            future_w = weather[i : i+window, f, :].flatten()
            X[idx, f, :] = np.concatenate([past_p, stats, past_w, future_w])
        idx += 1
    return X

def flatten_add_farm(X_3d):
    num_s, num_f, f_dim = X_3d.shape
    X_2d = X_3d.reshape(num_s * num_f, f_dim)
    farm_ids = np.tile(np.arange(num_f), num_s).reshape(-1, 1)
    return np.concatenate([X_2d, farm_ids.astype(np.float32)], axis=1)

# =====================================================================
# 3. 预测管道
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Stacking 聚合纯预测管道")
    parser.add_argument("--test-data", type=str, default="data/wind_test_cleaned.npy", help="输入数据的路径 (.npy格式)")
    parser.add_argument("--xgb-model", type=str, default="saved_models/final_ensemble/layer1_xgb_full.pkl", help="层1: XGBoost权重路径")
    parser.add_argument("--gnn-model", type=str, default="saved_models/final_ensemble/layer1_gnn_full.pth", help="层1: GNN-LSTM权重路径")
    parser.add_argument("--gnn-scaler", type=str, default="saved_models/final_ensemble/scaler.pkl", help="层1: GNN-LSTM Scaler路径")
    parser.add_argument("--gnn-adj", type=str, default="saved_models/final_ensemble/adjacency.pt", help="层1: GNN-LSTM Adjacency路径")
    parser.add_argument("--meta-model", type=str, default="saved_models/final_ensemble/layer2_ridge.pkl", help="层2: 聚合模型权重路径")
    parser.add_argument("--output", type=str, default="data/stacking_final_predictions.npy", help="最终预测结果保存路径")
    parser.add_argument("--batch-size", type=int, default=128, help="GNN网络推理 Batch Size")
    parser.add_argument("--clip", action="store_true", help="是否将输出限制在 0 到 1 之间 (推荐)")
    args = parser.parse_args()

    if not os.path.exists(args.test_data):
        print(f"执行失败: 找不到指定的测试数据文件: {args.test_data}")
        return

    print("1/6 读取数据并提取基础特征...")
    data = np.load(args.test_data, allow_pickle=True)
    xgb_w, gnn_w, power = extract_features_from_raw(data)

    print("2/6 构建 XGBoost 数据流...")
    X_xgb_3d = build_xgb_sliding_window(xgb_w, power)
    X_xgb_2d = flatten_add_farm(X_xgb_3d)

    print("3/6 运行 XGBoost 首层预测...")
    xgb_model = joblib.load(args.xgb_model)
    pred_xgb = xgb_model.predict(X_xgb_2d)

    print("4/6 构建 GNN-LSTM 数据流...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = joblib.load(args.gnn_scaler)
    adj = torch.load(args.gnn_adj).to(device)

    indices = np.arange(len(gnn_w) - 48)
    test_dataset = WindGlobalDataset(gnn_w, indices, scaler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("5/6 运行 GNN-LSTM 首层预测...")
    gnn_model = WindGNNLSTM().to(device)
    # 强制将权重加载进目标设备
    gnn_model.load_state_dict(torch.load(args.gnn_model, map_location=device, weights_only=True))
    gnn_model.eval()
    
    gnn_preds = []
    with torch.no_grad():
        for bx in test_loader:
            pred = gnn_model(bx.to(device), adj)
            gnn_preds.append(pred.cpu().numpy())
    pred_gnn_2d = np.concatenate(gnn_preds, axis=0).reshape(-1, 24)

    print("6/6 结合二层元模型进行最终 Stacking 预测...")
    meta_model = joblib.load(args.meta_model)
    Meta_X = np.concatenate([pred_xgb, pred_gnn_2d], axis=1)
    
    final_pred_2d = meta_model.predict(Meta_X)
    final_pred_3d = final_pred_2d.reshape(-1, 10, 24)

    if args.clip:
        final_pred_3d = np.clip(final_pred_3d, 0.0, 1.0)

    # 导出文件
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, final_pred_3d)

    print("-" * 40)
    print("预测流程执行完毕！")
    print(f"输出数据形状: {final_pred_3d.shape}")
    print(f"预测结果已保存至: {args.output}")

if __name__ == "__main__":
    main()
