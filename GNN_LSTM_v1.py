"""
File: GNN_LSTM_baseline.py
Description: 基于数据驱动图（皮尔逊相关系数）和 GNN-LSTM 串联架构的风电出力预测模型。
核心特征：
1. 包含时间戳的特征工程（正余弦周期编码）
2. 全局视角的 Dataset（保留 Node 维度）
3. Spatio-Temporal 建模（先 GCN 融合空间，后 LSTM 提取时序）
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# =====================================================================
# 模块一：特征工程与图构建 (Feature Engineering & Graph Construction)
# =====================================================================

def process_time_features(raw_data):
    """
    处理原始数据中的时间戳，生成周期性特征并重新拼接。
    :param raw_data: 原始数据 numpy array，形状为 (Time, Nodes=10, Features=6)
                     Index 0: 时间字符串 (YYYY-MM-DD HH:MM:SS)
                     Index 1-4: 气象特征 (U10, V10, U100, V100)
                     Index 5: 风电出力 (Power)
    :return: 处理后的 numpy array，剔除字符串时间，加入时间编码与物理风速。
             最终特征: [U10, V10, U100, V100, WS10, WS100, Power, hour_sin, hour_cos, month_sin, month_cos]
    """
    timestamps = pd.to_datetime(raw_data[:, 0, 0])

    hours = timestamps.hour.values.astype(np.float32)
    months = timestamps.month.values.astype(np.float32)

    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    month_sin = np.sin(2.0 * np.pi * months / 12.0)
    month_cos = np.cos(2.0 * np.pi * months / 12.0)

    time_feats = np.stack([hour_sin, hour_cos, month_sin, month_cos], axis=-1)
    time_feats = np.repeat(time_feats[:, np.newaxis, :], raw_data.shape[1], axis=1).astype(np.float32)

    numeric_feats = raw_data[:, :, 1:].astype(np.float32)
    u10 = numeric_feats[:, :, 0]
    v10 = numeric_feats[:, :, 1]
    u100 = numeric_feats[:, :, 2]
    v100 = numeric_feats[:, :, 3]
    power = numeric_feats[:, :, 4]

    ws10 = np.sqrt(np.square(u10) + np.square(v10)).astype(np.float32)
    ws100 = np.sqrt(np.square(u100) + np.square(v100)).astype(np.float32)

    ws10 = np.expand_dims(ws10, axis=-1)
    ws100 = np.expand_dims(ws100, axis=-1)
    power = np.expand_dims(power, axis=-1)

    processed = np.concatenate(
        [
            numeric_feats[:, :, :4],
            ws10,
            ws100,
            power,
            time_feats,
        ],
        axis=-1,
    )
    return processed


def compute_adjacency_matrix(train_data, power_idx, threshold=0.5):
    """
    基于训练集的风电出力计算风场间的皮尔逊相关系数矩阵。
    :param train_data: 处理后的训练集数据 (Time, Nodes=10, Features)
    :param power_idx: 风电出力在 Feature 维度中的索引
    :param threshold: 相关性阈值，低于该值的置为 0
    :return: 邻接矩阵 Tensor，形状 (10, 10)
    """
    power_values = train_data[:, :, power_idx]  # (T, N)
    corr = np.corrcoef(power_values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    adj = np.where(np.abs(corr) >= threshold, corr, 0.0)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)

    # 对称归一化，提升 GCN 训练稳定性
    degree = np.sum(adj, axis=1)
    degree = np.clip(degree, 1e-8, None)
    d_inv_sqrt = np.diag(np.power(degree, -0.5))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

    return torch.tensor(adj_norm, dtype=torch.float32)


# =====================================================================
# 模块二：全局视角的 Dataset (Global Dataset)
# =====================================================================

class WindGlobalDataset(Dataset):
    """
    风电全局数据集，一次性输出 10 个风电场的数据，保留 Nodes 维度。
    """

    def __init__(self, processed_data, mode='train', split_ratio=0.8, window_size=24, scaler=None):
        """
        初始化数据集。
        """
        self.window_size = window_size
        self.power_idx = 6
        self.weather_idxs = [0, 1, 2, 3, 4, 5]
        self.time_idxs = [7, 8, 9, 10]
        self.future_cov_idxs = self.weather_idxs + self.time_idxs

        train_len = int(len(processed_data) * split_ratio)
        train_data = processed_data[:train_len]
        val_data = processed_data[train_len:]

        if mode == 'train':
            self.scaler = StandardScaler()
            weather_flat = train_data[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
            self.scaler.fit(weather_flat)
            self.segment = train_data
        else:
            if scaler is None:
                raise ValueError('Validation mode requires a fitted scaler from training set.')
            self.scaler = scaler
            self.segment = val_data

        self.data = self.segment.copy()
        segment_weather = self.segment[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
        self.data[:, :, self.weather_idxs] = self.scaler.transform(segment_weather).reshape(
            self.segment.shape[0], self.segment.shape[1], len(self.weather_idxs)
        )

        self.indices = list(range(self.window_size, len(self.data) - self.window_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        # 过去 24 步 (24, 10, 11)
        past_seq = self.data[t - self.window_size:t, :, :]
        
        # 未来 24 步 (24, 10, 11)
        future_seq = self.data[t:t + self.window_size, :, :].copy()
        # 【关键】：将未来的风电出力 (Target) 抹零，防止数据泄露
        future_seq[:, :, self.power_idx] = 0.0 
        
        # 【关键】：沿时间维度 (axis=0) 拼接！长度变成 48
        x = np.concatenate([past_seq, future_seq], axis=0).astype(np.float32)

        # Label 保持不变 (24, 10)
        y = self.data[t:t + self.window_size, :, self.power_idx].astype(np.float32)

        return torch.tensor(x), torch.tensor(y)


# =====================================================================
# 模块三：GNN-LSTM 模型架构 (Model Architecture)
# =====================================================================

class GraphConvLayer(nn.Module):
    """
    简单的图卷积层 (GCN Layer)
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        """
        :param x: (Batch, Seq_Len, Nodes, in_features)
        :param adj: (Nodes, Nodes)
        :return: (Batch, Seq_Len, Nodes, out_features)
        """
        support = torch.matmul(x, self.weight)  # (B, S, N, F_out)
        aggregated = torch.einsum('ij,bslj->bsli', adj, support.transpose(2, 3)).transpose(2, 3)
        return F.relu(aggregated)


class WindGNNLSTM(nn.Module):
    def __init__(self, num_nodes=10, in_dim=11, gnn_dim=32, lstm_dim=64, num_layers=2, output_dim=24, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        # 前置非线性特征提取器：将原始11维特征映射到高阶表征空间
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, gnn_dim),
        )

        # GCN 输入改为高阶特征维度
        self.gcn1 = GraphConvLayer(gnn_dim, gnn_dim)
        self.gcn2 = GraphConvLayer(gnn_dim, gnn_dim)

        # LSTM 输入维度 = 图特征(gnn_dim) + 原始特征(in_dim)
        self.lstm_input_dim = gnn_dim + in_dim
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 全局轨迹解码：用最终隐藏状态一次性生成未来 output_dim 步
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, adj):
        # x 形状: (Batch, 48, 10, 11)
        b, s, n, f = x.shape

        # 1. 前置特征提取：低阶特征 -> 高阶特征
        x_features = self.feature_extractor(x)  # (B, 48, 10, gnn_dim)
        
        # 2. 双层图卷积提取空间特征
        x_gcn = self.gcn1(x_features, adj)
        x_gcn = self.gcn2(x_gcn, adj)  # (B, 48, 10, gnn_dim)
        
        # 3. 残差拼接：保留原始输入信息
        x_fused = torch.cat([x_gcn, x], dim=-1) # (B, 48, 10, 43)
        
        # 4. 维度转换，送入 LSTM
        x_lstm_in = x_fused.permute(0, 2, 1, 3).contiguous().view(b * n, s, -1)
        _, (h_n, _) = self.lstm(x_lstm_in)
        
        # 5. 全局状态解码：取最后一层隐藏状态，一次性回归未来轨迹
        last_hidden = h_n[-1] # (B*N, lstm_dim)
        out = self.fc(last_hidden) # (B*N, output_dim)
        
        # 还原形状为 (Batch, Nodes, Output_Seq)
        out = out.view(b, n, self.output_dim)
        return out


# =====================================================================
# 模块四：主函数与训练循环 (Main Script)
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='GNN-LSTM v2 训练脚本')

    # 数据相关参数
    parser.add_argument('--data-path', type=str, default='data/wind_train_val_2012-01-02_to_2013-07-13.npy')
    parser.add_argument('--split-ratio', type=float, default=0.8)
    parser.add_argument('--window-size', type=int, default=24)
    parser.add_argument('--corr-threshold', type=float, default=0.5)

    # 训练相关参数
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--log-root', type=str, default='logs')

    # 模型相关参数
    parser.add_argument('--num-nodes', type=int, default=10)
    parser.add_argument('--input-dim', type=int, default=11)
    parser.add_argument('--gnn-dim', type=int, default=32)
    parser.add_argument('--lstm-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--output-dim', type=int, default=24)
    parser.add_argument('--dropout', type=float, default=0.2)

    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(args.log_root, f'gnn_lstm_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_data = np.load(args.data_path, allow_pickle=True)
    processed_data = process_time_features(raw_data)

    train_len = int(len(processed_data) * args.split_ratio)
    train_data = processed_data[:train_len]

    adj = compute_adjacency_matrix(
        train_data=train_data,
        power_idx=6,
        threshold=args.corr_threshold,
    ).to(device)

    train_dataset = WindGlobalDataset(
        processed_data=processed_data,
        mode='train',
        split_ratio=args.split_ratio,
        window_size=args.window_size,
    )
    val_dataset = WindGlobalDataset(
        processed_data=processed_data,
        mode='val',
        split_ratio=args.split_ratio,
        window_size=args.window_size,
        scaler=train_dataset.scaler,
    )

    joblib.dump(train_dataset.scaler, os.path.join(exp_dir, 'scaler.pkl'))
    torch.save(adj.detach().cpu(), os.path.join(exp_dir, 'adjacency.pt'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = WindGNNLSTM(
        num_nodes=args.num_nodes,
        in_dim=args.input_dim,
        gnn_dim=args.gnn_dim,
        lstm_dim=args.lstm_dim,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)

    # 模型DNA：供通用测试脚本自动恢复预处理与网络结构
    model_config = {
        'model_type': 'GNN_LSTM_v1',
        'num_nodes': args.num_nodes,
        'input_dim': args.input_dim,
        'gnn_dim': args.gnn_dim,
        'lstm_dim': args.lstm_dim,
        'num_layers': args.num_layers,
        'output_dim': args.output_dim,
        'dropout': args.dropout,
        'power_idx': 6,
        'weather_idxs': [0, 1, 2, 3, 4, 5],
        'use_synthetic_wind': True,
        'use_time_encode': True,
        'use_adaptive_adj': False,
        'future_power_fill': 'zero',
    }

    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(model_config, f, ensure_ascii=False, indent=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 【新增】当 val_mae 连续 5 轮不降，学习率减半，继续逼近最优解
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()

    print(f'实验开始，保存目录: {exp_dir}')

    with open(os.path.join(exp_dir, 'train_log.txt'), 'w') as log_file:
        for epoch in range(args.epochs):
            model.train()
            train_loss_sum = 0.0

            for bx, by in train_loader:
                bx = bx.to(device)
                by = by.to(device)  # (B, S, N)
                target = by.permute(0, 2, 1).contiguous()  # (B, N, S)

                optimizer.zero_grad()
                pred = model(bx, adj)  # (B, N, S)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()

            model.eval()
            val_mae_sum = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    by = by.to(device)
                    target = by.permute(0, 2, 1).contiguous()

                    pred = model(bx, adj)
                    val_mae = torch.mean(torch.abs(pred - target))
                    val_mae_sum += val_mae.item()

            train_loss = train_loss_sum / max(len(train_loader), 1)
            val_mae = val_mae_sum / max(len(val_loader), 1)

            status = f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}'
            print(status)
            # 【新增】告诉调度器当前的验证集误差
            scheduler.step(val_mae)
            log_file.write(status + '\n')
            log_file.flush()

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                save_path = os.path.join(exp_dir, f'gnn_lstm_epoch_{epoch + 1}.pth')
                torch.save(model.state_dict(), save_path)
                with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
                    json.dump(model_config, f, ensure_ascii=False, indent=4)
                print(f'模型已保存: {save_path}')


if __name__ == '__main__':
    main()
