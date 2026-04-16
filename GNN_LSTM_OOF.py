"""
File: GNN_LSTM_OOF.py
Description: GNN-LSTM with 3-fold TimeSeriesSplit OOF generation for stacking.

Key points:
1) Keep original feature engineering and model architecture unchanged.
2) Use TimeSeriesSplit(n_splits=3) on pool_data (first 80% timeline).
3) Save 2D OOF predictions and 2D test predictions for stacking.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# =====================================================================
# Module 1: Feature Engineering & Graph Construction
# =====================================================================

def process_time_features(raw_data):
    """
    Convert timestamp to cyclic features and concatenate with original numeric features.
    Input raw feature order after timestamp: [U10, V10, U100, V100, Power]
    Output features:
    [U10, V10, U100, V100, WS10, WS100, Power, hour_sin, hour_cos, month_sin, month_cos]
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
    """Build Pearson-correlation adjacency from training power series."""
    power_values = train_data[:, :, power_idx]  # (T, N)
    corr = np.corrcoef(power_values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    adj = np.where(np.abs(corr) >= threshold, corr, 0.0)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)

    degree = np.sum(adj, axis=1)
    degree = np.clip(degree, 1e-8, None)
    d_inv_sqrt = np.diag(np.power(degree, -0.5))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

    return torch.tensor(adj_norm, dtype=torch.float32)


# =====================================================================
# Module 2: Global Dataset (Refactored for pre-split segments)
# =====================================================================

class WindGlobalDataset(Dataset):
    """
    【对齐 XGBoost 专属版】：通过传入指定的 indices 进行切分，确保样本数量和 XGB 绝对一致。
    """
    def __init__(self, processed_data, indices, scaler=None, is_train=False, window_size=24):
        self.window_size = window_size
        self.power_idx = 6
        self.weather_idxs = [0, 1, 2, 3, 4, 5]
        
        self.indices = indices
        self.full_data = processed_data.copy()

        # 防数据泄露：如果是训练集，只拿训练时间段内的天气数据做归一化
        if is_train:
            self.scaler = StandardScaler()
            # 找到训练集的最大时间步，提取历史天气
            max_t = max(self.indices) + window_size * 2
            train_weather = self.full_data[:max_t, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
            self.scaler.fit(train_weather)
        else:
            if scaler is None:
                raise ValueError("Validation/Test needs a fitted scaler!")
            self.scaler = scaler

        # 对全局天气进行归一化（因为后面取切片时直接取）
        weather_shape = self.full_data[:, :, self.weather_idxs].shape
        self.full_data[:, :, self.weather_idxs] = self.scaler.transform(
            self.full_data[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
        ).reshape(weather_shape)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 核心：将 index 还原为真实的时间步 t (和 XGBoost 完全一致的对齐公式)
        t = self.indices[idx] + self.window_size

        past_seq = self.full_data[t - self.window_size : t, :, :]
        future_seq = self.full_data[t : t + self.window_size, :, :].copy()
        
        # 抹零未来真实的功率 (防止泄露)
        future_seq[:, :, self.power_idx] = 0.0 
        
        x = np.concatenate([past_seq, future_seq], axis=0).astype(np.float32)
        y = self.full_data[t : t + self.window_size, :, self.power_idx].astype(np.float32)

        return torch.tensor(x), torch.tensor(y)
# =====================================================================
# Module 3: Model Architecture
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
    def __init__(self, num_nodes=10, in_dim=11, gnn_dim=32, lstm_dim=64, num_layers=2, output_dim=24, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, gnn_dim),
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

        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, adj):
        b, s, n, _ = x.shape

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
# Module 4: Train / Eval / Inference Helpers
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='GNN-LSTM OOF training script')

    parser.add_argument('--data-path', type=str, default='data/wind_train_val_2012-01-02_to_2013-07-13.npy')
    parser.add_argument('--window-size', type=int, default=24)
    parser.add_argument('--corr-threshold', type=float, default=0.5)

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    parser.add_argument('--num-nodes', type=int, default=10)
    parser.add_argument('--input-dim', type=int, default=11)
    parser.add_argument('--gnn-dim', type=int, default=32)
    parser.add_argument('--lstm-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--output-dim', type=int, default=24)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--n-splits', type=int, default=3)
    parser.add_argument('--pool-ratio', type=float, default=0.8)
    parser.add_argument('--stacking-dir', type=str, default='stacking_data')

    return parser.parse_args()


def build_model(args, device):
    return WindGNNLSTM(
        num_nodes=args.num_nodes,
        in_dim=args.input_dim,
        gnn_dim=args.gnn_dim,
        lstm_dim=args.lstm_dim,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)


def run_train_eval_epochs(model, train_loader, val_loader, adj, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
    )
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for bx, by in train_loader:
            bx = bx.to(adj.device)
            by = by.to(adj.device)
            target = by.permute(0, 2, 1).contiguous()

            optimizer.zero_grad()
            pred = model(bx, adj)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_mae_sum = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(adj.device)
                by = by.to(adj.device)
                target = by.permute(0, 2, 1).contiguous()

                pred = model(bx, adj)
                val_mae = torch.mean(torch.abs(pred - target))
                val_mae_sum += val_mae.item()

        train_loss = train_loss_sum / max(len(train_loader), 1)
        val_mae = val_mae_sum / max(len(val_loader), 1)
        scheduler.step(val_mae)
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}')


def run_full_train_only(model, full_train_loader, adj, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for bx, by in full_train_loader:
            bx = bx.to(adj.device)
            by = by.to(adj.device)
            target = by.permute(0, 2, 1).contiguous()

            optimizer.zero_grad()
            pred = model(bx, adj)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(len(full_train_loader), 1)
        print(f'[FULL] Epoch {epoch + 1}, Loss: {train_loss:.4f}')


def infer_loader_preds_2d(model, loader, adj):
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(adj.device)
            pred = model(bx, adj)  # (B, N, 24)
            preds.append(pred.detach().cpu().numpy())

    if len(preds) == 0:
        return np.empty((0, 24), dtype=np.float32)

    pred_3d = np.concatenate(preds, axis=0)  # (num_samples, N, 24)
    return pred_3d.reshape(-1, pred_3d.shape[-1]).astype(np.float32)


def run_full_train_only(model, full_train_loader, adj, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for bx, by in full_train_loader:
            bx = bx.to(adj.device)
            by = by.to(adj.device)
            target = by.permute(0, 2, 1).contiguous()

            optimizer.zero_grad()
            pred = model(bx, adj)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(len(full_train_loader), 1)
        print(f'[FULL] Epoch {epoch + 1}, Loss: {train_loss:.4f}')


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_data = np.load(args.data_path, allow_pickle=True)
    processed_data = process_time_features(raw_data)

    # === 1. Build full sliding-window index space ===
    # Keep exact alignment with XGBoost style formula.
    num_samples = len(processed_data) - args.window_size * 2 + 1
    split_idx = int(num_samples * 0.8)

    # First 80% as OOF candidate pool, last 20% as final test set.
    pool_indices = np.arange(split_idx)
    test_indices = np.arange(split_idx, num_samples)

    # Use the actual training span to compute adjacency.
    train_max_t = split_idx + args.window_size * 2
    adj = compute_adjacency_matrix(
        train_data=processed_data[:train_max_t],
        power_idx=6,
        threshold=args.corr_threshold,
    ).to(device)

    # =========================
    # Layer 1: 3-fold OOF
    # =========================
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    oof_preds_list = []

    for fold_id, (train_idx_idx, val_idx_idx) in enumerate(tscv.split(pool_indices), start=1):
        print(f"\n========== 开始 Fold {fold_id} ==========")

        fold_train_indices = pool_indices[train_idx_idx]
        fold_val_indices = pool_indices[val_idx_idx]

        train_dataset = WindGlobalDataset(
            processed_data,
            fold_train_indices,
            is_train=True,
            window_size=args.window_size,
        )
        val_dataset = WindGlobalDataset(
            processed_data,
            fold_val_indices,
            scaler=train_dataset.scaler,
            is_train=False,
            window_size=args.window_size,
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 【重点】每一折必须重新初始化模型和优化器！
        model = WindGNNLSTM(
            num_nodes=args.num_nodes,
            in_dim=args.input_dim,
            gnn_dim=args.gnn_dim,
            lstm_dim=args.lstm_dim,
            num_layers=args.num_layers,
            output_dim=args.output_dim,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )

        for epoch in range(args.epochs):
            model.train()
            train_loss_sum = 0.0

            for bx, by in train_loader:
                bx = bx.to(device)
                by = by.to(device)
                target = by.permute(0, 2, 1).contiguous()

                optimizer.zero_grad()
                pred = model(bx, adj)
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
            scheduler.step(val_mae)
            print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}')

        val_preds_2d = infer_loader_preds_2d(model, val_loader, adj)
        oof_preds_list.append(val_preds_2d)
        print(f'Fold {fold_id} val preds 2D shape: {val_preds_2d.shape}')

    if len(oof_preds_list) == 0:
        raise RuntimeError('No valid OOF predictions were generated. Check data length and window size.')

    oof_preds_2d = np.concatenate(oof_preds_list, axis=0)
    os.makedirs(args.stacking_dir, exist_ok=True)
    oof_save_path = os.path.join(args.stacking_dir, 'GNN_OOF_Pred.npy')
    np.save(oof_save_path, oof_preds_2d)
    print(f"✅ GNN OOF 保存成功，形状: {oof_preds_2d.shape}")

    # =========================
    # Full training + test inference
    # =========================
    full_train_dataset = WindGlobalDataset(
        processed_data,
        pool_indices,
        is_train=True,
        window_size=args.window_size,
    )
    test_dataset = WindGlobalDataset(
        processed_data,
        test_indices,
        scaler=full_train_dataset.scaler,
        is_train=False,
        window_size=args.window_size,
    )

    full_train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("\n========== 开始全量模型训练 ==========")
    full_model = WindGNNLSTM(
        num_nodes=args.num_nodes,
        in_dim=args.input_dim,
        gnn_dim=args.gnn_dim,
        lstm_dim=args.lstm_dim,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(full_model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.L1Loss()

    for epoch in range(args.epochs):
        full_model.train()
        train_loss_sum = 0.0
        for bx, by in full_train_loader:
            bx = bx.to(device)
            by = by.to(device)
            target = by.permute(0, 2, 1).contiguous()

            optimizer.zero_grad()
            pred = full_model(bx, adj)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(len(full_train_loader), 1)
        print(f'[FULL] Epoch {epoch + 1}, Loss: {train_loss:.4f}')

    full_model.eval()
    test_preds = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            pred = full_model(bx, adj)
            test_preds.append(pred.cpu().numpy())

    test_preds_3d = np.concatenate(test_preds, axis=0)
    test_preds_2d = test_preds_3d.reshape(-1, args.window_size)

    # Save trained full model for Layer-1 ensemble reuse.
    final_model_dir = 'saved_models/final_ensemble'
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, 'layer1_gnn_full.pth')
    torch.save(full_model.state_dict(), final_model_path)

    # =========================
    # Save stacking features
    # =========================
    test_save_path = os.path.join(args.stacking_dir, 'GNN_Test_Pred.npy')

    np.save(test_save_path, test_preds_2d)

    print('\nDone.')
    print(f'OOF predictions saved to: {oof_save_path}, shape={oof_preds_2d.shape}')
    print(f'Test predictions saved to: {test_save_path}, shape={test_preds_2d.shape}')
    print(f'Full model saved to: {final_model_path}')


if __name__ == '__main__':
    main()
