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
    """Global dataset that keeps Node dimension and outputs all 10 farms together."""

    def __init__(self, data_segment, mode='train', window_size=24, scaler=None):
        self.window_size = window_size
        self.power_idx = 6
        self.weather_idxs = [0, 1, 2, 3, 4, 5]
        self.time_idxs = [7, 8, 9, 10]
        self.future_cov_idxs = self.weather_idxs + self.time_idxs

        self.segment = data_segment
        self.scaler = scaler

        if mode == 'train':
            self.scaler = StandardScaler()
            weather_flat = self.segment[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
            self.scaler.fit(weather_flat)
        elif mode in ('val', 'test'):
            if self.scaler is None:
                raise ValueError(f'{mode} mode requires a fitted scaler from training set.')
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        self.data = self.segment.copy()
        segment_weather = self.segment[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
        self.data[:, :, self.weather_idxs] = self.scaler.transform(segment_weather).reshape(
            self.segment.shape[0],
            self.segment.shape[1],
            len(self.weather_idxs),
        )

        self.indices = list(range(self.window_size, len(self.data) - self.window_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        past_seq = self.data[t - self.window_size:t, :, :]

        future_seq = self.data[t:t + self.window_size, :, :].copy()
        future_seq[:, :, self.power_idx] = 0.0

        x = np.concatenate([past_seq, future_seq], axis=0).astype(np.float32)
        y = self.data[t:t + self.window_size, :, self.power_idx].astype(np.float32)

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

    # Split timeline into pool (for OOF) and holdout test.
    split_idx = int(len(processed_data) * args.pool_ratio)
    pool_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]

    if len(pool_data) <= 2 * args.window_size:
        raise ValueError('pool_data is too short for windowed training.')
    if len(test_data) <= 2 * args.window_size:
        raise ValueError('test_data is too short for windowed inference.')

    # =========================
    # Layer 1: 3-fold OOF
    # =========================
    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    oof_preds_list = []

    for fold_id, (train_idx, val_idx) in enumerate(tscv.split(np.arange(len(pool_data))), start=1):
        fold_train_data = pool_data[train_idx]
        fold_val_data = pool_data[val_idx]

        print(f'\n===== Fold {fold_id}/{args.n_splits} =====')
        print(f'Fold train length: {len(fold_train_data)}, val length: {len(fold_val_data)}')

        if len(fold_train_data) <= 2 * args.window_size or len(fold_val_data) <= 2 * args.window_size:
            print('Skip fold due to insufficient samples after windowing.')
            continue

        adj = compute_adjacency_matrix(
            train_data=fold_train_data,
            power_idx=6,
            threshold=args.corr_threshold,
        ).to(device)

        train_dataset = WindGlobalDataset(
            data_segment=fold_train_data,
            mode='train',
            window_size=args.window_size,
        )
        val_dataset = WindGlobalDataset(
            data_segment=fold_val_data,
            mode='val',
            window_size=args.window_size,
            scaler=train_dataset.scaler,
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = build_model(args, device)
        run_train_eval_epochs(model, train_loader, val_loader, adj, args)

        val_preds_2d = infer_loader_preds_2d(model, val_loader, adj)
        oof_preds_list.append(val_preds_2d)
        print(f'Fold {fold_id} val preds 2D shape: {val_preds_2d.shape}')

    if len(oof_preds_list) == 0:
        raise RuntimeError('No valid OOF predictions were generated. Check data length and window size.')

    oof_preds_2d = np.concatenate(oof_preds_list, axis=0)

    # =========================
    # Full training + test inference
    # =========================
    full_adj = compute_adjacency_matrix(
        train_data=pool_data,
        power_idx=6,
        threshold=args.corr_threshold,
    ).to(device)

    full_train_dataset = WindGlobalDataset(
        data_segment=pool_data,
        mode='train',
        window_size=args.window_size,
    )
    test_dataset = WindGlobalDataset(
        data_segment=test_data,
        mode='test',
        window_size=args.window_size,
        scaler=full_train_dataset.scaler,
    )

    full_train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    full_model = build_model(args, device)
    run_full_train_only(full_model, full_train_loader, full_adj, args)

    test_preds_2d = infer_loader_preds_2d(full_model, test_loader, full_adj)

    # =========================
    # Save stacking features
    # =========================
    os.makedirs(args.stacking_dir, exist_ok=True)
    oof_save_path = os.path.join(args.stacking_dir, 'GNN_OOF_Pred.npy')
    test_save_path = os.path.join(args.stacking_dir, 'GNN_Test_Pred.npy')

    np.save(oof_save_path, oof_preds_2d)
    np.save(test_save_path, test_preds_2d)

    print('\nDone.')
    print(f'OOF predictions saved to: {oof_save_path}, shape={oof_preds_2d.shape}')
    print(f'Test predictions saved to: {test_save_path}, shape={test_preds_2d.shape}')


if __name__ == '__main__':
    main()
