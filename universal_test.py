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
from torch.utils.data import DataLoader, Dataset


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
    def __init__(
        self,
        num_nodes=10,
        in_dim=11,
        gnn_dim=32,
        lstm_dim=64,
        num_layers=2,
        output_dim=24,
        dropout=0.1,
        use_adaptive_adj=False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.use_adaptive_adj = use_adaptive_adj

        if self.use_adaptive_adj:
            self.node_embedding1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
            self.node_embedding2 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)

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
        b, s, n, f = x.shape

        if self.use_adaptive_adj:
            adp_adj = F.softmax(F.relu(torch.mm(self.node_embedding1, self.node_embedding2.t())), dim=-1)
            mixed_adj = 0.7 * adj + 0.3 * adp_adj
        else:
            mixed_adj = adj

        x_features = self.feature_extractor(x)
        x_gcn = self.gcn1(x_features, mixed_adj)
        x_gcn = self.gcn2(x_gcn, mixed_adj)

        x_fused = torch.cat([x_gcn, x], dim=-1)
        x_lstm_in = x_fused.permute(0, 2, 1, 3).contiguous().view(b * n, s, -1)
        _, (h_n, _) = self.lstm(x_lstm_in)

        out = self.fc(h_n[-1])
        out = out.view(b, n, self.output_dim)
        return out


class UniversalDataset(Dataset):
    def __init__(self, data, scaler, config, window_size=24):
        self.window_size = window_size
        self.power_idx = int(config['power_idx'])
        self.weather_idxs = list(config['weather_idxs'])
        self.future_power_fill = config.get('future_power_fill', 'zero')
        self.data = data.copy()

        weather = self.data[:, :, self.weather_idxs].reshape(-1, len(self.weather_idxs))
        self.data[:, :, self.weather_idxs] = scaler.transform(weather).reshape(
            self.data.shape[0], self.data.shape[1], len(self.weather_idxs)
        )

        self.indices = list(range(self.window_size, len(self.data) - self.window_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        past_seq = self.data[t - self.window_size : t, :, :]
        future_seq = self.data[t : t + self.window_size, :, :].copy()

        if self.future_power_fill == 'last':
            last_power = past_seq[-1:, :, self.power_idx]
            future_seq[:, :, self.power_idx] = np.repeat(last_power, self.window_size, axis=0)
        else:
            future_seq[:, :, self.power_idx] = 0.0

        x = np.concatenate([past_seq, future_seq], axis=0).astype(np.float32)
        y = self.data[t : t + self.window_size, :, self.power_idx].astype(np.float32)
        return torch.tensor(x), torch.tensor(y)


class UniversalWindPredictor:
    def __init__(self, model_dir, model_file, device):
        self.device = device
        self.model_dir = model_dir
        self.model_file = model_file

        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'config.json not found in {model_dir}')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

        adj_path = os.path.join(model_dir, 'adjacency.pt')
        if os.path.exists(adj_path):
            self.adj = torch.load(adj_path, map_location=self.device, weights_only=False).to(self.device)
        else:
            self.adj = None

        self.model = self._build_model()
        state = torch.load(os.path.join(model_dir, model_file), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self):
        model_type = self.config.get('model_type', 'GNN_LSTM_v1')

        if model_type in ['GNN_LSTM_v1', 'GNN_LSTM_v2', 'GNN_LSTM']:
            return WindGNNLSTM(
                num_nodes=int(self.config.get('num_nodes', 10)),
                in_dim=int(self.config['input_dim']),
                gnn_dim=int(self.config['gnn_dim']),
                lstm_dim=int(self.config['lstm_dim']),
                num_layers=int(self.config['num_layers']),
                output_dim=int(self.config['output_dim']),
                dropout=float(self.config.get('dropout', 0.2)),
                use_adaptive_adj=bool(self.config.get('use_adaptive_adj', False)),
            )

        raise ValueError(f'Unsupported model_type: {model_type}')

    def preprocess_data(self, raw_data):
        numeric_data = raw_data[:, :, 1:].astype(float)

        # 逐风场沿时间轴插值缺失值
        for n in range(numeric_data.shape[1]):
            df = pd.DataFrame(numeric_data[:, n, :])
            df.interpolate(method='linear', limit_direction='both', inplace=True)
            numeric_data[:, n, :] = df.values

        feat_list = []
        weather_uv = numeric_data[:, :, 0:4]
        feat_list.append(weather_uv)

        if self.config.get('use_synthetic_wind', True):
            u10, v10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
            u100, v100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
            ws10 = np.sqrt(np.square(u10) + np.square(v10))[..., np.newaxis]
            ws100 = np.sqrt(np.square(u100) + np.square(v100))[..., np.newaxis]
            feat_list.extend([ws10, ws100])

        power = numeric_data[:, :, 4:5]
        feat_list.append(power)

        if self.config.get('use_time_encode', True):
            timestamps = pd.to_datetime(raw_data[:, 0, 0])
            hours = timestamps.hour.values.astype(np.float32)
            months = timestamps.month.values.astype(np.float32)

            hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
            hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
            month_sin = np.sin(2.0 * np.pi * months / 12.0)
            month_cos = np.cos(2.0 * np.pi * months / 12.0)
            time_feats = np.stack([hour_sin, hour_cos, month_sin, month_cos], axis=-1)
            time_feats = np.repeat(time_feats[:, np.newaxis, :], raw_data.shape[1], axis=1)
            feat_list.append(time_feats)

        processed = np.concatenate(feat_list, axis=-1).astype(np.float32)
        if np.isnan(processed).sum() > 0:
            processed = np.nan_to_num(processed, nan=0.0)

        expected_dim = int(self.config['input_dim'])
        if processed.shape[-1] != expected_dim:
            raise ValueError(
                f"Feature dim mismatch: got {processed.shape[-1]}, expected {expected_dim} from config.json"
            )

        return processed

    def evaluate(self, raw_test_data, batch_size=64):
        processed = self.preprocess_data(raw_test_data)

        dataset = UniversalDataset(processed, self.scaler, self.config, window_size=24)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        mae_sum = 0.0
        mse_sum = 0.0

        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                target = by.permute(0, 2, 1).contiguous()

                if self.adj is None:
                    raise RuntimeError('adjacency.pt is required for GNN_LSTM models')

                pred = self.model(bx, self.adj)
                mae_sum += torch.sum(torch.abs(pred - target)).item()
                mse_sum += torch.sum(torch.square(pred - target)).item()

        total = len(dataset) * int(self.config.get('num_nodes', 10)) * int(self.config['output_dim'])
        test_mae = mae_sum / total
        test_rmse = np.sqrt(mse_sum / total)
        return test_mae, test_rmse, len(dataset)


def main():
    parser = argparse.ArgumentParser(description='Universal test script for wind forecasting models')
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--test-path', type=str, default='data/wind_test_2013-07-14_to_end.npy')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = UniversalWindPredictor(args.model_dir, args.model_file, device)

    raw_test_data = np.load(args.test_path, allow_pickle=True)
    test_mae, test_rmse, n_samples = predictor.evaluate(raw_test_data, batch_size=args.batch_size)

    print('-' * 50)
    print('Universal Test Report')
    print('-' * 50)
    print(f'Model Type: {predictor.config.get("model_type", "Unknown")}')
    print(f'Test Samples: {n_samples}')
    print(f'Test MAE  : {test_mae:.4f}')
    print(f'Test RMSE : {test_rmse:.4f}')
    print('-' * 50)

    model_stem = os.path.splitext(os.path.basename(args.model_file))[0]
    report_path = os.path.join(args.model_dir, f'universal_test_report_{model_stem}.txt')
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('Universal Wind Test Report\n')
        f.write('=' * 50 + '\n')
        f.write(f'Time: {now_str}\n')
        f.write(f'Model Directory: {args.model_dir}\n')
        f.write(f'Model File: {args.model_file}\n')
        f.write(f'Model Type: {predictor.config.get("model_type", "Unknown")}\n')
        f.write(f'Test Data: {args.test_path}\n')
        f.write('-' * 50 + '\n')
        f.write(f'Test MAE: {test_mae:.6f}\n')
        f.write(f'Test RMSE: {test_rmse:.6f}\n')
        f.write('=' * 50 + '\n')

    print(f'Report saved: {report_path}')


if __name__ == '__main__':
    main()
