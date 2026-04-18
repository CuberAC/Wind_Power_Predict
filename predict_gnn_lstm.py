import argparse
import json
import os

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
        aggregated = torch.einsum("ij,bslj->bsli", adj, support.transpose(2, 3)).transpose(2, 3)
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
        dropout=0.2,
    ):
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

        out = self.fc(h_n[-1])
        out = out.view(b, n, self.output_dim)
        return out


class UniversalDataset(Dataset):
    def __init__(self, data, scaler, config, window_size=24):
        self.window_size = window_size
        self.power_idx = int(config["power_idx"])
        self.weather_idxs = list(config["weather_idxs"])
        self.future_power_fill = config.get("future_power_fill", "zero")
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

        if self.future_power_fill == "last":
            last_power = past_seq[-1:, :, self.power_idx]
            future_seq[:, :, self.power_idx] = np.repeat(last_power, self.window_size, axis=0)
        else:
            future_seq[:, :, self.power_idx] = 0.0

        x = np.concatenate([past_seq, future_seq], axis=0).astype(np.float32)
        y = self.data[t : t + self.window_size, :, self.power_idx].astype(np.float32)
        return torch.tensor(x), torch.tensor(y)


class GNNLSTMPredictor:
    def __init__(self, model_dir, model_file, device):
        self.device = device
        self.model_dir = model_dir
        self.model_file = model_file

        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"找不到 scaler: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        adj_path = os.path.join(model_dir, "adjacency.pt")
        if not os.path.exists(adj_path):
            raise FileNotFoundError(f"找不到邻接矩阵文件: {adj_path}")
        self.adj = torch.load(adj_path, map_location=self.device, weights_only=False).to(self.device)

        self.model = self._build_model()
        state = torch.load(os.path.join(model_dir, model_file), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self):
        model_type = self.config.get("model_type", "GNN_LSTM_v1")
        if model_type not in ["GNN_LSTM_v1", "GNN_LSTM_v2", "GNN_LSTM"]:
            raise ValueError(f"当前脚本仅支持 GNN_LSTM 系列，收到 model_type={model_type}")

        return WindGNNLSTM(
            num_nodes=int(self.config.get("num_nodes", 10)),
            in_dim=int(self.config["input_dim"]),
            gnn_dim=int(self.config["gnn_dim"]),
            lstm_dim=int(self.config["lstm_dim"]),
            num_layers=int(self.config["num_layers"]),
            output_dim=int(self.config["output_dim"]),
            dropout=float(self.config.get("dropout", 0.2)),
        )

    def preprocess_data(self, raw_data):
        numeric_data = raw_data[:, :, 1:].astype(float)

        for n in range(numeric_data.shape[1]):
            df = pd.DataFrame(numeric_data[:, n, :])
            df.interpolate(method="linear", limit_direction="both", inplace=True)
            numeric_data[:, n, :] = df.values

        feat_list = []
        weather_uv = numeric_data[:, :, 0:4]
        feat_list.append(weather_uv)

        if self.config.get("use_synthetic_wind", True):
            u10, v10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
            u100, v100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
            ws10 = np.sqrt(np.square(u10) + np.square(v10))[..., np.newaxis]
            ws100 = np.sqrt(np.square(u100) + np.square(v100))[..., np.newaxis]
            feat_list.extend([ws10, ws100])

        power = numeric_data[:, :, 4:5]
        feat_list.append(power)

        if self.config.get("use_time_encode", True):
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

        expected_dim = int(self.config["input_dim"])
        if processed.shape[-1] != expected_dim:
            raise ValueError(
                f"特征维度不匹配: 实际 {processed.shape[-1]}, 配置期望 {expected_dim}"
            )

        return processed


def run_inference(predictor, raw_data, batch_size=64, window_size=24, clip_to_unit=True):
    processed = predictor.preprocess_data(raw_data)
    dataset = UniversalDataset(processed, predictor.scaler, predictor.config, window_size=window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pred_chunks = []

    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(predictor.device)

            pred = predictor.model(bx, predictor.adj).detach().cpu().numpy().astype(np.float32)
            if clip_to_unit:
                pred = np.clip(pred, 0.0, 1.0)

            pred_chunks.append(pred)

    if not pred_chunks:
        raise ValueError("未生成任何预测样本，请检查测试集长度与 window_size")

    preds = np.concatenate(pred_chunks, axis=0)
    return preds, len(dataset)


def main():
    parser = argparse.ArgumentParser(description="GNN_LSTM_v1 纯预测脚本")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="saved_models/GNN_LSTM",
        help="模型目录（需包含 config.json/scaler.pkl/adjacency.pt）",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="gnn_lstm_epoch_80.pth",
        help="模型权重文件名",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/wind_test_cleaned.npy",
        help="测试集路径",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_models/GNN_LSTM",
        help="输出目录",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="默认会将预测裁剪到[0,1]，如不需要可加此参数关闭",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="predictions.npy",
        help="预测文件名",
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"找不到测试集文件: {args.test_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = GNNLSTMPredictor(args.model_dir, args.model_file, device)

    raw_data = np.load(args.test_path, allow_pickle=True)

    preds, n_samples = run_inference(
        predictor=predictor,
        raw_data=raw_data,
        batch_size=args.batch_size,
        window_size=args.window_size,
        clip_to_unit=(not args.no_clip),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, args.output_file)
    np.save(pred_path, preds)

    print("=" * 60)
    print("GNN_LSTM_v1 纯预测完成")
    print(f"测试集: {args.test_path}")
    print(f"样本数: {n_samples}")
    print(f"预测文件: {pred_path}")
    print(f"预测数组形状: {preds.shape} (Sample, Farm, Horizon)")
    print("=" * 60)


if __name__ == "__main__":
    main()
