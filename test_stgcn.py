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


def process_advanced_features(raw_data):
    """
    输入:
        raw_data: np.ndarray, [T, N, F_raw]
          - idx 0: time
          - idx 1: U10
          - idx 2: V10
          - idx 3: U100
          - idx 4: V100
          - idx 5: Power

    输出:
        features: np.ndarray, [T, N, 13]
        [U10, V10, U100, V100, WS10, WD10, WS100, WD100,
         Power, hour_sin, hour_cos, month_sin, month_cos]
    """
    if raw_data.ndim != 3:
        raise ValueError(f"raw_data 应为 3 维 [T, N, F_raw]，实际为 {raw_data.shape}")
    if raw_data.shape[-1] < 6:
        raise ValueError("raw_data 最后一维至少包含索引 0..5")

    u10 = raw_data[:, :, 1].astype(np.float32)
    v10 = raw_data[:, :, 2].astype(np.float32)
    u100 = raw_data[:, :, 3].astype(np.float32)
    v100 = raw_data[:, :, 4].astype(np.float32)
    power = raw_data[:, :, 5].astype(np.float32)

    ws10 = np.sqrt(u10 ** 2 + v10 ** 2).astype(np.float32)
    wd10 = np.arctan2(v10, u10).astype(np.float32)
    ws100 = np.sqrt(u100 ** 2 + v100 ** 2).astype(np.float32)
    wd100 = np.arctan2(v100, u100).astype(np.float32)

    time_col = raw_data[:, 0, 0]
    dt = pd.to_datetime(time_col)

    hour = dt.hour.to_numpy(dtype=np.float32)
    month = dt.month.to_numpy(dtype=np.float32)

    hour_sin = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    month_sin = np.sin(2.0 * np.pi * (month - 1.0) / 12.0).astype(np.float32)
    month_cos = np.cos(2.0 * np.pi * (month - 1.0) / 12.0).astype(np.float32)

    num_nodes = raw_data.shape[1]
    hour_sin = np.repeat(hour_sin[:, None], num_nodes, axis=1)
    hour_cos = np.repeat(hour_cos[:, None], num_nodes, axis=1)
    month_sin = np.repeat(month_sin[:, None], num_nodes, axis=1)
    month_cos = np.repeat(month_cos[:, None], num_nodes, axis=1)

    features = np.stack(
        [
            u10,
            v10,
            u100,
            v100,
            ws10,
            wd10,
            ws100,
            wd100,
            power,
            hour_sin,
            hour_cos,
            month_sin,
            month_cos,
        ],
        axis=-1,
    ).astype(np.float32)

    return features


class GraphLearning(nn.Module):
    def __init__(self, num_nodes, embed_dim=10, alpha=3.0):
        super().__init__()
        self.alpha = alpha
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))

    def forward(self):
        A = F.relu(torch.tanh(self.alpha * torch.mm(self.E1, self.E2.t())))
        A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A = F.softmax(A, dim=-1)
        return A


class TemporalInception(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv_k2 = nn.Conv2d(c_in, c_out, kernel_size=(1, 2), padding="same")
        self.conv_k3 = nn.Conv2d(c_in, c_out, kernel_size=(1, 3), padding="same")
        self.conv_k5 = nn.Conv2d(c_in, c_out, kernel_size=(1, 5), padding="same")
        self.conv_k7 = nn.Conv2d(c_in, c_out, kernel_size=(1, 7), padding="same")

    def forward(self, x):
        x2 = self.conv_k2(x)
        x3 = self.conv_k3(x)
        x5 = self.conv_k5(x)
        x7 = self.conv_k7(x)
        return x2 + x3 + x5 + x7


class SpatialGCN(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.theta = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x, A):
        support = self.theta(x)
        support_tn = support.permute(0, 1, 3, 2)
        out_tn = torch.einsum("vw,bcfw->bcfv", A, support_tn)
        out = out_tn.permute(0, 1, 3, 2)
        return self.act(out)


class STGCNBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.t1 = TemporalInception(hidden_dim, hidden_dim)
        self.s = SpatialGCN(hidden_dim, hidden_dim)
        self.t2 = TemporalInception(hidden_dim, hidden_dim)
        self.norm = nn.BatchNorm2d(hidden_dim)

    def forward(self, x, A):
        out = self.t1(x)
        out = self.s(out, A)
        out = self.t2(out)
        out = self.norm(out)
        return out


class WindSTGCN(nn.Module):
    def __init__(
        self,
        num_nodes=10,
        in_dim=13,
        hidden_dim=64,
        output_dim=24,
        embed_dim=10,
        alpha=3.0,
        dropout=0.1,
        input_seq_len=48,
        num_blocks=3,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks 必须 >= 1，当前 {num_blocks}")
        self.input_seq_len = input_seq_len
        self.input_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1))
        self.graph_learning = GraphLearning(num_nodes=num_nodes, embed_dim=embed_dim, alpha=alpha)
        self.blocks = nn.ModuleList([STGCNBlock(hidden_dim=hidden_dim) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.end_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=128,
                kernel_size=(1, input_seq_len),
            ),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.input_proj(x)

        A = self.graph_learning()
        for block in self.blocks:
            res = x
            x = block(x, A)
            x = x + res
        x = self.dropout(x)

        if x.shape[-1] != self.input_seq_len:
            raise ValueError(
                f"输入序列长度不匹配: 模型期望 {self.input_seq_len}，实际 {x.shape[-1]}"
            )

        x_end = self.end_conv(x)
        x_end = x_end.squeeze(-1)
        x_end = x_end.permute(0, 2, 1)
        out = self.fc(x_end)
        return out


def clean_test_data(raw_data):
    """
    对每个节点做线性插值，处理 NaN。
    """
    if raw_data.ndim != 3:
        raise ValueError(f"raw_data 应为 3 维 [T, N, F_raw]，实际为 {raw_data.shape}")

    cleaned = raw_data.copy().astype(object)
    T, N, F_raw = cleaned.shape

    for node in range(N):
        node_data = cleaned[:, node, :]
        df = pd.DataFrame(node_data)

        # 时间列单独保留，仅做前后填充
        time_col = df.iloc[:, 0].copy()
        time_col = time_col.ffill().bfill()

        # 数值列线性插值
        for col in range(1, F_raw):
            numeric_col = pd.to_numeric(df.iloc[:, col], errors="coerce")
            numeric_col = numeric_col.interpolate(method="linear", limit_direction="both")
            df.iloc[:, col] = numeric_col

        df.iloc[:, 0] = time_col
        cleaned[:, node, :] = df.to_numpy(dtype=object)

    return cleaned


class WindSTGCNTestDataset(Dataset):
    def __init__(self, features, window_size=24, power_idx=8):
        self.features = features.astype(np.float32)
        self.window_size = window_size
        self.power_idx = power_idx

        min_required = 2 * self.window_size
        if len(self.features) < min_required:
            raise ValueError(
                f"测试数据长度不足，至少需要 {min_required} 个时间步，当前 {len(self.features)}"
            )

    def __len__(self):
        return len(self.features) - 2 * self.window_size + 1

    def __getitem__(self, idx):
        w = self.window_size
        hist = self.features[idx : idx + w]
        fut = self.features[idx + w : idx + 2 * w]

        future_template = np.repeat(hist[-1:, :, :], repeats=w, axis=0)
        future_template[:, :, self.power_idx] = 0.0

        x = np.concatenate([hist, future_template], axis=0)
        y = fut[:, :, self.power_idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def parse_args():
    parser = argparse.ArgumentParser(description="Universal STGCN test script")
    parser.add_argument("--test-path", type=str, default="data/wind_test_cleaned.npy")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.join(args.model_dir, "config.json")
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")
    ckpt_path = os.path.join(args.model_dir, "best_model.pth")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"未找到 scaler 文件: {scaler_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型权重: {ckpt_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    scaler = joblib.load(scaler_path)

    raw_test = np.load(args.test_path, allow_pickle=True)
    raw_test = clean_test_data(raw_test)
    test_feat = process_advanced_features(raw_test)

    normalize_cols = config.get("normalize_cols", [0, 1, 2, 3, 4, 6])
    test_2d = test_feat[:, :, normalize_cols].reshape(-1, len(normalize_cols))
    test_feat[:, :, normalize_cols] = scaler.transform(test_2d).reshape(
        test_feat.shape[0], test_feat.shape[1], len(normalize_cols)
    )

    window_size = int(config.get("window_size", 24))
    power_idx = int(config.get("power_idx", 8))

    test_dataset = WindSTGCNTestDataset(
        test_feat,
        window_size=window_size,
        power_idx=power_idx,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = WindSTGCN(
        num_nodes=int(config.get("num_nodes", test_feat.shape[1])),
        in_dim=int(config.get("in_dim", 13)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        output_dim=int(config.get("window_size", 24)),
        embed_dim=int(config.get("embed_dim", 10)),
        alpha=float(config.get("alpha", 3.0)),
        dropout=float(config.get("dropout", 0.1)),
        input_seq_len=int(config.get("input_seq_len", 2 * int(config.get("window_size", 24)))),
        num_blocks=int(config.get("num_blocks", 3)),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    abs_sum = 0.0
    sq_sum = 0.0
    n_points = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device, dtype=torch.float32).permute(0, 2, 1)

            pred = model(x_batch)
            diff = pred - y_batch

            abs_sum += diff.abs().sum().item()
            sq_sum += (diff ** 2).sum().item()
            n_points += diff.numel()

    mae = abs_sum / max(n_points, 1)
    rmse = float(np.sqrt(sq_sum / max(n_points, 1)))

    print("=" * 60)
    print("STGCN Test Results")
    print("=" * 60)
    print(f"Model Dir : {args.model_dir}")
    print(f"Test Path : {args.test_path}")
    print(f"Samples   : {len(test_dataset)}")
    print(f"MAE       : {mae:.6f}")
    print(f"RMSE      : {rmse:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
