import argparse
import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

    def forward(self):
        scores = torch.mm(self.E1, self.E2.t()) / (self.E1.size(1) ** 0.5)
        A = F.relu(torch.tanh(scores))
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
        # [B, T, N, F] -> [B, F, N, T]
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

        # [B, H, N, T] -> [B, 128, N, 1]
        x_end = self.end_conv(x)
        # [B, 128, N]
        x_end = x_end.squeeze(-1)
        # [B, N, 128]
        x_end = x_end.permute(0, 2, 1)
        out = self.fc(x_end)  # [B, N, output_dim]
        return out


class WindSTGCNDataset(Dataset):
    """
    输入 data: [T, N, 13]
    输出:
      X: [2*window_size, N, 13]  (前24真实 + 后24模板，其中 power 置0)
      Y: [window_size, N]         (未来24小时真实 power)
    """

    def __init__(self, data, window_size=24, power_idx=8):
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.power_idx = power_idx

        min_required = 2 * self.window_size
        if len(self.data) < min_required:
            raise ValueError(
                f"数据长度不足，至少需要 {min_required} 个时间步，当前 {len(self.data)}"
            )

    def __len__(self):
        return len(self.data) - 2 * self.window_size + 1

    def __getitem__(self, idx):
        w = self.window_size
        hist = self.data[idx : idx + w]  # [w, N, F]
        fut = self.data[idx + w : idx + 2 * w]  # [w, N, F]

        future_template = np.repeat(hist[-1:, :, :], repeats=w, axis=0)
        future_template[:, :, self.power_idx] = 0.0

        x = np.concatenate([hist, future_template], axis=0)  # [2w, N, F]
        y = fut[:, :, self.power_idx]  # [w, N]

        return torch.from_numpy(x), torch.from_numpy(y)


def parse_args():
    parser = argparse.ArgumentParser(description="Train STGCN baseline for wind power forecasting")
    parser.add_argument("--data-path", type=str, default="data/wind_train_val_2012-01-02_to_2013-07-13.npy")
    parser.add_argument("--save-root", type=str, default="saved_models/STGCN_models")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    raw_data = np.load(args.data_path, allow_pickle=True)
    features = process_advanced_features(raw_data)  # [T, N, 13]

    total_steps = features.shape[0]
    split_idx = int(total_steps * 0.8)
    if split_idx < 2 * args.window_size:
        raise ValueError("训练集长度不足，无法按窗口切片，请减小 window-size 或检查数据")
    if (total_steps - split_idx) < 2 * args.window_size:
        raise ValueError("验证集长度不足，无法按窗口切片，请减小 window-size 或检查数据")

    train_feat = features[:split_idx].copy()
    val_feat = features[split_idx:].copy()

    # 只归一化 [U10, V10, U100, V100, WS10, WS100]，严格不处理 power/wd/time
    normalize_cols = [0, 1, 2, 3, 4, 6]
    scaler = StandardScaler()
    train_2d = train_feat[:, :, normalize_cols].reshape(-1, len(normalize_cols))
    scaler.fit(train_2d)

    train_feat[:, :, normalize_cols] = scaler.transform(train_2d).reshape(
        train_feat.shape[0], train_feat.shape[1], len(normalize_cols)
    )

    val_2d = val_feat[:, :, normalize_cols].reshape(-1, len(normalize_cols))
    val_feat[:, :, normalize_cols] = scaler.transform(val_2d).reshape(
        val_feat.shape[0], val_feat.shape[1], len(normalize_cols)
    )

    scaler_path = os.path.join(save_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    train_dataset = WindSTGCNDataset(
        train_feat,
        window_size=args.window_size,
        power_idx=8,
    )
    val_dataset = WindSTGCNDataset(
        val_feat,
        window_size=args.window_size,
        power_idx=8,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_nodes = features.shape[1]
    device = torch.device(args.device)

    model = WindSTGCN(
        num_nodes=num_nodes,
        in_dim=13,
        hidden_dim=args.hidden_dim,
        output_dim=args.window_size,
        embed_dim=args.embed_dim,
        alpha=args.alpha,
        dropout=args.dropout,
        input_seq_len=2 * args.window_size,
        num_blocks=args.num_blocks,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    criterion_train = nn.MSELoss()

    config_dict = vars(args).copy()
    config_dict.update(
        {
            "model_name": "STGCN_baseline",
            "timestamp": timestamp,
            "save_dir": save_dir,
            "num_nodes": int(num_nodes),
            "in_dim": 13,
            "power_idx": 8,
            "input_seq_len": int(2 * args.window_size),
            "num_blocks": int(args.num_blocks),
            "normalize_cols": normalize_cols,
            "feature_order": [
                "U10",
                "V10",
                "U100",
                "V100",
                "WS10",
                "WD10",
                "WS100",
                "WD100",
                "Power",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
            ],
            "train_steps": int(split_idx),
            "val_steps": int(total_steps - split_idx),
        }
    )

    best_val_mae = float("inf")
    no_improve = 0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device=device, dtype=torch.float32).permute(0, 2, 1)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_batch)
            loss = criterion_train(pred, y_batch)
            loss.backward()
            optimizer.step()

            bs = x_batch.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_abs_sum = 0.0
        val_sq_sum = 0.0
        val_points = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device=device, dtype=torch.float32)
                y_batch = y_batch.to(device=device, dtype=torch.float32).permute(0, 2, 1)
                pred = model(x_batch)

                diff = pred - y_batch
                val_abs_sum += diff.abs().sum().item()
                val_sq_sum += (diff ** 2).sum().item()
                val_points += diff.numel()

        val_mae = val_abs_sum / max(val_points, 1)
        val_rmse = float(np.sqrt(val_sq_sum / max(val_points, 1)))
        scheduler.step(val_mae)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"TrainLoss={train_loss:.6f} ValMAE={val_mae:.6f} ValRMSE={val_rmse:.6f} LR={current_lr:.2e}"
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "lr": current_lr,
            }
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve = 0

            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}, best ValMAE={best_val_mae:.6f}")
            break

    np.savetxt(
        os.path.join(save_dir, "train_log.csv"),
        np.array([[r["epoch"], r["train_loss"], r["val_mae"], r["val_rmse"], r["lr"]] for r in log_rows]),
        delimiter=",",
        header="epoch,train_loss,val_mae,val_rmse,lr",
        comments="",
    )

    print(f"Training done. Artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
