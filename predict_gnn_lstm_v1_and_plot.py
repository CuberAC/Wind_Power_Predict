import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from universal_test import UniversalDataset, UniversalWindPredictor


def build_ws_future(raw_data, window_size=24):
    """构建与 Dataset 索引对齐的未来 24h 100m 合成风速，形状 (Samples, Nodes, Horizon)。"""
    u100 = raw_data[:, :, 3].astype(np.float32)
    v100 = raw_data[:, :, 4].astype(np.float32)
    ws100 = np.sqrt(np.square(u100) + np.square(v100)).astype(np.float32)

    indices = list(range(window_size, len(raw_data) - window_size))
    ws_rows = []
    for t in indices:
        ws_rows.append(ws100[t : t + window_size, :].transpose(1, 0))
    return np.asarray(ws_rows, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def run_inference(predictor, raw_data, batch_size=64, window_size=24, clip_to_unit=True):
    processed = predictor.preprocess_data(raw_data)
    dataset = UniversalDataset(processed, predictor.scaler, predictor.config, window_size=window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pred_chunks = []
    true_chunks = []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(predictor.device)
            target = by.permute(0, 2, 1).contiguous().numpy().astype(np.float32)

            pred = predictor.model(bx, predictor.adj).detach().cpu().numpy().astype(np.float32)
            if clip_to_unit:
                pred = np.clip(pred, 0.0, 1.0)

            pred_chunks.append(pred)
            true_chunks.append(target)

    preds = np.concatenate(pred_chunks, axis=0)
    y_true = np.concatenate(true_chunks, axis=0)
    return preds, y_true, len(dataset)


def save_metrics(preds, y_true, report_path):
    global_mae = float(mean_absolute_error(y_true.reshape(-1), preds.reshape(-1)))
    global_rmse = float(np.sqrt(mean_squared_error(y_true.reshape(-1), preds.reshape(-1))))

    per_farm = []
    for farm_id in range(preds.shape[1]):
        farm_true = y_true[:, farm_id, :].reshape(-1)
        farm_pred = preds[:, farm_id, :].reshape(-1)
        farm_mae = float(mean_absolute_error(farm_true, farm_pred))
        farm_rmse = float(np.sqrt(mean_squared_error(farm_true, farm_pred)))
        per_farm.append((farm_id, farm_mae, farm_rmse))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("GNN_LSTM_v1 Validation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Global MAE: {global_mae:.6f}\n")
        f.write(f"Global RMSE: {global_rmse:.6f}\n")
        f.write("\nPer-Farm Metrics:\n")
        for farm_id, farm_mae, farm_rmse in per_farm:
            f.write(f"Farm {farm_id}: MAE={farm_mae:.6f}, RMSE={farm_rmse:.6f}\n")

    return global_mae, global_rmse


def plot_power_curve_scatter(preds, y_true, ws_future, out_dir):
    """每个风场一张图：横轴风速，纵轴出力，真实/预测同图散点对比。"""
    os.makedirs(out_dir, exist_ok=True)

    for farm_id in range(preds.shape[1]):
        ws_flat = ws_future[:, farm_id, :].reshape(-1)
        y_true_flat = y_true[:, farm_id, :].reshape(-1)
        y_pred_flat = preds[:, farm_id, :].reshape(-1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(ws_flat, y_true_flat, s=8, alpha=0.35, label="True Power", color="tab:blue")
        ax.scatter(ws_flat, y_pred_flat, s=8, alpha=0.35, label="Pred Power", color="tab:orange")
        ax.set_title(f"Farm {farm_id} | Wind Speed vs Power")
        ax.set_xlabel("Wind Speed (100m)")
        ax.set_ylabel("Power")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()

        fig.savefig(os.path.join(out_dir, f"farm_{farm_id}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_first_10_days_timeseries(preds, y_true, timestamps, sample_indices, out_dir):
    """每个风场一张图：十天时间序列（真实 vs 预测）。使用一步预测轨迹。"""
    os.makedirs(out_dir, exist_ok=True)

    time_axis = timestamps[sample_indices]
    points_10d = min(24 * 10, len(time_axis))

    for farm_id in range(preds.shape[1]):
        true_1step = y_true[:, farm_id, 0]
        pred_1step = preds[:, farm_id, 0]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_axis[:points_10d], true_1step[:points_10d], label="True Power", color="tab:blue")
        ax.plot(time_axis[:points_10d], pred_1step[:points_10d], label="Pred Power", color="tab:orange")
        ax.set_title(f"Farm {farm_id} | First 10 Days Power Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Power")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()

        fig.savefig(os.path.join(out_dir, f"farm_{farm_id}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GNN_LSTM_v1 调用与可视化脚本")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="saved_models/GNN_LSTM/GNN_LSTM_v1",
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
        default="data/wind_val_20.npy",
        help="测试集路径",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="saved_models/GNN_LSTM/GNN_LSTM_v1/inference_wind_val_20",
        help="输出目录",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="默认会将预测裁剪到[0,1]，如不需要可加此参数关闭",
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"找不到测试集文件: {args.test_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = UniversalWindPredictor(args.model_dir, args.model_file, device)

    raw_data = np.load(args.test_path, allow_pickle=True)
    timestamps = pd.to_datetime(raw_data[:, 0, 0]).to_numpy()

    preds, y_true, n_samples = run_inference(
        predictor=predictor,
        raw_data=raw_data,
        batch_size=args.batch_size,
        window_size=args.window_size,
        clip_to_unit=(not args.no_clip),
    )

    ws_future, sample_indices = build_ws_future(raw_data, window_size=args.window_size)
    if ws_future.shape[0] != preds.shape[0]:
        raise ValueError(f"样本对齐失败: ws={ws_future.shape[0]}, pred={preds.shape[0]}")

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "predictions.npy")
    true_path = os.path.join(args.output_dir, "ground_truth.npy")
    np.save(pred_path, preds)
    np.save(true_path, y_true)

    report_path = os.path.join(args.output_dir, "report.txt")
    global_mae, global_rmse = save_metrics(preds, y_true, report_path)

    scatter_dir = os.path.join(args.output_dir, "power_curve_scatter_by_farm")
    ts_dir = os.path.join(args.output_dir, "time_series_10days_by_farm")
    plot_power_curve_scatter(preds, y_true, ws_future, scatter_dir)
    plot_first_10_days_timeseries(preds, y_true, timestamps, sample_indices, ts_dir)

    print("=" * 60)
    print("GNN_LSTM_v1 推理与可视化完成")
    print(f"测试集: {args.test_path}")
    print(f"样本数: {n_samples}")
    print(f"Global MAE: {global_mae:.6f}")
    print(f"Global RMSE: {global_rmse:.6f}")
    print(f"预测文件: {pred_path}")
    print(f"真实值文件: {true_path}")
    print(f"散点图目录: {scatter_dir}")
    print(f"十天曲线图目录: {ts_dir}")
    print(f"报告文件: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
