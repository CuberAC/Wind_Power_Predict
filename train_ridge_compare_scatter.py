import argparse
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def add_physical_features(ws_array):
    """当前增强版使用的物理边界和布尔开关特征。"""
    cut_in = 3.71
    rated = 4.50
    cut_out = 15.0

    ws = np.asarray(ws_array, dtype=np.float32)
    is_zero_power = ((ws < cut_in) | (ws > cut_out)).astype(np.float32)
    is_full_power = ((ws >= rated) & (ws <= cut_out)).astype(np.float32)
    is_ramp_zone = ((ws >= cut_in) & (ws < rated)).astype(np.float32)

    ws_clipped = np.where(
        (ws < cut_in) | (ws > cut_out),
        0.0,
        np.where(ws >= rated, rated, ws),
    ).astype(np.float32)

    return is_zero_power, is_full_power, is_ramp_zone, ws_clipped


def process_ridge_features_raw(raw_data, horizon=24):
    """原始版特征：删除边界阶段与布尔开关相关特征。"""
    if raw_data.ndim != 3:
        raise ValueError(f"raw_data 应为 3 维 [T, N, F]，实际为 {raw_data.shape}")
    if raw_data.shape[-1] < 6:
        raise ValueError("raw_data 最后一维至少应包含 6 个字段: time,u10,v10,u100,v100,power")

    raw_data = raw_data.astype(object)
    t_steps, num_nodes, _ = raw_data.shape
    if t_steps < (3 + horizon):
        raise ValueError(f"时间步不足，至少需要 {3 + horizon}，当前 {t_steps}")

    u100 = raw_data[:, :, 3].astype(np.float32)
    v100 = raw_data[:, :, 4].astype(np.float32)
    power = raw_data[:, :, 5].astype(np.float32)

    ws = np.sqrt(u100 ** 2 + v100 ** 2).astype(np.float32)
    wd = np.arctan2(v100, u100).astype(np.float32)
    sin_wd = np.sin(wd).astype(np.float32)
    cos_wd = np.cos(wd).astype(np.float32)

    x_rows = []
    y_rows = []
    ws_rows = []
    farm_ids = []

    for t in range(2, t_steps - horizon):
        ws_t_all = ws[t]
        for n in range(num_nodes):
            hist_ws = ws[t - 2 : t + 1, n]
            hist_mean = float(hist_ws.mean())
            hist_max = float(hist_ws.max())
            first_diff = float(ws[t, n] - ws[t - 1, n])

            ws_curr = float(ws[t, n])
            ws_curr_cube = ws_curr ** 3
            sin_curr = float(sin_wd[t, n])
            cos_curr = float(cos_wd[t, n])

            others_ws = np.delete(ws_t_all, n).astype(np.float32)

            fut_ws = ws[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_ws_cube = (fut_ws ** 3).astype(np.float32)
            fut_sin = sin_wd[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_cos = cos_wd[t + 1 : t + horizon + 1, n].astype(np.float32)

            feat = np.concatenate(
                [
                    np.array(
                        [
                            hist_mean,
                            hist_max,
                            first_diff,
                            ws_curr,
                            ws_curr_cube,
                            sin_curr,
                            cos_curr,
                        ],
                        dtype=np.float32,
                    ),
                    others_ws,
                    fut_ws,
                    fut_ws_cube,
                    fut_sin,
                    fut_cos,
                ],
                axis=0,
            )

            target = power[t + 1 : t + horizon + 1, n].astype(np.float32)
            x_rows.append(feat)
            y_rows.append(target)
            ws_rows.append(fut_ws)
            farm_ids.append(n)

    x = np.stack(x_rows, axis=0).astype(np.float32)
    y = np.stack(y_rows, axis=0).astype(np.float32)
    ws_future = np.stack(ws_rows, axis=0).astype(np.float32)
    farm_ids = np.asarray(farm_ids, dtype=np.int32)
    return x, y, ws_future, farm_ids

def process_ridge_features_current(raw_data, horizon=24):
    """当前增强版特征：保留边界阶段与布尔开关相关特征。"""
    if raw_data.ndim != 3:
        raise ValueError(f"raw_data 应为 3 维 [T, N, F]，实际为 {raw_data.shape}")
    if raw_data.shape[-1] < 6:
        raise ValueError("raw_data 最后一维至少应包含 6 个字段: time,u10,v10,u100,v100,power")

    raw_data = raw_data.astype(object)
    t_steps, num_nodes, _ = raw_data.shape
    if t_steps < (3 + horizon):
        raise ValueError(f"时间步不足，至少需要 {3 + horizon}，当前 {t_steps}")

    u100 = raw_data[:, :, 3].astype(np.float32)
    v100 = raw_data[:, :, 4].astype(np.float32)
    power = raw_data[:, :, 5].astype(np.float32)

    ws = np.sqrt(u100 ** 2 + v100 ** 2).astype(np.float32)
    wd = np.arctan2(v100, u100).astype(np.float32)
    sin_wd = np.sin(wd).astype(np.float32)
    cos_wd = np.cos(wd).astype(np.float32)
    is_zero_power, is_full_power, is_ramp_zone, ws_clipped = add_physical_features(ws)

    x_rows = []
    y_rows = []
    ws_rows = []
    farm_ids = []

    for t in range(2, t_steps - horizon):
        ws_t_all = ws[t]
        for n in range(num_nodes):
            hist_ws = ws[t - 2 : t + 1, n]
            hist_mean = float(hist_ws.mean())
            hist_max = float(hist_ws.max())
            first_diff = float(ws[t, n] - ws[t - 1, n])

            ws_curr = float(ws[t, n])
            ws_curr_cube = ws_curr ** 3
            sin_curr = float(sin_wd[t, n])
            cos_curr = float(cos_wd[t, n])
            zero_curr = float(is_zero_power[t, n])
            full_curr = float(is_full_power[t, n])
            ramp_curr = float(is_ramp_zone[t, n])
            ws_clip_curr = float(ws_clipped[t, n])
            ws_clip_curr_cube = ws_clip_curr ** 3

            others_ws = np.delete(ws_t_all, n).astype(np.float32)

            fut_ws = ws[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_ws_cube = (fut_ws ** 3).astype(np.float32)
            fut_sin = sin_wd[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_cos = cos_wd[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_zero = is_zero_power[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_full = is_full_power[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_ramp = is_ramp_zone[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_ws_clip = ws_clipped[t + 1 : t + horizon + 1, n].astype(np.float32)
            fut_ws_clip_cube = (fut_ws_clip ** 3).astype(np.float32)

            feat = np.concatenate(
                [
                    np.array(
                        [
                            hist_mean,
                            hist_max,
                            first_diff,
                            ws_curr_cube,
                            sin_curr,
                            cos_curr,
                            zero_curr,
                            full_curr,
                            ramp_curr,
                            ws_clip_curr,
                            ws_clip_curr_cube,
                        ],
                        dtype=np.float32,
                    ),
                    others_ws,
                    fut_ws,
                    fut_ws_cube,
                    fut_sin,
                    fut_cos,
                    fut_zero,
                    fut_full,
                    fut_ramp,
                    fut_ws_clip,
                    fut_ws_clip_cube,
                ],
                axis=0,
            )

            target = power[t + 1 : t + horizon + 1, n].astype(np.float32)
            x_rows.append(feat)
            y_rows.append(target)
            ws_rows.append(fut_ws)
            farm_ids.append(n)

    x = np.stack(x_rows, axis=0).astype(np.float32)
    y = np.stack(y_rows, axis=0).astype(np.float32)
    ws_future = np.stack(ws_rows, axis=0).astype(np.float32)
    farm_ids = np.asarray(farm_ids, dtype=np.int32)
    return x, y, ws_future, farm_ids


def split_train_val(x, y, ws_future, farm_ids, ratio=0.8):
    split_idx = int(len(x) * ratio)
    if split_idx <= 5 or (len(x) - split_idx) <= 0:
        raise ValueError("训练/验证切分后样本不足")

    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    ws_val = ws_future[split_idx:]
    farm_val = farm_ids[split_idx:]
    return x_train, x_val, y_train, y_val, ws_val, farm_val


def train_raw_ridge(x_train, y_train):
    """最原始岭回归：固定 alpha=1.0，不做边界/布尔增强。"""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = Ridge(alpha=1.0)
    model.fit(x_train_scaled, y_train)
    return scaler, model, {"alpha": 1.0}


def train_current_ridge(x_train, y_train):
    """当前增强版：沿用 GridSearchCV 搜索 alpha。"""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    ridge = Ridge()
    param_grid = {"alpha": np.logspace(-3, 4, 50)}
    tscv = TimeSeriesSplit(n_splits=5)

    grid = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(x_train_scaled, y_train)
    return scaler, grid.best_estimator_, {"alpha": float(grid.best_params_["alpha"])}


def evaluate_and_plot(model_name, scaler, model, x_val, y_val, ws_val, farm_val, output_dir):
    x_val_scaled = scaler.transform(x_val)
    y_pred = model.predict(x_val_scaled).astype(np.float32)
    # 方案1：推理后做物理约束裁剪，避免负功率。
    y_pred = np.clip(y_pred, 0.0, None)

    global_mae = float(mean_absolute_error(y_val, y_pred))
    global_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

    per_farm_metrics = []
    scatter_dir = os.path.join(output_dir, f"scatter_{model_name}_by_farm")
    os.makedirs(scatter_dir, exist_ok=True)

    for farm_id in range(10):
        mask = farm_val == farm_id
        if not np.any(mask):
            print(f"[{model_name}] Farm {farm_id} 无验证样本，跳过绘图")
            continue

        ws_f = ws_val[mask].reshape(-1)
        y_true_f = y_val[mask].reshape(-1)
        y_pred_f = y_pred[mask].reshape(-1)

        farm_mae = float(mean_absolute_error(y_true_f, y_pred_f))
        farm_rmse = float(np.sqrt(mean_squared_error(y_true_f, y_pred_f)))
        per_farm_metrics.append((farm_id, farm_mae, farm_rmse))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(ws_f, y_true_f, s=8, alpha=0.35, label="True Power", color="tab:blue")
        ax.scatter(ws_f, y_pred_f, s=8, alpha=0.35, label="Pred Power", color="tab:orange")
        ax.set_title(f"{model_name} | Farm {farm_id} | MAE={farm_mae:.4f}")
        ax.set_xlabel("Wind Speed")
        ax.set_ylabel("Power")
        ax.grid(True, alpha=0.25)
        ax.legend()
        plt.tight_layout()

        scatter_path = os.path.join(scatter_dir, f"farm_{farm_id}.png")
        fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    metrics_path = os.path.join(output_dir, f"metrics_{model_name}.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Global Validation MAE: {global_mae:.6f}\n")
        f.write(f"Global Validation RMSE: {global_rmse:.6f}\n")
        f.write("\nPer-farm Validation Metrics:\n")
        for farm_id, farm_mae, farm_rmse in per_farm_metrics:
            f.write(f"Farm {farm_id}: MAE={farm_mae:.6f}, RMSE={farm_rmse:.6f}\n")

    print(f"[{model_name}] Global MAE={global_mae:.6f}, RMSE={global_rmse:.6f}")
    print(f"[{model_name}] 分风场散点图目录: {scatter_dir}")
    print(f"[{model_name}] 指标文件: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="岭回归原始版 vs 当前增强版 对比训练与散点图")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/wind_train_val_2012-01-02_to_2013-07-13.npy",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="saved_models/ridge_regression_compare",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"找不到数据文件: {args.data_path}")

    raw_data = np.load(args.data_path, allow_pickle=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print("===== 1) 原始版 Ridge（去掉边界/布尔开关）=====")
    x_raw, y_raw, ws_raw, farm_raw = process_ridge_features_raw(raw_data, horizon=args.horizon)
    x_train, x_val, y_train, y_val, ws_val, farm_val = split_train_val(x_raw, y_raw, ws_raw, farm_raw, ratio=0.8)
    scaler_raw, model_raw, params_raw = train_raw_ridge(x_train, y_train)
    joblib.dump(scaler_raw, os.path.join(save_dir, "raw_scaler.pkl"))
    joblib.dump(model_raw, os.path.join(save_dir, "raw_ridge_model.pkl"))
    with open(os.path.join(save_dir, "raw_params.txt"), "w", encoding="utf-8") as f:
        f.write(str(params_raw))
    evaluate_and_plot("raw", scaler_raw, model_raw, x_val, y_val, ws_val, farm_val, save_dir)

    print("===== 2) 当前增强版 Ridge（含边界/布尔开关）=====")
    x_cur, y_cur, ws_cur, farm_cur = process_ridge_features_current(raw_data, horizon=args.horizon)
    x_train, x_val, y_train, y_val, ws_val, farm_val = split_train_val(x_cur, y_cur, ws_cur, farm_cur, ratio=0.8)
    scaler_cur, model_cur, params_cur = train_current_ridge(x_train, y_train)
    joblib.dump(scaler_cur, os.path.join(save_dir, "current_scaler.pkl"))
    joblib.dump(model_cur, os.path.join(save_dir, "current_ridge_model.pkl"))
    with open(os.path.join(save_dir, "current_params.txt"), "w", encoding="utf-8") as f:
        f.write(str(params_cur))
    evaluate_and_plot("current", scaler_cur, model_cur, x_val, y_val, ws_val, farm_val, save_dir)

    print("=" * 50)
    print(f"✅ 全部完成，结果目录: {save_dir}")


if __name__ == "__main__":
    main()
