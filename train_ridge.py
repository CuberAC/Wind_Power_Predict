import argparse
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def add_physical_features(ws_array):
        """
        基于风机物理边界生成硬截断与布尔开关特征。

        返回:
            is_zero_power, is_full_power, is_ramp_zone, ws_clipped
        """
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


def process_ridge_features(raw_data, horizon=24):
    """
    构造用于岭回归的强特征。

    输入:
      raw_data: np.ndarray, [T, N, 6]
        idx 0: time (本函数不直接使用)
        idx 1: U10
        idx 2: V10
        idx 3: U100
        idx 4: V100
        idx 5: Power

    输出:
      X: [samples, num_features]
      Y: [samples, horizon]
    """
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

    # t 代表预测起点，预测 y[t+1 : t+horizon+1]
    for t in range(2, t_steps - horizon):
        ws_t_all = ws[t]  # [N]
        for n in range(num_nodes):
            hist_ws = ws[t - 2 : t + 1, n]  # 最近 3 小时: t-2,t-1,t
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

            # 其他 9 个风场当前风速
            others_ws = np.delete(ws_t_all, n).astype(np.float32)

            # 未来 24 小时预报特征（以未来气象作为可用外生变量）
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

    x = np.stack(x_rows, axis=0).astype(np.float32)
    y = np.stack(y_rows, axis=0).astype(np.float32)
    return x, y


def train_ridge(data_path, save_root):
    raw_data = np.load(data_path, allow_pickle=True)
    x, y = process_ridge_features(raw_data, horizon=24)

    split_idx = int(len(x) * 0.8)
    if split_idx <= 5:
        raise ValueError(f"训练样本过少，当前 split_idx={split_idx}")
    if (len(x) - split_idx) <= 0:
        raise ValueError("测试样本为空，请检查数据")

    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

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

    best_model = grid.best_estimator_
    print(f"Best alpha: {grid.best_params_['alpha']}")

    pred_test = best_model.predict(x_test_scaled)
    test_mae = mean_absolute_error(y_test, pred_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    scaler_path = os.path.join(save_dir, "scaler.pkl")
    model_path = os.path.join(save_dir, "best_ridge_model.pkl")
    joblib.dump(scaler, scaler_path)
    joblib.dump(best_model, model_path)

    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved model to: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ridge Regression for 24-step wind power forecasting")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/wind_train_val_2012-01-02_to_2013-07-13.npy",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="saved_models/ridge_regression",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_ridge(data_path=args.data_path, save_root=args.save_root)


if __name__ == "__main__":
    main()
