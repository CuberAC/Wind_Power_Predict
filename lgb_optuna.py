import os
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


# ==========================================
# 1) 特征工程函数（你可按需替换内部逻辑）
# ==========================================
def extract_features_from_raw(data):
    """输入原始三维数据，返回 weather_features 和 Power。"""
    timestamps = pd.to_datetime(data[:, 0, 0])
    hours = timestamps.hour.values
    months = timestamps.month.values

    sin_hour = np.sin(2 * np.pi * hours / 24)
    cos_hour = np.cos(2 * np.pi * hours / 24)
    sin_month = np.sin(2 * np.pi * months / 12)
    cos_month = np.cos(2 * np.pi * months / 12)

    num_farms = data.shape[1]
    sin_hour_full = np.repeat(sin_hour[:, np.newaxis], num_farms, axis=1)
    cos_hour_full = np.repeat(cos_hour[:, np.newaxis], num_farms, axis=1)
    sin_month_full = np.repeat(sin_month[:, np.newaxis], num_farms, axis=1)
    cos_month_full = np.repeat(cos_month[:, np.newaxis], num_farms, axis=1)

    numeric_data = data[:, :, 1:].astype(np.float32)
    u10, v10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
    u100, v100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
    power = numeric_data[:, :, 4]

    ws10 = np.sqrt(u10 ** 2 + v10 ** 2)
    ws100 = np.sqrt(u100 ** 2 + v100 ** 2)
    ws100_cube = ws100 ** 3
    wind_dir_rad = np.arctan2(v100, u100)
    sin_wdir = np.sin(wind_dir_rad)
    cos_wdir = np.cos(wind_dir_rad)

    weather_features = np.stack(
        [
            u10,
            v10,
            u100,
            v100,
            ws10,
            ws100,
            ws100_cube,
            sin_wdir,
            cos_wdir,
            sin_hour_full,
            cos_hour_full,
            sin_month_full,
            cos_month_full,
        ],
        axis=-1,
    )

    return weather_features, power


def build_sliding_window_in_memory(weather_features, power_data, window_size=24):
    """返回滑窗构造后的 X_all/Y_all。

    X_all: (num_samples, num_farms, feature_dim)
    Y_all: (num_samples, num_farms, window_size)
    """
    num_time_steps = weather_features.shape[0]
    num_farms = weather_features.shape[1]
    num_samples = num_time_steps - window_size * 2 + 1

    feature_dim = window_size + 4 + (window_size * 13) + (window_size * 13)

    x_all = np.zeros((num_samples, num_farms, feature_dim), dtype=np.float32)
    y_all = np.zeros((num_samples, num_farms, window_size), dtype=np.float32)

    sample_idx = 0
    for i in range(window_size, num_time_steps - window_size + 1):
        for farm_id in range(num_farms):
            past_power = power_data[i - window_size : i, farm_id]
            past_power_stats = np.array(
                [
                    np.mean(past_power),
                    np.std(past_power),
                    np.max(past_power),
                    np.min(past_power),
                ],
                dtype=np.float32,
            )
            past_weather = weather_features[i - window_size : i, farm_id, :].flatten()
            future_weather = weather_features[i : i + window_size, farm_id, :].flatten()

            x_all[sample_idx, farm_id, :] = np.concatenate(
                [past_power, past_power_stats, past_weather, future_weather]
            )
            y_all[sample_idx, farm_id, :] = power_data[i : i + window_size, farm_id]

        sample_idx += 1

    return x_all, y_all


# ==========================================
# 2) Optuna 目标函数
# ==========================================
def objective(trial, X_train, Y_train, X_val, Y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "objective": "regression",
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": -1,
    }

    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))

    # 统一输入为 numpy，避免 DataFrame/ndarray 混用触发 feature names 警告
    X_train_np = np.asarray(X_train, dtype=np.float32)
    Y_train_np = np.asarray(Y_train, dtype=np.float32)
    X_val_np = np.asarray(X_val, dtype=np.float32)
    Y_val_np = np.asarray(Y_val, dtype=np.float32)

    model.fit(X_train_np, Y_train_np)
    y_pred = model.predict(X_val_np)

    rmse = float(np.sqrt(mean_squared_error(Y_val_np, y_pred)))
    return rmse


# ==========================================
# 3) 主流程
# ==========================================
def main():
    # 与 train_lgb.py 一致，只屏蔽这条重复告警
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
        module="sklearn",
    )

    data_path = "data/wind_train_val_2012-01-02_to_2013-07-13.npy"
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件: {data_path}")
        return

    print(f"[{time.strftime('%H:%M:%S')}] 📦 加载训练数据: {data_path}")
    raw_data = np.load(data_path, allow_pickle=True)


    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ 执行特征工程与滑窗构造 (仅用 farm0)...")
    weather_features, power_data = extract_features_from_raw(raw_data)
    X_all, Y_all = build_sliding_window_in_memory(weather_features, power_data, window_size=24)

    # 只用 farm0 的数据
    X_farm0 = X_all[:, 0, :]
    Y_farm0 = Y_all[:, 0, :]
    num_samples, feature_dim = X_farm0.shape
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 构建完成: samples={num_samples}, feature_dim={feature_dim}")

    # 1) 先按时间序列切分（80/20）
    split_idx = int(num_samples * 0.8)
    X_train_2d = X_farm0[:split_idx, :]
    Y_train_2d = Y_farm0[:split_idx, :]
    X_val_2d = X_farm0[split_idx:, :]
    Y_val_2d = Y_farm0[split_idx:, :]

    # 再次强制为 numpy float32，进一步避免潜在输入格式问题
    X_train_2d = np.asarray(X_train_2d, dtype=np.float32)
    Y_train_2d = np.asarray(Y_train_2d, dtype=np.float32)
    X_val_2d = np.asarray(X_val_2d, dtype=np.float32)
    Y_val_2d = np.asarray(Y_val_2d, dtype=np.float32)

    print(
        f"[{time.strftime('%H:%M:%S')}] 🔀 farm0 形状: "
        f"X_train={X_train_2d.shape}, Y_train={Y_train_2d.shape}, "
        f"X_val={X_val_2d.shape}, Y_val={Y_val_2d.shape}"
    )

    print(f"[{time.strftime('%H:%M:%S')}] 🚀 开始 Optuna 搜索 (n_trials=50)...")
    start_time = time.time()

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train_2d, Y_train_2d, X_val_2d, Y_val_2d),
        n_trials=50,
        show_progress_bar=False,
    )

    elapsed = time.time() - start_time
    best_rmse = float(study.best_value)
    best_params = study.best_params

    print("\n" + "=" * 50)
    print(f"🎯 最优 RMSE: {best_rmse:.6f}")
    print("🎯 最优参数:")
    print(best_params)
    print(f"⏱️ 搜索耗时: {elapsed:.1f} 秒")
    print("=" * 50)

    save_dir = "saved_models/lgbm_end2end"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "best_lgb_optuna_params.txt")

    lines = [
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "study: LightGBM + Optuna (multi-output, global mixed training)",
        f"best_rmse: {best_rmse:.8f}",
        "best_params:",
    ]
    for k, v in best_params.items():
        lines.append(f"  {k}: {v}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"💾 最优参数已保存到: {out_path}")


if __name__ == "__main__":
    main()
