import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor


# =========================
# Feature layout constants
# =========================
WINDOW_SIZE = 24
NUM_FARMS = 10

NUM_PAST_POWER = WINDOW_SIZE
NUM_PAST_STATS = 4
NUM_WEATHER_FEATURES = 13
NUM_WEATHER_BLOCK = WINDOW_SIZE * NUM_WEATHER_FEATURES

PAST_WEATHER_START = NUM_PAST_POWER + NUM_PAST_STATS
PAST_WEATHER_END = PAST_WEATHER_START + NUM_WEATHER_BLOCK
FUTURE_WEATHER_START = PAST_WEATHER_END
FUTURE_WEATHER_END = FUTURE_WEATHER_START + NUM_WEATHER_BLOCK
FARM_ID_COL = FUTURE_WEATHER_END


def extract_features_from_raw(data):
    """从原始 npy 数组中提取与 train_lgb.py 完全一致的特征。"""
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

    ws10 = np.sqrt(u10**2 + v10**2)
    ws100 = np.sqrt(u100**2 + v100**2)
    ws100_cube = ws100**3
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


def build_sliding_window(weather_features, power_data, window_size=24):
    """在内存中构建滑动窗口监督学习矩阵 X 和 Y。"""
    num_time_steps = weather_features.shape[0]
    num_farms = weather_features.shape[1]
    num_samples = num_time_steps - window_size * 2 + 1

    feature_dim = window_size + 4 + (window_size * 13) + (window_size * 13)

    X_all = np.zeros((num_samples, num_farms, feature_dim), dtype=np.float32)
    Y_all = np.zeros((num_samples, num_farms, window_size), dtype=np.float32)

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

            X_all[sample_idx, farm_id, :] = np.concatenate(
                [past_power, past_power_stats, past_weather, future_weather]
            )
            Y_all[sample_idx, farm_id, :] = power_data[i : i + window_size, farm_id]

        sample_idx += 1

    return X_all, Y_all


def flatten_and_add_farm_id(X_3d, Y_3d):
    """把 (N, 10, F) / (N, 10, 24) 展平为二维，并给 X 追加 Farm_ID 列。"""
    if X_3d.ndim != 3 or Y_3d.ndim != 3:
        raise ValueError("X_3d 和 Y_3d 必须是三维数组。")

    num_steps, num_farms, feature_dim = X_3d.shape
    if Y_3d.shape[0] != num_steps or Y_3d.shape[1] != num_farms:
        raise ValueError("X_3d 和 Y_3d 在时间维或风场维不一致。")

    X_2d = X_3d.reshape(num_steps * num_farms, feature_dim)
    Y_2d = Y_3d.reshape(num_steps * num_farms, Y_3d.shape[2])

    farm_ids = np.tile(np.arange(num_farms), num_steps).reshape(-1, 1)
    X_2d_with_farm = np.concatenate([X_2d, farm_ids.astype(np.float32)], axis=1)

    return X_2d_with_farm.astype(np.float32), Y_2d.astype(np.float32)


def build_meta_features(pred_xgb, pred_lgb, raw_x):
    """
    全量特征拼接法：[XGB预测24维, LGB预测24维, 原始全量特征(含过去未来气象), Farm_ID 1维]
    """
    meta_x = np.concatenate([pred_xgb, pred_lgb, raw_x], axis=1)
    return meta_x.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="OOF + Stacking for Wind Power Forecasting")
    parser.add_argument(
        "--raw-data",
        type=str,
        default="data/wind_train_val_2012-01-02_to_2013-07-13.npy",
        help="原始数据路径",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="滑动窗口长度",
    )
    args = parser.parse_args()

    # 与你现有 LGB 训练脚本一致，避免刷屏警告
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
        module="sklearn",
    )

    data = np.load(args.raw_data, allow_pickle=True)

    weather_features, power_data = extract_features_from_raw(data)
    X_all, Y_all = build_sliding_window(
        weather_features,
        power_data,
        window_size=args.window_size,
    )

    if X_all is None or Y_all is None:
        raise ValueError("请在 extract_features_from_raw / build_sliding_window 中返回有效数组。")

    num_steps = X_all.shape[0]
    split_idx = int(num_steps * 0.8)

    # Step 2) 初始时序切分
    X_train_3d = X_all[:split_idx, :, :]
    Y_train_3d = Y_all[:split_idx, :, :]
    X_test_3d = X_all[split_idx:, :, :]
    Y_test_3d = Y_all[split_idx:, :, :]

    # Step 3) Layer 1 OOF
    # 第一层参数不是固定值，是可调的；这里给的是可直接替换的默认起点。
    xgb_params = {
        "n_estimators": 598, 
        "max_depth": 7, 
        "learning_rate": 0.013698166894754917,
        "subsample": 0.7186255612121745, 
        "colsample_bytree": 0.9993604825235307,
        "min_child_weight": 2,
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }
    lgb_params = {
        "n_estimators": 600,
        "learning_rate": 0.013682416119938175,
        "num_leaves": 41,
        "max_depth": 11,
        "min_child_samples": 56,
        "subsample": 0.6404736081419506,
        "colsample_bytree": 0.743050136518371,
        "objective": "regression",
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": -1,
    }

    tscv = TimeSeriesSplit(n_splits=3)

    oof_pred_xgb_list = []
    oof_pred_lgb_list = []
    oof_raw_features_list = []
    oof_y_list = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_3d), start=1):
        X_fold_train_3d = X_train_3d[train_idx, :, :]
        Y_fold_train_3d = Y_train_3d[train_idx, :, :]
        X_fold_val_3d = X_train_3d[val_idx, :, :]
        Y_fold_val_3d = Y_train_3d[val_idx, :, :]

        X_fold_train_2d, Y_fold_train_2d = flatten_and_add_farm_id(
            X_fold_train_3d,
            Y_fold_train_3d,
        )
        X_fold_val_2d, Y_fold_val_2d = flatten_and_add_farm_id(
            X_fold_val_3d,
            Y_fold_val_3d,
        )

        model_xgb = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
        model_lgb = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_params))

        model_xgb.fit(X_fold_train_2d, Y_fold_train_2d)
        model_lgb.fit(X_fold_train_2d, Y_fold_train_2d)

        val_pred_xgb = model_xgb.predict(X_fold_val_2d)
        val_pred_lgb = model_lgb.predict(X_fold_val_2d)

        oof_pred_xgb_list.append(val_pred_xgb)
        oof_pred_lgb_list.append(val_pred_lgb)
        oof_raw_features_list.append(X_fold_val_2d)
        oof_y_list.append(Y_fold_val_2d)

        fold_mae_xgb = mean_absolute_error(Y_fold_val_2d, val_pred_xgb)
        fold_mae_lgb = mean_absolute_error(Y_fold_val_2d, val_pred_lgb)
        print(f"[Fold {fold}] XGB MAE={fold_mae_xgb:.6f} | LGB MAE={fold_mae_lgb:.6f}")

    # Step 4) 构造 Layer 2 元特征
    OOF_Pred_XGB = np.concatenate(oof_pred_xgb_list, axis=0)
    OOF_Pred_LGB = np.concatenate(oof_pred_lgb_list, axis=0)
    OOF_Raw_X = np.concatenate(oof_raw_features_list, axis=0)
    OOF_Y = np.concatenate(oof_y_list, axis=0)

    Meta_X_train = build_meta_features(OOF_Pred_XGB, OOF_Pred_LGB, OOF_Raw_X)

    # Step 5) 训练元模型
    meta_model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            max_depth=3,
            num_leaves=15,
            n_estimators=150,
            random_state=42,
            verbosity=-1,
        )
    )
    farm_id_col_idx = Meta_X_train.shape[1] - 1
    meta_model.fit(
        Meta_X_train, OOF_Y,
        **{'categorical_feature': [farm_id_col_idx]}
    )

    # Step 6) 测试集推理
    X_train_full_2d, Y_train_full_2d = flatten_and_add_farm_id(X_train_3d, Y_train_3d)
    X_test_2d, Y_test_2d = flatten_and_add_farm_id(X_test_3d, Y_test_3d)

    model_xgb_full = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
    model_lgb_full = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_params))

    model_xgb_full.fit(X_train_full_2d, Y_train_full_2d)
    model_lgb_full.fit(X_train_full_2d, Y_train_full_2d)

    Test_Pred_XGB = model_xgb_full.predict(X_test_2d)
    Test_Pred_LGB = model_lgb_full.predict(X_test_2d)

    Meta_X_test = build_meta_features(Test_Pred_XGB, Test_Pred_LGB, X_test_2d)

    Final_Predictions = meta_model.predict(Meta_X_test)

    stack_rmse = np.sqrt(mean_squared_error(Y_test_2d, Final_Predictions))
    stack_mae = mean_absolute_error(Y_test_2d, Final_Predictions)

    xgb_mae = mean_absolute_error(Y_test_2d, Test_Pred_XGB)
    lgb_mae = mean_absolute_error(Y_test_2d, Test_Pred_LGB)

    print("=" * 60)
    print("Final Test Metrics")
    print(f"Stacking RMSE: {stack_rmse:.6f}")
    print(f"Stacking MAE : {stack_mae:.6f}")
    print(f"XGB MAE      : {xgb_mae:.6f}")
    print(f"LGB MAE      : {lgb_mae:.6f}")
    print("=" * 60)
    
    # ==========================================
    # Step 7) 保存 Stacking 模型全家桶 (用于未来线上预测)
    # ==========================================
    save_dir = "saved_models/stacking_final"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"💾 正在保存 Stacking 全家桶至 {save_dir}/ ...")
    # 1. 保存第一层的基础模型 (注意：保存的是用全量训练集 fit 出来的 full 模型)
    joblib.dump(model_xgb_full, os.path.join(save_dir, "layer1_xgb_full.pkl"))
    joblib.dump(model_lgb_full, os.path.join(save_dir, "layer1_lgb_full.pkl"))
    
    # 2. 保存第二层的元模型
    joblib.dump(meta_model, os.path.join(save_dir, "layer2_meta_model.pkl"))
    
    print("🎉 所有模型保存完毕！可以用于最终的隐藏数据集测试了！")

if __name__ == "__main__":
    main()
