# ==============================================================================
# 脚本名称: test_lgb.py
# 任务目标: 评测 train_lgb.py 训练并保存的 LightGBM 模型
# 使用方法: python test_lgb.py
# ============================================================================== 

import argparse
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def extract_features_from_raw(data):
    """从原始 npy 数组中提取与 train_lgb.py 一致的特征。"""
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
    """在内存中构建与 train_lgb.py 一致的 X/Y。"""
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


def fill_nan_in_test(numeric_test):
    """与现有测试脚本口径一致，对 NaN 做时间维插值。"""
    if not np.isnan(numeric_test).any():
        return numeric_test

    print("⚠️ 警告：测试集中发现 NaN，正在进行插值处理...")
    for f in range(numeric_test.shape[1]):
        for c in range(numeric_test.shape[2]):
            series = pd.Series(numeric_test[:, f, c])
            numeric_test[:, f, c] = series.interpolate().ffill().bfill().values
    print("✅ 空值填充完毕。")
    return numeric_test


def evaluate_lgb_models(model_dir, x_all, y_all):
    """逐风场加载并评测 LightGBM 模型。"""
    num_farms = x_all.shape[1]
    results = []

    print(f"\n🚀 开始评测 LightGBM 模型目录: {model_dir}")
    for farm_id in range(num_farms):
        model_path = os.path.join(model_dir, f"lgbm_farm_{farm_id}.pkl")
        if not os.path.exists(model_path):
            print(f"⚠️ 找不到风场 {farm_id} 的模型: {model_path}，跳过")
            continue

        model = joblib.load(model_path)

        x_test = np.ascontiguousarray(x_all[:, farm_id, :], dtype=np.float32)
        y_true = np.asarray(y_all[:, farm_id, :], dtype=np.float32)

        print(f"   ⚙️ Farm {farm_id} - 推断矩阵维度: {x_test.shape[1]} 维")
        y_pred = model.predict(x_test)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        results.append((farm_id, rmse, mae))

        print(f"   ✅ Farm {farm_id} 测试完毕: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    return results


def save_report(results, model_dir, test_data_path):
    """保存评测报告。"""
    if not results:
        print("❌ 没有可用模型被评测，未生成报告。")
        return None

    avg_rmse = float(np.mean([r[1] for r in results]))
    avg_mae = float(np.mean([r[2] for r in results]))

    header = (
        "模型版本: lgbm_end2end\n"
        f"测试数据集: {test_data_path}\n"
        + "=" * 30
        + "\n"
    )
    body = "\n".join(
        [f"Farm {farm_id}: RMSE = {rmse:.4f}, MAE = {mae:.4f}" for farm_id, rmse, mae in results]
    )
    footer = (
        "\n"
        + "=" * 30
        + f"\n全局平均 RMSE: {avg_rmse:.4f}"
        + f"\n全局平均 MAE: {avg_mae:.4f}"
    )

    report_text = header + body + footer
    report_path = os.path.join(model_dir, "test_results_lgbm_end2end.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + "=" * 40)
    print(f"🎉 评测报告已生成: {report_path}")
    print(f"🏆 全局平均 RMSE: {avg_rmse:.4f}")
    print(f"🏆 全局平均 MAE: {avg_mae:.4f}")
    print("=" * 40)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="LightGBM 模型评测脚本")
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/wind_test_cleaned.npy",
        help="测试集 npy 文件路径（默认与 test_xgboost.py 对齐）",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="saved_models/lgbm_end2end",
        help="LightGBM 模型目录",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="滑动窗口长度，需与训练时一致",
    )
    args = parser.parse_args()

    # 与 train_lgb.py 一致，只屏蔽这条重复告警
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
        module="sklearn"
    )

    if not os.path.exists(args.test_data):
        print(f"❌ 找不到测试集文件: {args.test_data}")
        return

    if not os.path.exists(args.model_dir):
        print(f"❌ 找不到模型目录: {args.model_dir}")
        return

    print(f"[{time.strftime('%H:%M:%S')}] 📦 加载测试集: {args.test_data}")
    raw_test = np.load(args.test_data, allow_pickle=True)

    numeric_test = raw_test[:, :, 1:].astype(np.float32)
    numeric_test = fill_nan_in_test(numeric_test)

    weather_feat, power_data = feature_engineering_factory_compat(numeric_test, raw_test)
    x_all, y_all = build_sliding_window_in_memory(
        weather_feat, power_data, window_size=args.window_size
    )

    print(
        f"[{time.strftime('%H:%M:%S')}] ✅ 构建完成: 样本数={x_all.shape[0]} | "
        f"风场数={x_all.shape[1]} | 特征维度={x_all.shape[2]}"
    )

    results = evaluate_lgb_models(args.model_dir, x_all, y_all)
    save_report(results, args.model_dir, args.test_data)


# 为了兼容训练脚本的时间特征来源，优先从 raw_test 解析时间戳；
# 若解析失败则退化为与 xgb 测试脚本一致的纯序列小时特征。
def feature_engineering_factory_compat(numeric_test, raw_test):
    try:
        weather_feat, power_data = extract_features_from_raw(raw_test)
        return weather_feat, power_data
    except Exception:
        u10, v10 = numeric_test[:, :, 0], numeric_test[:, :, 1]
        u100, v100 = numeric_test[:, :, 2], numeric_test[:, :, 3]
        power_data = numeric_test[:, :, 4]

        ws10 = np.sqrt(u10 ** 2 + v10 ** 2)
        ws100 = np.sqrt(u100 ** 2 + v100 ** 2)
        ws100_cube = ws100 ** 3
        wdir_rad = np.arctan2(v100, u100)
        sin_wdir = np.sin(wdir_rad)
        cos_wdir = np.cos(wdir_rad)

        time_steps = numeric_test.shape[0]
        num_farms = numeric_test.shape[1]
        hours = np.array([h % 24 for h in range(time_steps)])
        sin_hour = np.repeat(np.sin(2 * np.pi * hours / 24)[:, np.newaxis], num_farms, axis=1)
        cos_hour = np.repeat(np.cos(2 * np.pi * hours / 24)[:, np.newaxis], num_farms, axis=1)
        sin_month = np.zeros_like(sin_hour)
        cos_month = np.zeros_like(cos_hour)

        weather_feat = np.stack(
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
                sin_hour,
                cos_hour,
                sin_month,
                cos_month,
            ],
            axis=-1,
        )
        return weather_feat, power_data


if __name__ == "__main__":
    main()
