# ==============================================================================
# 脚本名称: predict_v3_1.py
# 任务目标: 针对 v3_1 模型的独立调用脚本，加载 10 个风场的模型和路由字典，
#           对输入 npy 文件进行预测，并可选保存预测结果与评测报告。
# 使用方法: python predict_v3_1.py --test-data data/wind_test_cleaned.npy
# ==============================================================================

import argparse
import os

import joblib
import numpy as np
import pandas as pd


def fill_nan_in_test(numeric_test):
    """对测试集做时间维插值，保持与训练/评测脚本一致的清洗口径。"""
    if not np.isnan(numeric_test).any():
        return numeric_test

    print("⚠️ 检测到 NaN，开始插值处理...")
    for farm_id in range(numeric_test.shape[1]):
        for col_id in range(numeric_test.shape[2]):
            series = pd.Series(numeric_test[:, farm_id, col_id])
            numeric_test[:, farm_id, col_id] = series.interpolate().ffill().bfill().values
    print("✅ NaN 处理完成。")
    return numeric_test


def build_v3_1_features(numeric_data, window_size=24):
    """把原始 (Time, Farm, 5) 数值数据重建为 v3_1 预测所需特征。"""
    u10, v10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
    u100, v100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
    power_data = numeric_data[:, :, 4]

    ws10 = np.sqrt(u10 ** 2 + v10 ** 2)
    ws100 = np.sqrt(u100 ** 2 + v100 ** 2)
    ws100_cube = ws100 ** 3
    wind_dir_rad = np.arctan2(v100, u100)
    sin_wdir, cos_wdir = np.sin(wind_dir_rad), np.cos(wind_dir_rad)

    time_steps, num_farms = numeric_data.shape[0], numeric_data.shape[1]
    hours = np.array([h % 24 for h in range(time_steps)], dtype=np.float32)
    sin_hour = np.repeat(np.sin(2 * np.pi * hours / 24)[:, np.newaxis], num_farms, axis=1)
    cos_hour = np.repeat(np.cos(2 * np.pi * hours / 24)[:, np.newaxis], num_farms, axis=1)
    sin_month = np.zeros_like(sin_hour)
    cos_month = np.zeros_like(cos_hour)

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
            sin_hour,
            cos_hour,
            sin_month,
            cos_month,
        ],
        axis=-1,
    )

    x_list = [[] for _ in range(num_farms)]
    for i in range(window_size, time_steps - window_size + 1):
        past_power_global = power_data[i - window_size : i, :].flatten()

        for farm_id in range(num_farms):
            past_power_local = power_data[i - window_size : i, farm_id]
            past_weather = weather_features[i - window_size : i, farm_id, :].flatten()
            future_weather = weather_features[i : i + window_size, farm_id, :].flatten()
            power_stats = np.array(
                [
                    np.mean(past_power_local),
                    np.std(past_power_local),
                    np.max(past_power_local),
                    np.min(past_power_local),
                ],
                dtype=np.float32,
            )

            x_row = np.concatenate([past_power_global, power_stats, past_weather, future_weather])

            x_list[farm_id].append(x_row)

    x_farms = [np.asarray(x, dtype=np.float32) for x in x_list]
    return x_farms


def load_inputs(test_data_path):
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"找不到测试集文件: {test_data_path}")

    raw_test = np.load(test_data_path, allow_pickle=True)
    numeric_test = raw_test[:, :, 1:].astype(np.float32)
    numeric_test = fill_nan_in_test(numeric_test)
    return numeric_test


def main():
    parser = argparse.ArgumentParser(description="v3_1 风电功率预测调用脚本")
    parser.add_argument("--test-data", type=str, default="data/wind_test_cleaned.npy", help="输入 npy 文件")
    parser.add_argument("--model-dir", type=str, default="saved_models/v3_1", help="v3_1 模型目录")
    parser.add_argument("--routing-dict", type=str, default="data/v3_1_routing_dict.npy", help="路由字典文件")
    parser.add_argument("--output", type=str, default="saved_models/v3_1/v3_1_predictions.npy", help="预测结果保存路径")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"❌ 找不到模型目录: {args.model_dir}")
        return

    if not os.path.exists(args.routing_dict):
        print(f"❌ 找不到路由字典: {args.routing_dict}")
        return

    print(f"📦 加载测试集: {args.test_data}")
    numeric_test = load_inputs(args.test_data)

    print("🧩 重建 v3_1 特征并构造滑窗样本...")
    x_farms = build_v3_1_features(numeric_test)
    routing_dict = np.load(args.routing_dict)

    all_predictions = []

    print("🚀 开始逐风场预测...")
    for farm_id in range(10):
        model_path = os.path.join(args.model_dir, f"xgb_v3_1_farm_{farm_id}.pkl")
        if not os.path.exists(model_path):
            print(f"⚠️ 找不到风场 {farm_id} 的模型: {model_path}，跳过")
            continue

        model = joblib.load(model_path)
        x_test = x_farms[farm_id][:, routing_dict[farm_id]]
        x_test = np.ascontiguousarray(x_test, dtype=np.float32)

        print(f"   ⚙️ Farm {farm_id} 输入维度: {x_test.shape[1]}")
        y_pred = model.predict(x_test)
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)

        all_predictions.append(y_pred)
        print(f"   ✅ Farm {farm_id} 完成: 预测输出形状 = {y_pred.shape}")

    if not all_predictions:
        print("❌ 没有成功加载任何模型，未生成结果。")
        return

    predictions = np.stack(all_predictions, axis=0)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(args.output, predictions)

    print("\n" + "=" * 40)
    print(f"🎉 预测结果已保存: {args.output}")
    print(f"📐 输出数组形状: {predictions.shape} (Farm, Sample, Horizon)")
    print("=" * 40)


if __name__ == "__main__":
    main()