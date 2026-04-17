# ==============================================================================
# 脚本名称: predict_all_xgboost_versions.py
# 任务目标: 用一个脚本统一完成 v1/v2/v2_best/v3/v3_lite/v3_1 的纯预测，
#           测试段固定为 wind_train_val 的后 20%，并保存各版本预测结果。
# 使用方法: python predict_all_xgboost_versions.py
# ==============================================================================

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def feature_engineering_factory(numeric_data, version):
    """
    根据版本号，对原始数值数据进行特征重构
    numeric_data: (Time, Farm, 5) -> [U10, V10, U100, V100, Power]
    """
    u10, v10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
    u100, v100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
    power = numeric_data[:, :, 4]

    ws10 = np.sqrt(u10 ** 2 + v10 ** 2)
    ws100 = np.sqrt(u100 ** 2 + v100 ** 2)

    if version == "v1":
        weather_features = np.stack([u10, v10, u100, v100, ws10, ws100], axis=-1)
        return weather_features, power

    ws100_cube = ws100 ** 3
    wdir_rad = np.arctan2(v100, u100)
    sin_wdir, cos_wdir = np.sin(wdir_rad), np.cos(wdir_rad)

    time_steps = numeric_data.shape[0]
    num_farms = numeric_data.shape[1]
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
    return weather_features, power


def build_predict_X(weather_features, power_data, version, window_size=24):
    """根据版本逻辑构建滑窗预测特征 X。"""
    num_time_steps = weather_features.shape[0]
    num_farms = weather_features.shape[1]

    x_list = [[] for _ in range(num_farms)]

    for i in range(window_size, num_time_steps - window_size + 1):
        if "v3" in version:
            past_power_global = power_data[i - window_size : i, :].flatten()

        for farm_id in range(num_farms):
            past_power_local = power_data[i - window_size : i, farm_id]
            p_weather = weather_features[i - window_size : i, farm_id, :].flatten()
            f_weather = weather_features[i : i + window_size, farm_id, :].flatten()

            if version == "v1":
                x_row = np.concatenate([past_power_local, p_weather, f_weather])
            elif "v3" in version:
                p_stats = np.array(
                    [
                        np.mean(past_power_local),
                        np.std(past_power_local),
                        np.max(past_power_local),
                        np.min(past_power_local),
                    ],
                    dtype=np.float32,
                )
                x_row = np.concatenate([past_power_global, p_stats, p_weather, f_weather])
            elif version in ("v2", "v2_best"):
                p_stats = np.array(
                    [
                        np.mean(past_power_local),
                        np.std(past_power_local),
                        np.max(past_power_local),
                        np.min(past_power_local),
                    ],
                    dtype=np.float32,
                )
                x_row = np.concatenate([past_power_local, p_stats, p_weather, f_weather])
            else:
                raise ValueError(f"不支持的版本: {version}")

            x_list[farm_id].append(x_row)

    return [np.asarray(x, dtype=np.float32) for x in x_list]


def build_true_Y_and_future_ws(power_data, ws100, window_size=24):
    """构建真实 Y 和对应未来时段风速（用于可视化）。"""
    num_time_steps = power_data.shape[0]
    num_farms = power_data.shape[1]

    y_list = [[] for _ in range(num_farms)]
    ws_list = [[] for _ in range(num_farms)]

    for i in range(window_size, num_time_steps - window_size + 1):
        for farm_id in range(num_farms):
            y_row = power_data[i : i + window_size, farm_id]
            ws_row = ws100[i : i + window_size, farm_id]
            y_list[farm_id].append(y_row)
            ws_list[farm_id].append(ws_row)

    y_farms = [np.asarray(y, dtype=np.float32) for y in y_list]
    ws_farms = [np.asarray(w, dtype=np.float32) for w in ws_list]
    return y_farms, ws_farms


def plot_version_visualizations(
    version,
    preds,
    y_true_farms,
    ws_future_farms,
    timestamps,
    output_root,
    window_size,
):
    """为每个版本绘制两类图：风速-出力散点图、首周时间序列图。"""
    version_dir = os.path.join(output_root, f"plots_{version}")
    power_curve_dir = os.path.join(version_dir, "power_curve_by_farm")
    time_series_dir = os.path.join(version_dir, "time_series_week1_by_farm")
    os.makedirs(power_curve_dir, exist_ok=True)
    os.makedirs(time_series_dir, exist_ok=True)

    num_samples = preds.shape[1]
    time_axis = timestamps[window_size : window_size + num_samples]
    week_points = min(24 * 7, num_samples)

    for farm_id in range(preds.shape[0]):
        y_true = y_true_farms[farm_id]
        ws_future = ws_future_farms[farm_id]
        y_pred = preds[farm_id]

        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        ws_flat = ws_future.reshape(-1)

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(ws_flat, y_true_flat, s=8, alpha=0.35, label="True Power", color="tab:blue")
        ax1.scatter(ws_flat, y_pred_flat, s=8, alpha=0.35, label="Pred Power", color="tab:orange")
        ax1.set_title(f"{version} | Farm {farm_id} | Wind Speed vs Power")
        ax1.set_xlabel("Wind Speed (100m)")
        ax1.set_ylabel("Power")
        ax1.grid(True, alpha=0.25)
        ax1.legend()
        plt.tight_layout()
        fig1.savefig(os.path.join(power_curve_dir, f"farm_{farm_id}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig1)

        y_true_1step = y_true[:, 0]
        y_pred_1step = y_pred[:, 0]

        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(time_axis[:week_points], y_true_1step[:week_points], label="True Power", color="tab:blue")
        ax2.plot(time_axis[:week_points], y_pred_1step[:week_points], label="Pred Power", color="tab:orange")
        ax2.set_title(f"{version} | Farm {farm_id} | First Week Power Time Series")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Power")
        ax2.grid(True, alpha=0.25)
        ax2.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        fig2.savefig(os.path.join(time_series_dir, f"farm_{farm_id}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig2)

    return power_curve_dir, time_series_dir


def fill_nan_numeric(numeric_data):
    if not np.isnan(numeric_data).any():
        return numeric_data

    print("⚠️ 检测到 NaN，开始插值处理...")
    for farm_id in range(numeric_data.shape[1]):
        for col_id in range(numeric_data.shape[2]):
            series = pd.Series(numeric_data[:, farm_id, col_id])
            numeric_data[:, farm_id, col_id] = series.interpolate().ffill().bfill().values
    print("✅ NaN 处理完成。")
    return numeric_data


def load_optional_indices(version):
    top_indices = None
    routing_dict = None

    if version == "v3_lite":
        index_file = "data/top_indices_v3.npy"
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"找不到 v3_lite 索引文件: {index_file}")
        top_indices = np.load(index_file)

    if version == "v3_1":
        dict_file = "data/v3_1_routing_dict.npy"
        if not os.path.exists(dict_file):
            raise FileNotFoundError(f"找不到 v3_1 路由字典: {dict_file}")
        routing_dict = np.load(dict_file)

    return top_indices, routing_dict


def predict_single_version(version, numeric_test, timestamps, model_root, output_root, window_size=24):
    model_dir = os.path.join(model_root, version)
    if not os.path.exists(model_dir):
        print(f"⚠️ 版本 [{version}] 模型目录不存在，跳过: {model_dir}")
        return None

    top_indices, routing_dict = load_optional_indices(version)
    weather_feat, power_data = feature_engineering_factory(numeric_test, version)
    x_farms = build_predict_X(weather_feat, power_data, version, window_size=window_size)
    ws100 = np.sqrt(numeric_test[:, :, 2] ** 2 + numeric_test[:, :, 3] ** 2).astype(np.float32)
    y_true_farms, ws_future_farms = build_true_Y_and_future_ws(power_data, ws100, window_size=window_size)

    version_preds = []
    available_farms = []

    print(f"\n🚀 开始版本 [{version}] 纯预测...")
    for farm_id in range(10):
        model_path = os.path.join(model_dir, f"xgb_{version}_farm_{farm_id}.pkl")
        if not os.path.exists(model_path):
            print(f"   ⚠️ 缺少模型，跳过 Farm {farm_id}: {model_path}")
            continue

        model = joblib.load(model_path)
        x_test = x_farms[farm_id]

        if routing_dict is not None:
            x_test = x_test[:, routing_dict[farm_id]]
        elif top_indices is not None:
            x_test = x_test[:, top_indices]

        x_test = np.ascontiguousarray(x_test, dtype=np.float32)
        y_pred = model.predict(x_test)
        y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)

        version_preds.append(y_pred)
        available_farms.append(farm_id)
        print(f"   ✅ Farm {farm_id} 完成，输出形状: {y_pred.shape}")

    if not version_preds:
        print(f"❌ 版本 [{version}] 没有可用模型，未保存结果。")
        return None

    preds = np.stack(version_preds, axis=0)

    power_curve_dir, time_series_dir = plot_version_visualizations(
        version=version,
        preds=preds,
        y_true_farms=y_true_farms,
        ws_future_farms=ws_future_farms,
        timestamps=timestamps,
        output_root=output_root,
        window_size=window_size,
    )

    os.makedirs(output_root, exist_ok=True)
    pred_path = os.path.join(output_root, f"predictions_{version}_last20.npy")
    np.save(pred_path, preds)

    summary_path = os.path.join(output_root, f"predictions_{version}_last20_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"version: {version}\n")
        f.write("test_source: data/wind_val_20.npy\n")
        f.write(f"prediction_shape: {preds.shape}  # (Farm, Sample, Horizon)\n")
        f.write(f"available_farms: {available_farms}\n")
        f.write(f"window_size: {window_size}\n")
        f.write(f"power_curve_plot_dir: {power_curve_dir}\n")
        f.write(f"week1_timeseries_plot_dir: {time_series_dir}\n")

    print(f"🎉 版本 [{version}] 预测已保存: {pred_path}")
    return {
        "version": version,
        "pred_path": pred_path,
        "summary_path": summary_path,
        "shape": preds.shape,
        "power_curve_dir": power_curve_dir,
        "time_series_dir": time_series_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="XGBoost 多版本统一纯预测脚本")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/wind_val_20.npy",
        help="测试集路径（默认 data/wind_val_20.npy）",
    )
    parser.add_argument(
        "--versions",
        type=str,
        nargs="+",
        default=["v1", "v2", "v2_best", "v3", "v3_lite", "v3_1"],
        help="需要预测的版本列表",
    )
    parser.add_argument("--model-root", type=str, default="saved_models", help="模型根目录")
    parser.add_argument("--output-root", type=str, default="saved_models/predictions", help="预测输出目录")
    parser.add_argument("--window-size", type=int, default=24, help="滑窗长度")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"❌ 找不到数据文件: {args.data_path}")
        return

    print(f"📦 加载数据: {args.data_path}")
    test_raw = np.load(args.data_path, allow_pickle=True)
    numeric_test = test_raw[:, :, 1:].astype(np.float32)
    numeric_test = fill_nan_numeric(numeric_test)
    timestamps = pd.to_datetime(test_raw[:, 0, 0])

    print(
        f"🧪 测试集加载完成: 数据长度={numeric_test.shape[0]} | 风场数={numeric_test.shape[1]}"
    )

    all_results = []
    for version in args.versions:
        try:
            result = predict_single_version(
                version=version,
                numeric_test=numeric_test,
                timestamps=timestamps,
                model_root=args.model_root,
                output_root=args.output_root,
                window_size=args.window_size,
            )
            if result is not None:
                all_results.append(result)
        except Exception as exc:
            print(f"❌ 版本 [{version}] 预测失败: {exc}")

    if not all_results:
        print("❌ 没有任何版本成功完成预测。")
        return

    merged_summary = os.path.join(args.output_root, "all_versions_prediction_summary.txt")
    with open(merged_summary, "w", encoding="utf-8") as f:
        f.write("=== All Versions Prediction Summary ===\n")
        f.write("test_source: data/wind_val_20.npy\n\n")
        for item in all_results:
            f.write(f"version: {item['version']}\n")
            f.write(f"prediction_file: {item['pred_path']}\n")
            f.write(f"summary_file: {item['summary_path']}\n")
            f.write(f"shape: {item['shape']}\n")
            f.write(f"power_curve_plot_dir: {item['power_curve_dir']}\n")
            f.write(f"week1_timeseries_plot_dir: {item['time_series_dir']}\n")
            f.write("-" * 40 + "\n")

    print("\n" + "=" * 50)
    print("🎉 多版本预测任务完成")
    for item in all_results:
        print(f"- {item['version']}: {item['pred_path']} | shape={item['shape']}")
        print(f"  风速-出力散点图: {item['power_curve_dir']}")
        print(f"  首周时间序列图: {item['time_series_dir']}")
    print(f"📄 总汇总文件: {merged_summary}")
    print("=" * 50)


if __name__ == "__main__":
    main()