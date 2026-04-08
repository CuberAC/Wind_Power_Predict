# ==============================================================================
# 脚本名称: feature_engineering_v3.py
# 任务目标: 终极空间融合特征工程 (V3)
# 核心创新: 在保留 V2 所有物理与时序特征的基础上，引入全局空间相关性。
# 数据输出: features_v3.npz
# ==============================================================================

import numpy as np
import pandas as pd
import os
import time


def load_and_extract_advanced_features(file_path):
    """
    第一步：加载数据并构建深度的气象与时间特征 (同 V2 完全一致)
    1. 加载 file_path 的 .npy 数据
    2. 提取时间戳，计算小时(Hour)和月份(Month)的 Sin/Cos 周期特征
    3. 将周期特征扩展为 (时间步, 风电场数) 的形状
    4. 提取 U10, V10, U100, V100 和 Power
    5. 计算合成风速 WS10, WS100，以及风能立方 WS100_Cube
    6. 计算风向的 Sin_WDir 和 Cos_WDir
    7. 拼接出 13 维的气象时间特征群 (weather_features)
    8. 返回 weather_features 和 power_data
    """
    print(f"[{time.strftime('%H:%M:%S')}] 1. 正在加载原始数据...")
    data = np.load(file_path, allow_pickle=True)

    timestamps = pd.to_datetime(data[:, 0, 0])
    hours = timestamps.hour.values
    months = timestamps.month.values

    sin_hour = np.sin(2 * np.pi * hours / 24)
    cos_hour = np.cos(2 * np.pi * hours / 24)
    sin_month = np.sin(2 * np.pi * months / 12)
    cos_month = np.cos(2 * np.pi * months / 12)

    num_farms = data.shape[1]

    Sin_Hour = np.repeat(sin_hour[:, np.newaxis], num_farms, axis=1)
    Cos_Hour = np.repeat(cos_hour[:, np.newaxis], num_farms, axis=1)
    Sin_Month = np.repeat(sin_month[:, np.newaxis], num_farms, axis=1)
    Cos_Month = np.repeat(cos_month[:, np.newaxis], num_farms, axis=1)

    numeric_data = data[:, :, 1:].astype(np.float32)
    U10, V10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
    U100, V100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
    Power = numeric_data[:, :, 4]

    print(f"[{time.strftime('%H:%M:%S')}] 2. 正在计算物理与周期特征...")
    WS10 = np.sqrt(U10 ** 2 + V10 ** 2)
    WS100 = np.sqrt(U100 ** 2 + V100 ** 2)
    WS100_Cube = WS100 ** 3

    Wind_Dir_Rad = np.arctan2(V100, U100)
    Sin_WDir = np.sin(Wind_Dir_Rad)
    Cos_WDir = np.cos(Wind_Dir_Rad)

    weather_features = np.stack([
        U10, V10, U100, V100,
        WS10, WS100,
        WS100_Cube,
        Sin_WDir, Cos_WDir,
        Sin_Hour, Cos_Hour,
        Sin_Month, Cos_Month,
    ], axis=-1).astype(np.float32)

    print(f"   => 构造完毕！单时间点气象时间特征维度: {weather_features.shape[2]}")
    return weather_features, Power


def build_spatiotemporal_sliding_window(weather_features, power_data, window_size=24):
    """
    第二步：构建融入全局空间信息的滑动窗口矩阵

    1. 计算样本总数 num_samples
    2. 计算总特征维度:
       - 全局历史出力: num_farms * window_size
       - 当前风场历史统计: 4
       - 当前风场过去气象: window_size * 13
       - 当前风场未来气象: window_size * 13
    3. 初始化 X_all, Y_all
    4. 按滑窗时间步构建样本
    5. 对每个目标风场拼接空间+时序特征
    """
    print(f"[{time.strftime('%H:%M:%S')}] 3. 正在构建 V3 空间融合滑动窗口...")

    num_time_steps, num_farms, num_weather_features = weather_features.shape
    num_samples = num_time_steps - 2 * window_size + 1
    if num_samples <= 0:
        raise ValueError("时间步不足，无法构建包含过去与未来窗口的样本。")

    global_power_dim = num_farms * window_size
    feature_dim = global_power_dim + 4 + window_size * num_weather_features + window_size * num_weather_features

    X_all = np.zeros((num_samples, num_farms, feature_dim), dtype=np.float32)
    Y_all = np.zeros((num_samples, num_farms, window_size), dtype=np.float32)

    print("   => 维度配置:")
    print(f"      全局历史出力维度: {global_power_dim}")
    print("      当前风场历史统计维度: 4")
    print(f"      当前风场过去气象维度: {window_size * num_weather_features}")
    print(f"      当前风场未来气象维度: {window_size * num_weather_features}")
    print(f"      总特征维度 feature_dim: {feature_dim}")

    for sample_idx, i in enumerate(range(window_size, num_time_steps - window_size + 1)):
        past_power_global_matrix = power_data[i - window_size:i, :]
        past_power_global = past_power_global_matrix.reshape(-1)

        for farm_id in range(num_farms):
            past_power_local = past_power_global_matrix[:, farm_id]
            p_mean = np.mean(past_power_local)
            p_std = np.std(past_power_local)
            p_max = np.max(past_power_local)
            p_min = np.min(past_power_local)
            past_power_stats = np.array([p_mean, p_std, p_max, p_min], dtype=np.float32)

            past_weather = weather_features[i - window_size:i, farm_id, :].reshape(-1)
            future_weather = weather_features[i:i + window_size, farm_id, :].reshape(-1)

            x_row = np.concatenate([
                past_power_global,
                past_power_stats,
                past_weather,
                future_weather,
            ], axis=0)
            y_row = power_data[i:i + window_size, farm_id]

            X_all[sample_idx, farm_id, :] = x_row
            Y_all[sample_idx, farm_id, :] = y_row

        if (sample_idx + 1) % 2000 == 0 or sample_idx == num_samples - 1:
            print(f"   ... 已处理 {sample_idx + 1}/{num_samples} 个滑动窗口")

    return X_all, Y_all


def main():
    """
    第三步：主流程控制与压缩保存
    1. 检查数据文件是否存在
    2. 调用第一步获取基础特征
    3. 调用第二步获取 X_all 和 Y_all
    4. 打印最终 shape
    5. 压缩保存为 features_v3.npz
    """
    file_path = "wind_train_val_2012-01-02_to_2013-07-13.npy"
    output_filename = "features_v3.npz"

    if not os.path.exists(file_path):
        print(f"找不到原始数据文件: {file_path}")
        return

    weather_features, power_data = load_and_extract_advanced_features(file_path)
    X_all, Y_all = build_spatiotemporal_sliding_window(
        weather_features,
        power_data,
        window_size=24,
    )

    print(f"[{time.strftime('%H:%M:%S')}] 4. 正在压缩保存数据...")
    print(f"   => 最终特征矩阵 X 的形状: {X_all.shape}")
    print(f"   => 最终标签矩阵 Y 的形状: {Y_all.shape}")

    np.savez_compressed(output_filename, X=X_all, Y=Y_all)
    print(f"[{time.strftime('%H:%M:%S')}] 已完成，输出文件: {output_filename}")
  
if __name__ == "__main__":
    main()
