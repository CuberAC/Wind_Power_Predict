import numpy as np
import pandas as pd
import os


def load_and_extract_basic_features(file_path):
    """
    第一步：加载原始数据并计算基础物理特征
    1. 使用 np.load 加载 npy 文件，注意设置 allow_pickle=True
    2. 提取出 10个风电场的数值数据 (剥离第0列的时间戳)
    3. 针对提取的数值数据，计算合成风速:
       - WS10 = sqrt(U10^2 + V10^2)
       - WS100 = sqrt(U100^2 + V100^2)
    4. 将特征重新堆叠，顺序为: [U10, V10, U100, V100, WS10, WS100, Power]
    5. 返回处理后的三维矩阵，形状应为 (13416, 10, 7)
    """
    data = np.load(file_path, allow_pickle=True)

    # 原始维度: (时间步, 风场数, 6)，第0列是时间戳
    numeric_data = data[:, :, 1:].astype(np.float32)

    U10 = numeric_data[:, :, 0]
    V10 = numeric_data[:, :, 1]
    U100 = numeric_data[:, :, 2]
    V100 = numeric_data[:, :, 3]
    power = numeric_data[:, :, 4]

    ws10 = np.sqrt(U10 ** 2 + V10 ** 2)
    ws100 = np.sqrt(U100 ** 2 + V100 ** 2)

    processed_data = np.stack([U10, V10, U100, V100, ws10, ws100, power], axis=-1)
    return processed_data


def build_sliding_window_dataset(processed_data, window_size=24):
    """
    第二步：利用滑动窗口构建监督学习所需的 X 和 Y
    输入数据维度: (时间步, 风场数, 7)
    输出 X 维度: (样本数, 风场数, 312) -> 312 = 24(历史出力) + 24*6(历史气象) + 24*6(未来气象)
    输出 Y 维度: (样本数, 风场数, 24)  -> 未来24小时的真实出力

    逻辑：
    1. 遍历时间步从 window_size 到 len(data) - window_size
    2. 对每个时间步，遍历 10 个风电场：
       - 获取过去24小时的历史出力 (长度24)
       - 获取过去24小时的气象特征 (24 x 6) 并展平 (长度144)
       - 获取未来24小时的预报气象特征 (24 x 6) 并展平 (长度144)
       - 拼接成一维向量 X_row (长度312)
       - 获取未来24小时的实际出力作为 Y_row (长度24)
    3. 将所有样本存入列表并转为 numpy 数组返回。
    """
    X_all = []
    Y_all = []

    power_idx = 6
    weather_idxs = [0, 1, 2, 3, 4, 5]

    total_time_steps, n_farms, _ = processed_data.shape

    for t in range(window_size, total_time_steps - window_size + 1):
        X_t = []
        Y_t = []

        for farm_id in range(n_farms):
            farm_series = processed_data[:, farm_id, :]

            past_24h = farm_series[t - window_size:t]
            future_24h = farm_series[t:t + window_size]

            past_power = past_24h[:, power_idx]
            past_weather = past_24h[:, weather_idxs].reshape(-1)
            future_weather = future_24h[:, weather_idxs].reshape(-1)

            x_row = np.concatenate([past_power, past_weather, future_weather])
            y_row = future_24h[:, power_idx]

            X_t.append(x_row)
            Y_t.append(y_row)

        X_all.append(X_t)
        Y_all.append(Y_t)

    X_all = np.asarray(X_all, dtype=np.float32)
    Y_all = np.asarray(Y_all, dtype=np.float32)
    return X_all, Y_all


def main():
    """
    第三步：主流程控制与文件保存
    1. 定义原始文件路径 file_path
    2. 调用 load_and_extract_basic_features 获取基础特征矩阵
    3. 调用 build_sliding_window_dataset 获取 X_all 和 Y_all
    4. 打印 X_all 和 Y_all 的 shape 以便检查维度是否正确
    5. 使用 np.savez 将 X_all 和 Y_all 压缩保存为 'features_v1.npz' 文件
    6. 打印保存成功的提示信息
    """
    file_path = "wind_train_val_2012-01-02_to_2013-07-13.npy"
    output_path = "features_v1.npz"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到输入文件: {file_path}")

    processed_data = load_and_extract_basic_features(file_path)
    X_all, Y_all = build_sliding_window_dataset(processed_data, window_size=24)

    print(f"processed_data shape: {processed_data.shape}")
    print(f"X_all shape: {X_all.shape}")
    print(f"Y_all shape: {Y_all.shape}")

    np.savez(output_path, X=X_all, Y=Y_all)
    print(f"特征工程完成，文件已保存: {output_path}")


if __name__ == "__main__":
    main()
