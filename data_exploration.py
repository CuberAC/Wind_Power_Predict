import numpy as np
import pandas as pd
import os

def explore_data():
    default_path = 'data/wind_test_cleaned.npy'
    file_path = default_path if os.path.exists(default_path) else 'wind_train_val_2012-01-02_to_2013-07-13.npy'
    data = np.load(file_path, allow_pickle=True)

    print("="*50)
    print("1. 数据基本信息")

    timestamps = pd.to_datetime(data[:, 0, 0])
    numeric_data = data[:, :, 1:].astype(np.float32)

    print(f"数据文件: {file_path}")
    print(f"原始数据维度: {data.shape} (时间, 风场, 字段)")
    print(f"时间序列长度: {len(timestamps)} (从 {timestamps[0]} 到 {timestamps[-1]})")
    print(f"纯数值数据维度: {numeric_data.shape} -> (时间点, 风电场, 气象与出力特征)")
    print("="*50)

    feature_names = ['U10 (风速分量)', 'V10 (风速分量)', 'U100 (高空风速)', 'V100 (高空风速)', 'Power (风电出力)']

    print("2. 顺序样例数据")
    print("字段顺序: [Timestamp, Farm, U10, V10, U100, V100, Power]")

    sample_rows = 10
    max_time = min(sample_rows, data.shape[0])
    max_farm = min(3, data.shape[1])

    for t in range(max_time):
        for f in range(max_farm):
            values = numeric_data[t, f, :]
            print(
                f"t={t:04d}, Farm={f}, Time={timestamps[t]}, "
                f"U10={values[0]:.4f}, V10={values[1]:.4f}, "
                f"U100={values[2]:.4f}, V100={values[3]:.4f}, Power={values[4]:.4f}"
            )

    print("="*50)
    print(f"共打印前 {max_time} 个时间步、每步前 {max_farm} 个风场样例。")


if __name__ == "__main__":
    explore_data()