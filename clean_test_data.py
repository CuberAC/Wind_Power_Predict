# ==============================================================================
# 脚本名称: 00_clean_test_data.py
# 任务目标: 严格按照 3σ 原则清洗测试集，剔除污染严重的天数，余下进行线性插值
# ==============================================================================

import numpy as np
import pandas as pd
import os

def clean_test_dataset(input_file='data/wind_test_2013-07-14_to_end.npy', output_file='data/wind_test_cleaned.npy'):
    if not os.path.exists(input_file):
        print(f"❌ 找不到测试集文件: {input_file}")
        return

    print(f"📦 正在加载原始测试集: {input_file}...")
    data = np.load(input_file, allow_pickle=True)
    # data 维度: (Time_Steps, 10, 6) -> [时间戳, U10, V10, U100, V100, TARGETVAR]
    
    # 1. 提取时间特征，用于按“天”进行分组
    timestamps = pd.to_datetime(data[:, 0, 0])
    dates = timestamps.date
    
    # 提取纯数值矩阵
    numeric_data = data[:, :, 1:].astype(float)
    num_steps, num_farms, num_features = numeric_data.shape
    
    print("🔍 正在执行 3σ 异常值与 NaN 检测...")
    # 2. 计算每个风场、每个特征的全局 均值(mu) 和 标准差(sigma)
    # 使用 nanmean 和 nanstd 忽略原有的 NaN 值
    mu = np.nanmean(numeric_data, axis=0)       # 形状 (10, 5)
    sigma = np.nanstd(numeric_data, axis=0)     # 形状 (10, 5)
    
    # 3. 标记异常值 (NaN 或者是超出 mu ± 3*sigma 的值)
    is_nan = np.isnan(numeric_data)
    lower_bound = mu - 3 * sigma
    upper_bound = mu + 3 * sigma
    is_outlier = (numeric_data < lower_bound) | (numeric_data > upper_bound)
    
    # 得到综合异常掩码 (True 代表这个点是坏的)
    is_anomaly = is_nan | is_outlier
    
    # 4. 按天统计异常，执行 "1/6 废弃原则"
    # 如果某一个小时内，任意一个风场的任意一个特征坏了，这个小时就算“污染小时”
    # (为了保证 10 个风场的时间步完全对齐，我们做全局同步丢弃)
    anomalous_hours = np.any(is_anomaly, axis=(1, 2)) # 形状 (Time_Steps,)
    
    df_time = pd.DataFrame({'date': dates, 'is_anomaly': anomalous_hours})
    daily_anomaly_count = df_time.groupby('date')['is_anomaly'].sum()
    
    # 一天有 24 小时，1/6 就是 4 个小时。超过 4 个小时污染，这天就全废掉
    bad_dates = daily_anomaly_count[daily_anomaly_count > 4].index.values
    
    print(f"📊 统计结果: 总计 {len(daily_anomaly_count)} 天。")
    print(f"🗑️ 其中 {len(bad_dates)} 天污染严重(超4小时)，将被整天剔除。")
    
    # 5. 剔除坏天数的数据
    valid_mask = ~np.isin(dates, bad_dates)
    filtered_data = data[valid_mask].copy()
    filtered_numeric = numeric_data[valid_mask].copy()
    filtered_anomaly = is_anomaly[valid_mask].copy()
    
    print(f"⏳ 剔除后，剩余有效时间步: {filtered_data.shape[0]} / {num_steps}")
    
    # 6. 对剩余数据中零星的异常值进行线性插值
    print("🩹 正在对剩余的零星异常点执行线性插值(Linear Interpolation)...")
    # 先把超出 3σ 的真实离群点也变成 NaN，交给 Pandas 统一插值
    filtered_numeric[filtered_anomaly] = np.nan 
    
    for f in range(num_farms):
        for c in range(num_features):
            series = pd.Series(filtered_numeric[:, f, c])
            # interpolate 处理中间的空值，ffill/bfill 处理可能在头尾的空值
            filtered_numeric[:, f, c] = series.interpolate(method='linear').ffill().bfill().values
            
    # 7. 将清洗完毕的数值矩阵放回原数组
    filtered_data[:, :, 1:] = filtered_numeric
    
    # 8. 保存文件
    np.save(output_file, filtered_data, allow_pickle=True)
    print("="*50)
    print(f"🎉 终极纯净版测试集已生成: {output_file}")
    print("="*50)

if __name__ == "__main__":
    clean_test_dataset()