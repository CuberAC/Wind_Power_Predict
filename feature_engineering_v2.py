# ==============================================================================
# 任务目标: 深度风电特征工程 (V2) - 引入风能立方、风向编码、时间周期及历史统计量
# 数据输出: data/features_v2.npz (包含特征矩阵 X 和 标签矩阵 Y)
# ==============================================================================

import numpy as np
import pandas as pd
import os
import time

def load_and_extract_advanced_features(file_path):
    """
    第一步：加载数据并进行深度的向量化特征重构
    """
    print(f"[{time.strftime('%H:%M:%S')}] 1. 正在加载原始数据...")
    data = np.load(file_path, allow_pickle=True)
    
    # 1. 提取时间戳，并生成周期性时间特征 (广播到所有风电场)
    timestamps = pd.to_datetime(data[:, 0, 0])
    hours = timestamps.hour.values
    months = timestamps.month.values
    
    # 将时间转化为正余弦，映射到圆上
    sin_hour = np.sin(2 * np.pi * hours / 24)
    cos_hour = np.cos(2 * np.pi * hours / 24)
    sin_month = np.sin(2 * np.pi * months / 12)
    cos_month = np.cos(2 * np.pi * months / 12)
    
    num_time_steps = data.shape[0]
    num_farms = data.shape[1]
    
    # 将 1 维的时间特征扩展为 (时间步, 风电场数) 的二维矩阵，以便后续拼接
    Sin_Hour = np.repeat(sin_hour[:, np.newaxis], num_farms, axis=1)
    Cos_Hour = np.repeat(cos_hour[:, np.newaxis], num_farms, axis=1)
    Sin_Month = np.repeat(sin_month[:, np.newaxis], num_farms, axis=1)
    Cos_Month = np.repeat(cos_month[:, np.newaxis], num_farms, axis=1)

    # 2. 剥离纯气象与出力数据
    numeric_data = data[:, :, 1:].astype(np.float32)
    U10, V10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
    U100, V100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
    Power = numeric_data[:, :, 4]
    
    print(f"[{time.strftime('%H:%M:%S')}] 2. 正在计算物理与空间特征...")
    # 3. 计算 V1 的合成风速
    WS10 = np.sqrt(U10**2 + V10**2)
    WS100 = np.sqrt(U100**2 + V100**2)
    
    # 4. 新增：计算风能立方 (贝茨极限物理特性)
    WS100_Cube = WS100 ** 3
    
    # 5. 新增：计算风向角度编码 (arctan2返回 -pi 到 pi，直接求sin和cos完美避开断层)
    Wind_Dir_Rad = np.arctan2(V100, U100)
    Sin_WDir = np.sin(Wind_Dir_Rad)
    Cos_WDir = np.cos(Wind_Dir_Rad)
    
    # 6. 拼装强大的气象+时间特征群 (不包含Power)
    # 顺序: U, V, WS, Cube, Dir(sin/cos), Hour(sin/cos), Month(sin/cos)
    weather_features = np.stack([
        U10, V10, U100, V100,          # 0-3: 原始分量
        WS10, WS100,                   # 4-5: 合成风速
        WS100_Cube,                    # 6: 风能立方
        Sin_WDir, Cos_WDir,            # 7-8: 风向编码
        Sin_Hour, Cos_Hour,            # 9-10: 日周期
        Sin_Month, Cos_Month           # 11-12: 年周期
    ], axis=-1)
    
    print(f"   => 构造完毕！单个时间点的气象特征数飙升至: {weather_features.shape[2]} 维")
    return weather_features, Power

def build_advanced_sliding_window(weather_features, power_data, window_size=24):
    """
    第二步：利用滑动窗口构建 X 和 Y，并在过程中融入“历史时序统计特征”
    """
    print(f"[{time.strftime('%H:%M:%S')}] 3. 正在滚动构建机器学习矩阵 (切分时序片段)...")
    X, Y = [], []
    num_time_steps = weather_features.shape[0]
    num_farms = weather_features.shape[1]
    
    # 提前初始化一个全零的三维矩阵，比用 list.append 更节省内存且速度快
    # 样本总数 = 总时长 - 24(历史) - 24(未来) + 1
    num_samples = num_time_steps - window_size * 2 + 1
    
    # 计算新的 X 维度：
    # 24 (过去出力) + 4 (过去出力统计) + 24*13 (过去气象) + 24*13 (未来气象) = 652 维
    feature_dim = window_size + 4 + (window_size * 13) + (window_size * 13)
    
    X_all = np.zeros((num_samples, num_farms, feature_dim), dtype=np.float32)
    Y_all = np.zeros((num_samples, num_farms, window_size), dtype=np.float32)
    
    sample_idx = 0
    for i in range(window_size, num_time_steps - window_size + 1):
        for farm_id in range(num_farms):
            # 1. 获取过去 24 小时出力
            past_power = power_data[i-window_size : i, farm_id]
            
            # 2. 新增：过去 24 小时出力的统计浓缩特征
            p_mean = np.mean(past_power)
            p_std = np.std(past_power)
            p_max = np.max(past_power)
            p_min = np.min(past_power)
            past_power_stats = np.array([p_mean, p_std, p_max, p_min])
            
            # 3. 过去 24 小时气象展平
            past_weather = weather_features[i-window_size : i, farm_id, :].flatten()
            
            # 4. 未来 24 小时预报气象展平
            future_weather = weather_features[i : i+window_size, farm_id, :].flatten()
            
            # 拼接极其丰富的一维特征向量
            x_row = np.concatenate([past_power, past_power_stats, past_weather, future_weather])
            y_row = power_data[i : i+window_size, farm_id]
            
            X_all[sample_idx, farm_id, :] = x_row
            Y_all[sample_idx, farm_id, :] = y_row
            
        sample_idx += 1
        # 打印进度条
        if sample_idx % 2000 == 0:
            print(f"   ... 已处理 {sample_idx}/{num_samples} 个滑动窗口")

    return X_all, Y_all

def main():
    file_path = 'data/wind_train_val_2012-01-02_to_2013-07-13.npy'
    if not os.path.exists(file_path):
        print(f"❌ 找不到原始数据文件 {file_path}")
        return
        
    weather_features, power_data = load_and_extract_advanced_features(file_path)
    
    X_all, Y_all = build_advanced_sliding_window(weather_features, power_data, window_size=24)
    
    print(f"[{time.strftime('%H:%M:%S')}] 4. 正在压缩保存数据...")
    print(f"   => 最终特征矩阵 X 的形状: {X_all.shape}")
    print(f"   => 最终标签矩阵 Y 的形状: {Y_all.shape}")
    
    # 采用 np.savez_compressed 能够大幅减小生成的文件体积
    output_filename = 'data/features_v2.npz'
    np.savez_compressed(output_filename, X=X_all, Y=Y_all)
    
    print(f"[{time.strftime('%H:%M:%S')}] ✅ 恭喜！深度特征文件 {output_filename} 已成功生成。")

if __name__ == "__main__":
    main()