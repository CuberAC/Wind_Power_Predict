"""
File: data_exploration.py
Description: 对风电数据集进行全方位体检，验证特征维度、数据范围、缺失值与空间相关性。
"""

import numpy as np
import pandas as pd

def explore_data(file_path="data/wind_train_val_2012-01-02_to_2013-07-13.npy"):
    print(f"🚀 正在加载数据: {file_path} ...\n")
    
    try:
        # 带有字符串的 numpy 数组必须用 allow_pickle=True 加载
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # ==========================================
    # 1. 基础维度与类型检查
    # ==========================================
    print("-" * 50)
    print("📊 第一部分：基础维度与类型")
    print("-" * 50)
    print(f"[*] 数据整体形状 (Shape): {data.shape}")
    print(f"[*] 数据整体类型 (Dtype): {data.dtype}  <-- (如果是 object，说明混杂了字符串和数字)")
    
    if len(data.shape) != 3 or data.shape[1] != 10 or data.shape[2] != 6:
        print("⚠️ 警告：数据维度不是我们假设的 (Time, 10, 6)！请仔细检查！")
    else:
        print("✅ 维度检查通过：完美符合 (Time, Nodes=10, Features=6)！")

    # ==========================================
    # 2. 时间戳检查 (Index 0)
    # ==========================================
    print("\n" + "-" * 50)
    print("🕒 第二部分：时间戳检查 (Index 0)")
    print("-" * 50)
    # 取第 0 个风电场的时间戳序列
    timestamps = data[:, 0, 0]
    print(f"[*] 前 3 个时间戳样本: {timestamps[:3].tolist()}")
    print(f"[*] 后 3 个时间戳样本: {timestamps[-3:].tolist()}")
    
    try:
        dt_series = pd.to_datetime(timestamps)
        time_diff = dt_series.diff().mode()[0] # 计算最常见的时间间隔
        print(f"[*] 时间类型解析: 成功！解析为 pandas Datetime")
        print(f"[*] 数据时间跨度: 从 {dt_series.min()} 到 {dt_series.max()}")
        print(f"[*] 采样频率 (Time Delta): {time_diff}")
        
        # 检查是否有时间断层（缺失的时刻）
        expected_steps = int((dt_series.max() - dt_series.min()) / time_diff) + 1
        if len(dt_series) == expected_steps:
            print("✅ 时间连续性检查通过：没有任何时间断层！")
        else:
            print(f"⚠️ 警告：存在时间断层！预期 {expected_steps} 步，实际 {len(dt_series)} 步。")
    except Exception as e:
        print(f"❌ 时间戳解析失败，请检查格式: {e}")

    # ==========================================
    # 3. 气象特征检查 (Index 1-4)
    # ==========================================
    print("\n" + "-" * 50)
    print("🌤️ 第三部分：气象特征检查 (Index 1-4: U10, V10, U100, V100)")
    print("-" * 50)
    # 取出气象数据并强制转换为 float
    weather_data = data[:, :, 1:5].astype(float)
    
    # 检查 NaN
    nan_count_weather = np.isnan(weather_data).sum()
    print(f"[*] 气象数据中 NaN 值的数量: {nan_count_weather}")
    if nan_count_weather > 0:
        print("⚠️ 警告：气象数据存在缺失值，需要插值处理！")
        
    for i, name in enumerate(["U10 (Index 1)", "V10 (Index 2)", "U100 (Index 3)", "V100 (Index 4)"]):
        feature_slice = weather_data[:, :, i]
        print(f"  > {name:15s} | Min: {np.nanmin(feature_slice):8.3f} | Max: {np.nanmax(feature_slice):8.3f} | Mean: {np.nanmean(feature_slice):8.3f}")

    # ==========================================
    # 4. 目标特征检查 (Index 5: Power)
    # ==========================================
    print("\n" + "-" * 50)
    print("⚡ 第四部分：风电出力检查 (Index 5: Power)")
    print("-" * 50)
    power_data = data[:, :, 5].astype(float)
    
    nan_count_power = np.isnan(power_data).sum()
    print(f"[*] 出力数据中 NaN 值的数量: {nan_count_power}")
    
    p_min = np.nanmin(power_data)
    p_max = np.nanmax(power_data)
    print(f"[*] 出力数据范围 | Min: {p_min:.4f} | Max: {p_max:.4f} | Mean: {np.nanmean(power_data):.4f}")
    
    if p_min < 0:
        print("⚠️ 警告：风电出力存在负值！这通常不符合物理直觉，可能是弃风记录或传感器错误。")
    if p_max <= 1.0:
        print("✅ 提示：风电出力的最大值 <= 1.0，数据极有可能已经被【归一化】（Normalized by Capacity），类似于学长论文里的处理！")
    else:
        print("提示：风电出力的最大值 > 1.0，说明这是原始功率（MW），进模型前一定要做归一化！")

    # ==========================================
    # 5. GNN 可行性分析 (空间相关性)
    # ==========================================
    print("\n" + "-" * 50)
    print("🌐 第五部分：GNN 可行性验证 (节点相关性)")
    print("-" * 50)
    # 计算 10 个风电场出力的相关系数矩阵
    # 过滤掉 NaN 以便计算
    power_data_clean = np.nan_to_num(power_data)
    corr_matrix = np.corrcoef(power_data_clean.T)
    
    # 提取非对角线元素
    off_diag = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
    print(f"[*] 10个风电场之间的 Pearson 相关系数统计:")
    print(f"  > 最高相关系数: {off_diag.max():.4f}")
    print(f"  > 最低相关系数: {off_diag.min():.4f}")
    print(f"  > 平均相关系数: {off_diag.mean():.4f}")
    
    if off_diag.mean() > 0.4:
        print("✅ 结论：风场间存在较强的空间联动性，使用 GNN 绝对是个正确的方向！")
    else:
        print("⚠️ 警告：风场间整体相关性偏弱，GNN 的提升可能有限。")
        
    print("\n🎉 体检完成！请将控制台输出发给我看看！")

if __name__ == "__main__":
    explore_data("data/wind_train_val_2012-01-02_to_2013-07-13.npy")