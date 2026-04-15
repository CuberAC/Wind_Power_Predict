import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import time
import argparse
import warnings

# ==========================================
# 模块一：内置特征工程 (供训练和未来预测复用)
# ==========================================
def extract_features_from_raw(data):
    """从原始 npy 数组中提取深度气象与时间特征"""
    # 1. 时间周期特征
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

    # 2. 剥离气象与出力
    numeric_data = data[:, :, 1:].astype(np.float32)
    U10, V10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
    U100, V100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
    Power = numeric_data[:, :, 4]
    
    # 3. 计算物理特性 (合成风速、风能立方、风向编码)
    WS10 = np.sqrt(U10**2 + V10**2)
    WS100 = np.sqrt(U100**2 + V100**2)
    WS100_Cube = WS100 ** 3
    Wind_Dir_Rad = np.arctan2(V100, U100)
    Sin_WDir = np.sin(Wind_Dir_Rad)
    Cos_WDir = np.cos(Wind_Dir_Rad)
    
    # 4. 拼装特征群
    weather_features = np.stack([
        U10, V10, U100, V100, WS10, WS100, WS100_Cube, 
        Sin_WDir, Cos_WDir, Sin_Hour, Cos_Hour, Sin_Month, Cos_Month
    ], axis=-1)
    
    return weather_features, Power

def build_sliding_window_in_memory(weather_features, power_data, window_size=24):
    """在内存中构建滑动窗口监督学习矩阵 X 和 Y"""
    num_time_steps = weather_features.shape[0]
    num_farms = weather_features.shape[1]
    num_samples = num_time_steps - window_size * 2 + 1
    
    # 维度: 24(过去出力) + 4(统计量) + 24*13(过去气象) + 24*13(未来气象) = 652维
    feature_dim = window_size + 4 + (window_size * 13) + (window_size * 13)
    
    X_all = np.zeros((num_samples, num_farms, feature_dim), dtype=np.float32)
    Y_all = np.zeros((num_samples, num_farms, window_size), dtype=np.float32)
    
    sample_idx = 0
    for i in range(window_size, num_time_steps - window_size + 1):
        for farm_id in range(num_farms):
            past_power = power_data[i-window_size : i, farm_id]
            past_power_stats = np.array([
                np.mean(past_power), np.std(past_power), 
                np.max(past_power), np.min(past_power)
            ])
            past_weather = weather_features[i-window_size : i, farm_id, :].flatten()
            future_weather = weather_features[i : i+window_size, farm_id, :].flatten()
            
            X_all[sample_idx, farm_id, :] = np.concatenate([past_power, past_power_stats, past_weather, future_weather])
            Y_all[sample_idx, farm_id, :] = power_data[i : i+window_size, farm_id]
            
        sample_idx += 1
    return X_all, Y_all

# ==========================================
# 模块二：LightGBM 模型训练与评估
# ==========================================
def train_lgb_for_farm(farm_id, X_train, Y_train, X_val, Y_val, save_path):
    """训练单个风电场的 LightGBM 多输出模型"""
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始训练风电场 {farm_id} 的 LGBM 模型...")
    start_time = time.time()

    # 统一输入为 numpy，避免 DataFrame/ndarray 混用触发 feature names 警告
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    Y_val = np.asarray(Y_val, dtype=np.float32)
    
    # 定义 LightGBM 核心参数
    base_model = lgb.LGBMRegressor(
        n_estimators=600,           # Optuna 最优参数
        learning_rate=0.013682416119938175,
        num_leaves=41,
        max_depth=11,
        subsample=0.6404736081419506,
        colsample_bytree=0.743050136518371,
        min_child_samples=56,
        objective='regression',
        n_jobs=-1,                  # 调用所有 CPU 核心
        random_state=42,
        verbosity=-1                # 关闭 LightGBM 的底层警告输出
    )
    
    # 包装为多输出回归器 (预测未来 24 小时)
    model = MultiOutputRegressor(base_model)
    
    # 训练模型
    model.fit(X_train, Y_train)
    train_time = time.time() - start_time
    
    # 验证评估
    Y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
    mae = mean_absolute_error(Y_val, Y_pred)
    
    print(f"[{time.strftime('%H:%M:%S')}] 训练完成! 耗时: {train_time:.1f} 秒 | 验证集 RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    
    # 保存模型
    joblib.dump(model, save_path)
    return rmse, mae

# ==========================================
# 模块三：主流程 (端到端运行)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="端到端 LightGBM 风电预测训练")
    # 直接接收原始 npy 文件
    parser.add_argument('--raw_data', type=str, default='data/wind_train_val_2012-01-02_to_2013-07-13.npy', help='原始npy数据的路径')
    args = parser.parse_args()
    
    if not os.path.exists(args.raw_data):
        print(f"❌ 找不到原始文件 {args.raw_data}，请检查路径！")
        return
        
    # 仅屏蔽 sklearn 的该条重复警告，其他警告仍保持可见
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
        category=UserWarning,
        module="sklearn"
    )

    print(f"[{time.strftime('%H:%M:%S')}] 📦 1. 正在加载原始数据集: {args.raw_data}")
    raw_data = np.load(args.raw_data, allow_pickle=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ 2. 正在进行内存级特征工程 (动态构建特征)...")
    weather_features, power_data = extract_features_from_raw(raw_data)
    X_all, Y_all = build_sliding_window_in_memory(weather_features, power_data, window_size=24)
    
    num_samples, num_farms, feature_dim = X_all.shape
    print(f"   => 构建完成！风电场数: {num_farms} | 样本数: {num_samples} | 单个样本特征维度: {feature_dim}")
    
    # 划分训练集和验证集 (80% 训练, 20% 验证)
    split_ratio = 0.8
    split_idx = int(num_samples * split_ratio)
    
    # 创建保存目录
    save_dir = 'saved_models/lgbm_end2end'
    os.makedirs(save_dir, exist_ok=True)
    
    total_rmse, total_mae = 0.0, 0.0
    
    print(f"\n🚀 3. 开始批量训练 LightGBM 模型...")
    for farm_id in range(num_farms):
        X_train, Y_train = X_all[:split_idx, farm_id, :], Y_all[:split_idx, farm_id, :]
        X_val, Y_val     = X_all[split_idx:, farm_id, :], Y_all[split_idx:, farm_id, :]
        
        model_path = os.path.join(save_dir, f'lgbm_farm_{farm_id}.pkl')
        
        rmse, mae = train_lgb_for_farm(farm_id, X_train, Y_train, X_val, Y_val, model_path)
        total_rmse += rmse
        total_mae += mae
        
    print("\n" + "="*50)
    print("🎉 所有 10 个风电场的 LightGBM 模型端到端训练完毕！")
    print(f"   【全局平均 RMSE】: {total_rmse / num_farms:.4f}")
    print(f"   【全局平均 MAE 】: {total_mae / num_farms:.4f}")
    print(f"   💾 模型统一保存在: {save_dir}/ 目录下")
    print("="*50)

if __name__ == "__main__":
    main()