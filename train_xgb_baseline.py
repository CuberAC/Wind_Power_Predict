import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import time

def load_and_preprocess_data(file_path):
    """
    加载数据，清洗时间列，并构造合成风速特征
    """
    print("1. 正在加载数据...")
    data = np.load(file_path, allow_pickle=True)
    
    # 剥离时间列，保留纯数值 (U10, V10, U100, V100, Power)
    numeric_data = data[:, :, 1:].astype(np.float32)
    
    # 获取原始气象特征和出力
    U10 = numeric_data[:, :, 0]
    V10 = numeric_data[:, :, 1]
    U100 = numeric_data[:, :, 2]
    V100 = numeric_data[:, :, 3]
    Power = numeric_data[:, :, 4]
    
    # 🌟 特征工程：计算合成风速大小
    WS10 = np.sqrt(U10**2 + V10**2)
    WS100 = np.sqrt(U100**2 + V100**2)
    
    # 重新拼接特征，维度变为 (时间点, 风电场, 7)
    # 特征顺序: [U10, V10, U100, V100, WS10, WS100, Power]
    processed_data = np.stack([U10, V10, U100, V100, WS10, WS100, Power], axis=-1)
    
    print(f"数据预处理完成！加入了合成风速。当前数据维度: {processed_data.shape}")
    return processed_data

def build_dataset_for_farm(farm_data, window_size=24):
    """
    将时序数据“拍扁”成 XGBoost 需要的表格格式 X 和 Y
    输入数据特征顺序: [0:U10, 1:V10, 2:U100, 3:V100, 4:WS10, 5:WS100, 6:Power]
    """
    print("2. 正在构造特征工程矩阵 (Sliding Window)...")
    X, Y = [], []
    
    power_idx = 6 # 出力在最后一列
    weather_idxs = [0, 1, 2, 3, 4, 5] # 前6列全是气象特征
    
    # 滑动窗口机制
    # i 代表“预测时刻的起点”。为了保证有过去24h和未来24h的数据，i的取值范围受限
    for i in range(window_size, len(farm_data) - window_size + 1):
        
        # 提取过去 24h 的数据
        past_24h = farm_data[i-window_size : i]
        past_power = past_24h[:, power_idx]                   # 过去 24h 历史出力 (24个值)
        past_weather = past_24h[:, weather_idxs].flatten()    # 过去 24h 气象数据 (24 * 6 = 144个值)
        
        # 提取预测当天(未来) 24h 的气象数据
        future_24h = farm_data[i : i+window_size]
        future_weather = future_24h[:, weather_idxs].flatten() # 预测当天 24h 气象预报 (24 * 6 = 144个值)
        
        # 🎯 拼接所有的 X (24 + 144 + 144 = 312个特征)
        x_row = np.concatenate([past_power, past_weather, future_weather])
        
        # 🎯 提取目标 Y (预测当天的 24h 实际出力)
        y_row = future_24h[:, power_idx]
        
        X.append(x_row)
        Y.append(y_row)
        
    return np.array(X), np.array(Y)

def train_xgboost_for_farm(farm_id, X, Y):
    """
    划分 80-20 数据集，训练 XGBoost 模型并评估
    """
    # 3. 按时间顺序划分训练集(80%)和验证集(20%)
    # 时序预测绝不能用随机打乱(train_test_split自带shuffle)，必须按时间截断！
    split_ratio = 0.8
    split_idx = int(len(X) * split_ratio)
    
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]
    
    print(f"\n3. 开始训练 Farm {farm_id} 的模型...")
    print(f"   -> 训练集样本数: {len(X_train)}")
    print(f"   -> 验证集样本数: {len(X_val)}")
    print(f"   -> 单个样本特征数: {X_train.shape[1]}")
    
    # 4. 初始化 XGBoost
    # 使用 MultiOutputRegressor 包装，使 XGBoost 能够同时预测 24 个未来的时间点
    xgb_estimator = xgb.XGBRegressor(
        n_estimators=100,      # 树的数量（调试期设100，正式提分可加到 300-500）
        max_depth=5,           # 树的深度（防止过拟合）
        learning_rate=0.1,     # 学习率
        objective='reg:squarederror', # 损失函数：均方误差
        n_jobs=-1,             # 调用你电脑的所有 CPU 核心满负荷运算
        random_state=42
    )
    
    model = MultiOutputRegressor(xgb_estimator)
    
    # 5. 训练模型 (记录时间)
    start_time = time.time()
    model.fit(X_train, Y_train)
    print(f"   -> 训练完成！耗时: {time.time() - start_time:.2f} 秒")
    
    # 6. 在验证集上评估模型
    print("4. 正在验证集上评估模型...")
    Y_pred = model.predict(X_val)
    
    # 计算评估指标
    # 因为出力是 0~1 的归一化数据，RMSE如果是 0.1 左右，说明平均误差在 10% 左右
    rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
    mae = mean_absolute_error(Y_val, Y_pred)
    
    print("="*40)
    print(f"🏆 风电场 {farm_id} 验证集表现:")
    print(f"   RMSE (均方根误差): {rmse:.4f}")
    print(f"   MAE  (平均绝对误差): {mae:.4f}")
    print("="*40)
    
    # 7. 保存模型
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/xgb_model_farm_{farm_id}.pkl'
    joblib.dump(model, model_path)
    print(f"💾 模型已保存至: {model_path}")
    
    return model

if __name__ == "__main__":
    file_path = 'wind_train_val_2012-01-02_to_2013-07-13.npy'
    
    # 1. 预处理数据
    processed_data = load_and_preprocess_data(file_path)
    
    # 2. 我们先针对风电场 0 (Farm 0) 进行训练测试
    target_farm_id = 0
    farm_data = processed_data[:, target_farm_id, :]
    
    # 构造 X 和 Y
    X, Y = build_dataset_for_farm(farm_data, window_size=24)
    
    # 训练评估保存
    train_xgboost_for_farm(target_farm_id, X, Y)