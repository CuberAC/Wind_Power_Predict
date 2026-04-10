# ==============================================================================
# 任务目标: 使用 Optuna (贝叶斯优化) 为 XGBoost 寻找最优超参数
# ==============================================================================

import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import optuna
import os
import time

# 定义全局变量存放数据，避免在每次试验中重复加载
X_train, Y_train = None, None
X_val, Y_val = None, None

def load_data_for_tuning(data_path='data/features_v2.npz', target_farm_id=0):
    """
    加载数据，并严格划分 8-2 验证集，仅提取目标风电场的数据用于调参。
    """
    global X_train, Y_train, X_val, Y_val
    
    print(f"📦 正在加载特征数据: [{data_path}] ...")
    data = np.load(data_path)
    X_all = data[data.files[0]]
    Y_all = data[data.files[1]]
    
    num_samples = X_all.shape[0]
    split_idx = int(num_samples * 0.8)
    
    # 我们只取某一个风场（比如 Farm 0）来进行调参寻优
    X_farm = X_all[:, target_farm_id, :]
    Y_farm = Y_all[:, target_farm_id, :]
    
    X_train, Y_train = X_farm[:split_idx], Y_farm[:split_idx]
    X_val, Y_val     = X_farm[split_idx:], Y_farm[split_idx:]
     
    # ========== 🚀 核心修复：防止 GPU 显存崩溃 ==========
    X_train = np.ascontiguousarray(X_train)
    Y_train = np.ascontiguousarray(Y_train)
    X_val = np.ascontiguousarray(X_val)
    Y_val = np.ascontiguousarray(Y_val)
    # ====================================================
    # 将所有的 NaN 或 Inf 替换为 0 (或者均值)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    Y_train = np.nan_to_num(Y_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    Y_val = np.nan_to_num(Y_val, nan=0.0, posinf=0.0, neginf=0.0)
    # ==========================================
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    Y_val = Y_val.astype(np.float32)

    print(f"✅ 数据加载完毕。正在针对风电场 {target_farm_id} 开启参数寻优！")

def objective(trial):
    """
    完全抛弃 MultiOutputRegressor，手写干净的 GPU 循环
    """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        
        'objective': 'reg:squarederror',
        'random_state': 42,
        'tree_method': 'hist',  # 🚀 GPU 加速
        'device': 'cuda',       # 🚀 指定 GPU
        'nthread': 1            # 安全第一，单线程向显卡提交任务
    }
    
    # 预测未来 24 个点
    num_future_steps = Y_train.shape[1] 
    
    # 准备一个空数组，用来存放未来 24 个点的预测结果
    Y_pred_all = np.zeros_like(Y_val, dtype=np.float32)
    
    # ================= 🌟 核心重构：自己写循环 =================
    # 依次训练未来第 0 小时、第 1 小时 ... 第 23 小时
    for step in range(num_future_steps):
        # 取出我们要预测的具体哪一个小时作为当前的 y
        y_train_step = Y_train[:, step]
        y_val_step = Y_val[:, step]
        
        # 实例化原生 XGBoost 模型（没有任何 sklearn 包装）
        model = xgb.XGBRegressor(**param)
        
        # 训练这个小时的模型
        model.fit(X_train, y_train_step)
        
        # 预测并将结果填入大数组中
        Y_pred_all[:, step] = model.predict(X_val)
    # ==========================================================
        
    # 所有 24 个模型全跑完了，计算总的 RMSE
    rmse = np.sqrt(mean_squared_error(Y_val, Y_pred_all))
    
    return rmse

def main():
    # 1. 准备数据 (这里默认用 data/features_v2.npz，用 Farm 0 试水)
    load_data_for_tuning('data/features_v2.npz', target_farm_id=0)
    
    # 2. 创建一场研究 (Study)
    # direction='minimize' 代表我们的目标是让返回的误差(RMSE)越小越好
    study = optuna.create_study(direction='minimize', study_name="XGBoost_Wind_Power")
    
    # 3. 启动优化器！
    # n_trials 代表我们要让算法尝试多少种组合。
    # 对于 XGBoost，通常 30~50 次就能找到很不错的近似最优解。
    print(f"\n🚀 开始贝叶斯自动调参，预计尝试 30 种参数组合...")
    start_time = time.time()
    
    study.optimize(objective, n_trials=30)
    
    print(f"\n[{time.strftime('%H:%M:%S')}] 🎉 寻优完成！总耗时: {(time.time() - start_time)/60:.1f} 分钟")
    
    # 4. 打印最终的最优结果
    print("="*40)
    print("🏆 发现的最佳参数组合:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    print(f"\n🥇 该参数下得到的最低 RMSE: {study.best_value:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()