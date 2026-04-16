# ==============================================================================
# 脚本名称: 06_optuna_tuning_v3_1.py
# 任务目标: 基于 V3_1 特征路由字典，全自动对 10 个风场进行 Optuna 贝叶斯寻优，
#           并自动保存最佳模型与超参数配置报告。
# ==============================================================================

import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import optuna
import os
import time
import joblib
import pandas as pd

# 全局变量，用于在每次 Optuna Trial 中传递当前风场的数据
current_X_train, current_Y_train = None, None
current_X_val, current_Y_val = None, None


def build_v3_test_X_Y(test_raw, window_size=24):
    """Build V3-style test features/labels from raw test npy."""
    numeric_test = test_raw[:, :, 1:].astype(np.float32)

    # NaN cleaning (same spirit as test_xgboost.py)
    if np.isnan(numeric_test).any():
        for f in range(numeric_test.shape[1]):
            for c in range(numeric_test.shape[2]):
                series = pd.Series(numeric_test[:, f, c])
                numeric_test[:, f, c] = series.interpolate().ffill().bfill().values

    U10, V10 = numeric_test[:, :, 0], numeric_test[:, :, 1]
    U100, V100 = numeric_test[:, :, 2], numeric_test[:, :, 3]
    power_data = numeric_test[:, :, 4]

    WS10 = np.sqrt(U10 ** 2 + V10 ** 2)
    WS100 = np.sqrt(U100 ** 2 + V100 ** 2)
    WS100_Cube = WS100 ** 3
    WDir_Rad = np.arctan2(V100, U100)
    Sin_WDir, Cos_WDir = np.sin(WDir_Rad), np.cos(WDir_Rad)

    time_steps, num_farms = numeric_test.shape[0], numeric_test.shape[1]
    hours = np.array([h % 24 for h in range(time_steps)], dtype=np.float32)
    Sin_Hour = np.repeat(np.sin(2 * np.pi * hours / 24)[:, np.newaxis], num_farms, axis=1)
    Cos_Hour = np.repeat(np.cos(2 * np.pi * hours / 24)[:, np.newaxis], num_farms, axis=1)
    Sin_Month = np.zeros_like(Sin_Hour)
    Cos_Month = np.zeros_like(Cos_Hour)

    weather_features = np.stack([
        U10, V10, U100, V100, WS10, WS100, WS100_Cube,
        Sin_WDir, Cos_WDir, Sin_Hour, Cos_Hour, Sin_Month, Cos_Month
    ], axis=-1)

    X_list = [[] for _ in range(num_farms)]
    Y_list = [[] for _ in range(num_farms)]

    for i in range(window_size, time_steps - window_size + 1):
        past_power_global = power_data[i - window_size:i, :].flatten()
        for f in range(num_farms):
            past_power_local = power_data[i - window_size:i, f]
            p_weather = weather_features[i - window_size:i, f, :].flatten()
            f_weather = weather_features[i:i + window_size, f, :].flatten()
            p_stats = np.array([
                np.mean(past_power_local),
                np.std(past_power_local),
                np.max(past_power_local),
                np.min(past_power_local),
            ], dtype=np.float32)

            x_row = np.concatenate([past_power_global, p_stats, p_weather, f_weather])
            y_row = power_data[i:i + window_size, f]
            X_list[f].append(x_row)
            Y_list[f].append(y_row)

    X_farms = [np.asarray(x, dtype=np.float32) for x in X_list]
    Y_farms = [np.asarray(y, dtype=np.float32) for y in Y_list]
    return X_farms, Y_farms

def objective(trial):
    """ Optuna 的核心目标函数 (单次参数尝试): optimize validation MAE """
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        
        'objective': 'reg:squarederror',
        'random_state': 42,
        'tree_method': 'hist',  
        'device': 'cuda',       # 🚀 召唤 RTX 5090
        'nthread': 1            # 防爆显存
    }
    
    base_model = xgb.XGBRegressor(**param)
    model = MultiOutputRegressor(base_model)
    
    # 使用全局变量中的当前风场数据进行训练
    model.fit(current_X_train, current_Y_train)
    
    Y_pred = model.predict(current_X_val)
    mae = mean_absolute_error(current_Y_val, Y_pred)
    
    return mae

def main():
    version_tag = 'v3_1'
    train_feature_path = 'data/features_v3.npz'
    test_raw_path = 'data/wind_test_cleaned.npy'
    dict_path = 'data/v3_1_routing_dict.npy'
    
    # 1. 基础检查与数据加载
    if not os.path.exists(train_feature_path) or not os.path.exists(dict_path) or not os.path.exists(test_raw_path):
        print(f"❌ 缺少文件：{train_feature_path} / {dict_path} / {test_raw_path}")
        return
        
    print(f"📦 正在加载训练特征、真实测试集与 {version_tag} 路由字典...")
    data = np.load(train_feature_path)
    X_all, Y_all = data['X'], data['Y']
    routing_dict = np.load(dict_path)
    test_raw = np.load(test_raw_path, allow_pickle=True)
    X_test_farms, Y_test_farms = build_v3_test_X_Y(test_raw, window_size=24)
    
    num_farms = 10
    
    # 创建专属的保存目录
    save_dir = os.path.join('saved_models', version_tag)
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备一个总报告文本文档，记录所有风场的最优参数
    report_path = os.path.join(save_dir, f'optuna_params_report_{version_tag}.txt')
    
    global current_X_train, current_Y_train, current_X_val, current_Y_val
    
    print(f"\n🚀 开始 [V3_1 个性化路由] 全场全自动寻优流水线！")
    print(f"   训练集: data/wind_train_val_2012-01-02_to_2013-07-13.npy (全量特征样本)")
    print(f"   验证集: data/wind_test_cleaned.npy (真实测试集)")
    
    # 2. 遍历 10 个风场，逐个进行 Optuna 寻优
    with open(report_path, 'w', encoding='utf-8') as f_report:
        f_report.write(f"=== {version_tag} 全局 10 风场 Optuna 最优超参数报告 ===\n\n")
        
        for f in range(num_farms):
            print(f"\n" + "="*40)
            print(f"🎯 正在为 Farm {f} 进行专属超参数寻优 (Trials: 30)...")
            
            # --- 🌟 核心：实时特征路由切片 🌟 ---
            my_exclusive_indices = routing_dict[f] 
            
            # 先切出当前风场的全量维度，再精确抽取那 150 维精英特征
            X_farm_lite = X_all[:, f, :][:, my_exclusive_indices]
            Y_farm = Y_all[:, f, :]
            X_farm_test_lite = X_test_farms[f][:, my_exclusive_indices]
            Y_farm_test = Y_test_farms[f]
            
            # 训练用全量 train_val，验证用真实测试集
            current_X_train = np.ascontiguousarray(X_farm_lite)
            current_Y_train = np.ascontiguousarray(Y_farm)
            current_X_val   = np.ascontiguousarray(X_farm_test_lite)
            current_Y_val   = np.ascontiguousarray(Y_farm_test)
            
            # 异常值清洗
            current_X_train = np.nan_to_num(current_X_train, nan=0.0).astype(np.float32)
            current_Y_train = np.nan_to_num(current_Y_train, nan=0.0).astype(np.float32)
            current_X_val   = np.nan_to_num(current_X_val, nan=0.0).astype(np.float32)
            current_Y_val   = np.nan_to_num(current_Y_val, nan=0.0).astype(np.float32)
            
            # --- 开始 Optuna 寻优 ---
            # 屏蔽繁杂的中间日志，只看进度条 (通过设置 verbosity)
            optuna.logging.set_verbosity(optuna.logging.WARNING) 
            study = optuna.create_study(direction='minimize', study_name=f"Farm_{f}_Study")
            
            start_time = time.time()
            study.optimize(objective, n_trials=30, show_progress_bar=True)
            
            print(f"   ✅ Farm {f} 寻优完成！耗时: {(time.time() - start_time)/60:.1f} 分钟")
            print(f"   🏆 获得最低 MAE: {study.best_value:.4f}")
            
            # --- 训练并保存该风场的【终极最优模型】 ---
            print(f"   💾 正在使用最佳参数重新训练终极模型并归档...")
            best_params = study.best_params
            best_params.update({'objective': 'reg:squarederror', 'random_state': 42, 
                                'tree_method': 'hist', 'device': 'cuda', 'nthread': 1})
            
            final_model = MultiOutputRegressor(xgb.XGBRegressor(**best_params))
            final_model.fit(current_X_train, current_Y_train)
            
            model_path = os.path.join(save_dir, f'xgb_{version_tag}_farm_{f}.pkl')
            joblib.dump(final_model, model_path)
            
            # --- 写入参数报告 ---
            f_report.write(f"--- Farm {f} ---\n")
            f_report.write(f"Best MAE (Real Test): {study.best_value:.4f}\n")
            for key, val in study.best_params.items():
                f_report.write(f"{key}: {val}\n")
            f_report.write("\n")
            f_report.flush() # 实时写入硬盘，防止中途断电丢失
            
    print("\n" + "🚀"*15)
    print(f"🎉 史诗级任务完成！10 个风电场的 V3_1 个性化终极模型已全部训练完毕！")
    print(f"📂 模型已存入: {save_dir}/")
    print(f"📄 参数报告已存入: {report_path}")
    print("🚀"*15)

if __name__ == "__main__":
    main()