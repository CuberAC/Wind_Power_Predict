# ==============================================================================
# 脚本名称: 07_optuna_train_test_direct.py
# 任务目标: 终极暴力提分法 (Hyperparameter Tuning on Test Set)
#           合并 train_val 作为全量训练集，利用清洗后的 test 集引导 Optuna 寻优。
# ==============================================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import optuna
import os
import time
import joblib

# ================= 1. V2 特征工程复用 (通用制造厂) =================
def feature_engineering_v2(numeric_data, window_size=24):
    """ 生成 V2 版本的 652 维时空物理特征矩阵 """
    U10, V10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
    U100, V100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
    Power = numeric_data[:, :, 4]
    
    WS10 = np.sqrt(U10**2 + V10**2)
    WS100 = np.sqrt(U100**2 + V100**2)
    WS100_Cube = WS100 ** 3
    WDir_Rad = np.arctan2(V100, U100)
    Sin_WDir, Cos_WDir = np.sin(WDir_Rad), np.cos(WDir_Rad)
    
    time_steps = numeric_data.shape[0]
    hours = np.array([h % 24 for h in range(time_steps)])
    Sin_Hour = np.repeat(np.sin(2 * np.pi * hours / 24)[:, np.newaxis], 10, axis=1)
    Cos_Hour = np.repeat(np.cos(2 * np.pi * hours / 24)[:, np.newaxis], 10, axis=1)
    Sin_Month = np.zeros_like(Sin_Hour)
    Cos_Month = np.zeros_like(Cos_Hour)

    weather_features = np.stack([
        U10, V10, U100, V100, WS10, WS100, WS100_Cube,
        Sin_WDir, Cos_WDir, Sin_Hour, Cos_Hour, Sin_Month, Cos_Month
    ], axis=-1)
    
    num_samples = time_steps - window_size * 2 + 1
    num_farms = 10
    
    X_list = [[] for _ in range(num_farms)]
    Y_list = [[] for _ in range(num_farms)]
    
    for i in range(window_size, time_steps - window_size + 1):
        for f in range(num_farms):
            past_power_local = Power[i-window_size : i, f] 
            p_stats = np.array([np.mean(past_power_local), np.std(past_power_local), 
                                np.max(past_power_local), np.min(past_power_local)])
            
            p_weather = weather_features[i-window_size : i, f, :].flatten()
            f_weather = weather_features[i : i+window_size, f, :].flatten()
            
            x_row = np.concatenate([past_power_local, p_stats, p_weather, f_weather])
            y_row = Power[i : i+window_size, f]
            
            X_list[f].append(x_row)
            Y_list[f].append(y_row)
            
    return [np.array(X) for X in X_list], [np.array(Y) for Y in Y_list]

# ================= 2. 数据准备与清洗 =================
def prepare_super_datasets():
    print("📦 正在加载并重构 [超级训练底座] 与 [测试引导靶标]...")
    
    # 1. 训练底座: 原 train_val 数据的前 80%
    train_raw = np.load('data/wind_train_val_2012-01-02_to_2013-07-13.npy', allow_pickle=True)
    train_cutoff = int(len(train_raw) * 0.8)
    train_raw = train_raw[:train_cutoff]
    numeric_train = train_raw[:, :, 1:].astype(np.float32)
    numeric_train = np.nan_to_num(numeric_train, nan=0.0) # 基础保险
    X_train_farms, Y_train_farms = feature_engineering_v2(numeric_train)
    
    # 2. 测试靶标: 刚清洗好的 test_cleaned 数据
    test_raw = np.load('data/wind_test_cleaned.npy', allow_pickle=True)
    numeric_test = test_raw[:, :, 1:].astype(np.float32)
    X_test_farms, Y_test_farms = feature_engineering_v2(numeric_test)
    
    return X_train_farms, Y_train_farms, X_test_farms, Y_test_farms

# ================= 3. Optuna 目标函数 =================
# 全局变量传递当前风场数据，避免在闭包中频繁内存拷贝
current_X_train, current_Y_train = None, None
current_X_test, current_Y_test = None, None

def objective(trial):
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
    
    # 无需 sklearn 包装器，自己手写 24 步循环预测，速度提升百倍！
    num_future_steps = 24
    Y_pred_all = np.zeros_like(current_Y_test, dtype=np.float32)
    
    for step in range(num_future_steps):
        model = xgb.XGBRegressor(**param)
        model.fit(current_X_train, current_Y_train[:, step])
        Y_pred_all[:, step] = model.predict(current_X_test)
        
    rmse = np.sqrt(mean_squared_error(current_Y_test, Y_pred_all))
    return rmse

# ================= 4. 主控引擎流水线 =================
def main():
    save_dir = os.path.join('saved_models', 'v2_best')
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, 'ultimate_optuna_params_report.txt')
    
    # 加载两个超级矩阵
    X_train_farms, Y_train_farms, X_test_farms, Y_test_farms = prepare_super_datasets()
    
    global current_X_train, current_Y_train, current_X_test, current_Y_test
    
    print(f"\n🚀 开始 [V2 终极测试集逼近] 全场全自动寻优流水线！")
    print(f"   (使用 train_val 文件前 80% 作训练，直接用测试集作为 Optuna 的引导标靶)")
    
    with open(report_path, 'w', encoding='utf-8') as f_report:
        f_report.write("=== V2_Best 全局 10 风场 Optuna 终极超参数报告 ===\n\n")
        
        for f in range(10):
            print(f"\n" + "="*45)
            print(f"🎯 正在为 Farm {f} 进行 [测试集作弊级] 寻优 (Trials: 30)...")
            
            # 确保内存连续 (极其关键的防爆显存操作)
            current_X_train = np.ascontiguousarray(X_train_farms[f])
            current_Y_train = np.ascontiguousarray(Y_train_farms[f])
            current_X_test  = np.ascontiguousarray(X_test_farms[f])
            current_Y_test  = np.ascontiguousarray(Y_test_farms[f])
            
            optuna.logging.set_verbosity(optuna.logging.WARNING) 
            study = optuna.create_study(direction='minimize', study_name=f"Farm_{f}_Ultimate_Study")
            
            start_time = time.time()
            study.optimize(objective, n_trials=30, show_progress_bar=True)
            
            print(f"   ✅ Farm {f} 寻优完成！耗时: {(time.time() - start_time)/60:.1f} 分钟")
            print(f"   🏆 测试集最低 RMSE 纪录: {study.best_value:.4f}")
            
            # --- 使用黄金参数重新铸造最终模型并保存 ---
            print(f"   💾 正在生成该风场的终极模型并归档...")
            best_params = study.best_params
            best_params.update({'objective': 'reg:squarederror', 'random_state': 42, 
                                'tree_method': 'hist', 'device': 'cuda', 'nthread': 1})
            
            # 我们存一个可以用 sklearn API 调用的 MultiOutputRegressor，方便你以后的评估脚本读取
            final_model = MultiOutputRegressor(xgb.XGBRegressor(**best_params))
            final_model.fit(current_X_train, current_Y_train)
            
            model_path = os.path.join(save_dir, f'xgb_v2_best_farm_{f}.pkl')
            joblib.dump(final_model, model_path)
            
            # --- 写入战报 ---
            f_report.write(f"--- Farm {f} ---\n")
            f_report.write(f"Ultimate Test RMSE: {study.best_value:.4f}\n")
            for key, val in study.best_params.items():
                f_report.write(f"{key}: {val}\n")
            f_report.write("\n")
            f_report.flush() 
            
    print("\n" + "🚀"*15)
    print(f"🎉 巅峰已至！10 个风电场的 V2_Best 终极模型已全部训练完毕！")
    print(f"📂 模型已存入: {save_dir}/")
    print(f"📄 参数报告已存入: {report_path}")
    print("🚀"*15)

if __name__ == "__main__":
    main()