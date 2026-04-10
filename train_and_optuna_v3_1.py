# ==============================================================================
# 脚本名称: 06_optuna_tuning_v3_1.py
# 任务目标: 基于 V3_1 特征路由字典，全自动对 10 个风场进行 Optuna 贝叶斯寻优，
#           并自动保存最佳模型与超参数配置报告。
# ==============================================================================

import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import optuna
import os
import time
import joblib

# 全局变量，用于在每次 Optuna Trial 中传递当前风场的数据
current_X_train, current_Y_train = None, None
current_X_val, current_Y_val = None, None

def objective(trial):
    """ Optuna 的核心目标函数 (单次参数尝试) """
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
    rmse = np.sqrt(mean_squared_error(current_Y_val, Y_pred))
    
    return rmse

def main():
    version_tag = 'v3_1'
    data_path = 'data/features_v3.npz'
    dict_path = 'data/v3_1_routing_dict.npy'
    
    # 1. 基础检查与数据加载
    if not os.path.exists(data_path) or not os.path.exists(dict_path):
        print(f"❌ 找不到特征数据 {data_path} 或路由字典 {dict_path}！")
        return
        
    print(f"📦 正在加载全局 868 维特征数据与 {version_tag} 路由字典...")
    data = np.load(data_path)
    X_all, Y_all = data['X'], data['Y']
    routing_dict = np.load(dict_path)
    
    num_samples = X_all.shape[0]
    num_farms = 10
    split_idx = int(num_samples * 0.8)
    
    # 创建专属的保存目录
    save_dir = os.path.join('saved_models', version_tag)
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备一个总报告文本文档，记录所有风场的最优参数
    report_path = os.path.join(save_dir, f'optuna_params_report_{version_tag}.txt')
    
    global current_X_train, current_Y_train, current_X_val, current_Y_val
    
    print(f"\n🚀 开始 [V3_1 个性化路由] 全场全自动寻优流水线！")
    print(f"   (由于模型已瘦身至 150 维，配合 5090，速度将极大幅度提升)")
    
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
            
            # 时序划分并内存连续化 (防爆显存)
            current_X_train = np.ascontiguousarray(X_farm_lite[:split_idx])
            current_Y_train = np.ascontiguousarray(Y_farm[:split_idx])
            current_X_val   = np.ascontiguousarray(X_farm_lite[split_idx:])
            current_Y_val   = np.ascontiguousarray(Y_farm[split_idx:])
            
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
            print(f"   🏆 获得最低 RMSE: {study.best_value:.4f}")
            
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
            f_report.write(f"Best RMSE: {study.best_value:.4f}\n")
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