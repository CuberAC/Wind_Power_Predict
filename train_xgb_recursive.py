# ==============================================================================
# 脚本名称: 03_train_recursive_model.py
# 任务目标: 使用最优超参数，基于 V2 特征，执行硬核的【自回归滚动预测】
# 核心创新: 抛弃 MultiOutputRegressor，单步建模，滚动更新历史出力与统计特征
# ==============================================================================

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import time
import argparse

def train_and_evaluate_recursive(farm_id, X_train, Y_train, X_val, Y_val, save_path):
    """
    为单个风电场训练单步模型，并执行 24 步自回归滚动预测
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] 🚀 开启风电场 {farm_id} 的滚动预测训练...")
    start_time = time.time()
    
    # ================= 1. 训练阶段 (只看未来第 1 个小时) =================
    # Y_train 的形状是 (样本数, 24)，我们只取第 0 列 (紧接着的下一小时)
    y_train_step1 = Y_train[:, 0]
    
    # 实例化原生 XGBoost (这里填入 Optuna 跑出的最优参数)
    model = xgb.XGBRegressor(
        n_estimators=598,             
        max_depth=7,                  
        learning_rate=0.013698,          
        subsample=0.7186,                
        colsample_bytree=0.9994,        
        min_child_weight=2,           
        objective='reg:squarederror', 
        random_state=42,
        tree_method='hist',     
        device='cuda',
        nthread=1               
    )
    
    print("   -> 正在训练单步引擎 (One-Step-Ahead Model)...")
    model.fit(X_train, y_train_step1)
    
    train_time = time.time() - start_time
    print(f"   -> 训练完成！耗时极短: {train_time:.1f} 秒")
    
    # ================= 2. 预测阶段 (自回归滚动 24 步) =================
    print("   -> 正在执行 24 步时序推演 (Autoregressive Loop)...")
    
    num_samples = X_val.shape[0]
    window_size = 24  # 历史出力序列的长度
    
    # 准备一个空矩阵，存放滚出来的未来 24 个点的预测曲线
    Y_pred_recursive = np.zeros_like(Y_val, dtype=np.float32)
    
    # 深拷贝一份测试集特征起点，防止污染原始数据
    current_X = np.copy(X_val)
    
    # 开始 24 次循环推演
    for step in range(24):
        # 1. 拿当前的特征矩阵去预测下一个点 (第 step 个小时)
        pred_next_hour = model.predict(current_X)
        
        # 2. 将预测结果落袋为安
        Y_pred_recursive[:, step] = pred_next_hour
        
        # 3. 🌟 核心魔法：特征大挪移 (滚动更新历史数据) 🌟
        if step < 23:  # 最后一步预测完就不需要再更新特征了
            # A. 更新【过去 24 小时出力序列】(对应特征索引 0 到 23)
            # 把最老的时刻 (第 0 列) 挤掉，整体数据左移一格
            current_X[:, 0:23] = current_X[:, 1:24]
            # 把刚刚出炉的预测结果，填补到最新的时刻 (第 23 列)
            current_X[:, 23] = pred_next_hour
            
            # B. 同步更新【4 个历史统计特征】(对应特征索引 24 到 27)
            # 因为序列变了，均值、标准差、最大、最小值也必须重新计算！
            updated_past_power = current_X[:, 0:24]
            current_X[:, 24] = np.mean(updated_past_power, axis=1) # 均值
            current_X[:, 25] = np.std(updated_past_power, axis=1)  # 标准差
            current_X[:, 26] = np.max(updated_past_power, axis=1)  # 最大值
            current_X[:, 27] = np.min(updated_past_power, axis=1)  # 最小值
            
            # (注意：气象特征我们不需要滚动更新，因为不同 step 预测时，
            # 气象预报本来就是对未来的静态描述，只有出力序列是动态生长的)

    # ================= 3. 评估与保存 =================
    # 用最终滚出来的 24 步完整曲线，和真实的 Y_val 对比算总误差
    rmse = np.sqrt(mean_squared_error(Y_val, Y_pred_recursive))
    mae = mean_absolute_error(Y_val, Y_pred_recursive)
    
    print(f"   => [滚动预测验证] RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    
    joblib.dump(model, save_path)
    
    return rmse, mae

def main():
    parser = argparse.ArgumentParser(description="批量训练风电场 滚动预测模型")
    parser.add_argument('--data', type=str, default='features_v2.npz', help='特征数据的路径')
    args = parser.parse_args()
    
    data_path = args.data
    file_basename = os.path.basename(data_path).replace('.npz', '')
    version_tag = file_basename.split('_')[-1] if '_' in file_basename else 'latest'
    version_tag = f"{version_tag}_recursive" # 给保存的文件夹加上滚动预测的专属标签

    if not os.path.exists(data_path):
        print(f"❌ 找不到文件 {data_path}")
        return
        
    print(f"📦 正在加载特征数据: [{data_path}] ...")
    data = np.load(data_path)
    keys = data.files
    X_all = data[keys[0]]  
    Y_all = data[keys[1]]  
    
    num_samples = X_all.shape[0]
    num_farms = X_all.shape[1]
    
    # 严格时序划分 8-2
    split_idx = int(num_samples * 0.8)
    
    save_dir = os.path.join('saved_models', version_tag)
    os.makedirs(save_dir, exist_ok=True)
    
    total_rmse = 0.0
    total_mae = 0.0
    
    print(f"\n🌀 开启全局滚动预测训练 (当前版本: {version_tag}) ...")
    for farm_id in range(num_farms):
        X_train = X_all[:split_idx, farm_id, :]
        Y_train = Y_all[:split_idx, farm_id, :]
        X_val   = X_all[split_idx:, farm_id, :]
        Y_val   = Y_all[split_idx:, farm_id, :]
        
        # 强制内存连续与防爆显存清洗
        X_train = np.ascontiguousarray(X_train)
        Y_train = np.ascontiguousarray(Y_train)
        X_val = np.ascontiguousarray(X_val)
        Y_val = np.ascontiguousarray(Y_val)
        
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        Y_train = np.nan_to_num(Y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        Y_val = np.nan_to_num(Y_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        model_path = os.path.join(save_dir, f'xgb_{version_tag}_farm_{farm_id}.pkl')
        
        rmse, mae = train_and_evaluate_recursive(farm_id, X_train, Y_train, X_val, Y_val, model_path)
        
        total_rmse += rmse
        total_mae += mae
        
    print("\n" + "="*45)
    print("🎉 所有 10 个风电场的滚动预测模型训练完毕！")
    print(f"   【{version_tag} 全局平均 RMSE】: {total_rmse / num_farms:.4f}")
    print(f"   【{version_tag} 全局平均 MAE 】: {total_mae / num_farms:.4f}")
    print(f"   💾 模型已保存在: {save_dir}/ 目录下")
    print("="*45)

if __name__ == "__main__":
    main()