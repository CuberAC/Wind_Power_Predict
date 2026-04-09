import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import time
import argparse

def train_and_evaluate(farm_id, X_train, Y_train, X_val, Y_val, save_path):
    """
    为单个风电场训练模型并评估，保存至指定的 save_path
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始训练风电场 {farm_id} 的模型...")
    start_time = time.time()
    
    base_model = xgb.XGBRegressor(
        n_estimators: 1230
        max_depth: 9
        learning_rate: 0.010659217252788872
        subsample: 0.6523662083125326
        colsample_bytree: 0.7510208285575809
        min_child_weight: 5
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=42
    )
    model = MultiOutputRegressor(base_model)
    
    model.fit(X_train, Y_train)
    train_time = time.time() - start_time
    
    Y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
    mae = mean_absolute_error(Y_val, Y_pred)
    
    print(f"[{time.strftime('%H:%M:%S')}] 训练完成! 耗时: {train_time:.1f} 秒")
    print(f"   => [验证集表现] RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    
    # 直接使用传进来的智能路径保存
    joblib.dump(model, save_path)
    
    return rmse, mae

def main():
    parser = argparse.ArgumentParser(description="批量训练风电场 XGBoost 模型")
    parser.add_argument('--data', type=str, default='features_v1.npz', help='特征数据的路径')
    args = parser.parse_args()
    
    data_path = args.data
    
    # 智能提取版本号：提取 features_ 之后的全部内容
    file_basename = os.path.splitext(os.path.basename(data_path))[0]
    if file_basename.startswith('features_'):
        version_tag = file_basename[len('features_'):]
    else:
        version_tag = file_basename if file_basename else 'latest'

    if not os.path.exists(data_path):
        print(f"❌ 找不到文件 {data_path}，请检查路径是否正确！")
        return
        
    print(f"📦 正在加载特征数据: [{data_path}] ...")
    data = np.load(data_path)
    keys = data.files
    X_all = data[keys[0]]  
    Y_all = data[keys[1]]  
    
    num_samples = X_all.shape[0]
    num_farms = X_all.shape[1]
    
    print(f"✅ 数据加载成功！风场数: {num_farms} | 样本数: {num_samples} | 特征维度: {X_all.shape[2]}")
    
    split_ratio = 0.8
    split_idx = int(num_samples * split_ratio)
    
    # 智能创建专属保存目录
    save_dir = os.path.join('saved_models', version_tag)
    os.makedirs(save_dir, exist_ok=True)
    
    total_rmse = 0.0
    total_mae = 0.0
    
    print(f"\n🚀 开始批量训练流程 (当前版本标签: {version_tag})...")
    for farm_id in range(num_farms):
        X_train, Y_train = X_all[:split_idx, farm_id, :], Y_all[:split_idx, farm_id, :]
        X_val, Y_val     = X_all[split_idx:, farm_id, :], Y_all[split_idx:, farm_id, :]
        
        # 智能生成当前模型的保存路径
        model_path = os.path.join(save_dir, f'xgb_{version_tag}_farm_{farm_id}.pkl')
        
        rmse, mae = train_and_evaluate(farm_id, X_train, Y_train, X_val, Y_val, model_path)
        
        total_rmse += rmse
        total_mae += mae
        
    print("\n" + "="*40)
    print("🎉 所有 10 个风电场模型训练完毕！")
    print(f"   【{version_tag} 全局平均 RMSE】: {total_rmse / num_farms:.4f}")
    print(f"   【{version_tag} 全局平均 MAE 】: {total_mae / num_farms:.4f}")
    print(f"   💾 模型已全部分类保存在: {save_dir}/ 目录下")
    print("="*40)

if __name__ == "__main__":
    main()