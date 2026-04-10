import numpy as np
import joblib
import os

def select_top_features(model_path, data_path, top_n=150):
    print(f"1. 正在分析模型特征重要性: {model_path}")
    # 加载 V3 训练好的 Farm 0 模型
    multi_model = joblib.load(model_path)
    
    # 汇总 24 个单步模型的平均重要性
    num_features = multi_model.estimators_[0].feature_importances_.shape[0]
    avg_importance = np.zeros(num_features)
    for est in multi_model.estimators_:
        avg_importance += est.feature_importances_
    
    # 获取排序后的索引（从大到小）
    sorted_indices = np.argsort(avg_importance)[::-1]
    top_indices = sorted_indices[:top_n]
    
    print(f"2. 已选出贡献最大的 Top {top_n} 个特征。")
    
    # 加载原始 V3 数据
    print(f"3. 正在加载原始 V3 数据进行切分: {data_path}")
    data = np.load(data_path)
    X_all = data['X'] # (样本数, 10, 868)
    Y_all = data['Y'] # (样本数, 10, 24)
    
    # 只保留这 top_n 个特征维度
    # 注意：X_all 是三维的，我们要切的是最后一维
    X_lite = X_all[:, :, top_indices]
    
    print(f"4. 瘦身成功！特征维度从 {num_features} 降至 {X_lite.shape[2]}")
    
    # 保存为 Lite 版本
    output_path = 'data/features_v3_lite.npz'
    np.savez_compressed(output_path, X=X_lite, Y=Y_all)
    
    # 🌟 重要：把这 top_indices 存下来！
    # 因为以后在测试集上预测时，你也必须按同样的顺序切分特征！
    np.save('data/top_indices_v3.npy', top_indices)
    
    print(f"✅ 最终数据已保存至: {output_path}")
    print(f"✅ 特征索引已保存至: data/top_indices_v3.npy (推理时必用)")

if __name__ == "__main__":
    select_top_features('saved_models/v3/xgb_v3_farm_0.pkl', 'data/features_v3.npz', top_n=150)
    # 💡 注意：如果你已经跑完了 V3，就把上面路径换成 V3 的。