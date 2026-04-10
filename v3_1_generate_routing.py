# ==============================================================================
# 任务目标: 剖析 V3 模型，为 10 个风电场生成千人千面的 Top-150 特征路由字典
# ==============================================================================

import numpy as np
import joblib
import os

def generate_routing_dictionary(v3_model_dir='saved_models/v3', top_n=150, output_name='v3_1_routing_dict.npy'):
    num_farms = 10
    # 初始化一个 (10, 150) 的二维矩阵，用于存放每个风场的专属索引
    routing_indices = np.zeros((num_farms, top_n), dtype=int)
    
    print(f"🔍 正在深入解析 10 个风电场的 V3 全局模型，提取专属特征基因...")
    
    for f in range(num_farms):
        model_path = os.path.join(v3_model_dir, f'xgb_v3_farm_{f}.pkl')
        if not os.path.exists(model_path):
            print(f"❌ 找不到模型: {model_path}，请确认 V3 模型已全部训练完毕！")
            return
            
        multi_model = joblib.load(model_path)
        
        # 汇总该风场未来 24 个小时预测模型 (24 棵树集群) 的平均重要性
        num_features = multi_model.estimators_[0].feature_importances_.shape[0]
        avg_importance = np.zeros(num_features)
        for est in multi_model.estimators_:
            avg_importance += est.feature_importances_
            
        # 根据 Information Gain 从大到小排序，并截取 Top 150
        sorted_indices = np.argsort(avg_importance)[::-1]
        top_indices = sorted_indices[:top_n]
        
        # 存入字典对应的风场行
        routing_indices[f, :] = top_indices
        print(f"   ✅ Farm {f} 的 Top {top_n} 专属特征索引已提取。")

    # 保存这个极其珍贵的“路由字典”
    np.save(output_name, routing_indices)
    print(f"\n🎉 魔法路由字典已生成: {output_name} (大小仅几 KB！)")

if __name__ == "__main__":
    # 确保你的 saved_models/v3 下有 10 个跑好的 pkl 模型
    generate_routing_dictionary()