# ==============================================================================
# 任务目标: 兼容高维特征的 XGBoost 特征重要性分析与清洗指导
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import re

def generate_v3_feature_names(window_size=24, num_farms=10):
    """
    智能动态生成 V3 版本的 868 维特征名称列表。
    V3 结构: 240(全局出力) + 4(本场统计) + 312(过去气象) + 312(未来气象) = 868
    """
    feature_names = []
    
    # 1. 跨空间融合：全局 10 个风场过去 24 小时的出力
    for i in range(window_size, 0, -1):
        for f in range(num_farms):
            feature_names.append(f"Past_{i}h_Farm{f}_Power")
            
    # 2. 本场历史统计量
    feature_names.extend([
        "Local_Past24h_Mean", "Local_Past24h_Std", 
        "Local_Past24h_Max", "Local_Past24h_Min"
    ])
    
    weather_vars = ["U10", "V10", "U100", "V100", "WS10", "WS100", 
                    "Cube_Energy", "WDir_Sin", "WDir_Cos", "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos"]
    
    # 3. 本场过去 24 小时气象
    for i in range(window_size, 0, -1):
        for var in weather_vars:
            feature_names.append(f"Past_{i}h_{var}")
            
    # 4. 本场未来 24 小时预报气象
    for i in range(1, window_size + 1):
        for var in weather_vars:
            feature_names.append(f"Future_{i}h_{var}")
            
    return feature_names

def analyze_and_plot():
    # 我们读取 V3 跑出来的 Farm 0 模型
    model_path = 'saved_models/v3/xgb_v3_farm_0.pkl'  
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型：{model_path}")
        return
        
    print("🔍 加载模型中...")
    multi_model = joblib.load(model_path)
    
    # 动态生成名字
    feature_names = generate_v3_feature_names()
    num_features = len(feature_names)
    print(f"✅ 成功映射 {num_features} 个特征标签！")
    
    # 计算 24 个未来预测模型的平均特征重要性 (基于 Gain)
    avg_importance = np.zeros(num_features)
    for estimator in multi_model.estimators_:
        avg_importance += estimator.feature_importances_
    avg_importance /= len(multi_model.estimators_)
    
    # ================= 宏观分类特征贡献占比 (极其适合写进报告) =================
    category_scores = {
        "1. 未来气象预报 (核心动力)": 0.0,
        "2. 跨风场空间协同 (上游信息)": 0.0,
        "3. 本场历史气象 (环境基准)": 0.0,
        "4. 本场历史状态 (惯性与健康度)": 0.0
    }
    
    for name, score in zip(feature_names, avg_importance):
        if "Future_" in name:
            category_scores["1. 未来气象预报 (核心动力)"] += score
        elif "Farm" in name and "Power" in name: # 匹配跨风场特征
            # 区分是不是本场自己的历史出力 (假设当前分析的是 Farm 0)
            if "Farm0" in name:
                category_scores["4. 本场历史状态 (惯性与健康度)"] += score
            else:
                category_scores["2. 跨风场空间协同 (上游信息)"] += score
        elif "Past_" in name and "Power" not in name and "Local" not in name:
            category_scores["3. 本场历史气象 (环境基准)"] += score
        else: # 包含了本场统计特征
            category_scores["4. 本场历史状态 (惯性与健康度)"] += score

    print("\n📊 宏观特征板块贡献占比：")
    for cat, score in category_scores.items():
        print(f"   {cat}: {score*100:.2f}%")
        
    # ================= 细节 Top 30 核心驱动因素 =================
    feature_scores = list(zip(feature_names, avg_importance))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_n = 30
    print(f"\n🏆 微观最强 Top {top_n} 特征：")
    for rank, (name, score) in enumerate(feature_scores[:top_n], 1):
        print(f"   第{rank:2d}名 | 贡献: {score:.4f} | {name}")
        
    # 画图
    names = [x[0] for x in feature_scores[:top_n]]
    scores = [x[1] for x in feature_scores[:top_n]]
    
    plt.figure(figsize=(12, 10))
    plt.barh(names[::-1], scores[::-1], color='coral', edgecolor='black')
    plt.xlabel('Mean Information Gain', fontsize=12)
    plt.title('XGBoost - Top 30', fontsize=15, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('V3_Feature_Importance.png', dpi=300)
    print("\n📸 图表已保存为 'V3_Feature_Importance.png'！")
    plt.show()

if __name__ == "__main__":
    analyze_and_plot()