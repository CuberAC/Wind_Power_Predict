# ==============================================================================
# 脚本名称: test_xgboost.py
# 任务目标: 终极评测脚本，兼容 v1, v2, v3, v3_lite, v3_1 所有 XGBoost 模型
# 使用方法: python test_xgboost.py --version v3_1
# ==============================================================================

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

def feature_engineering_factory(numeric_data, version):
    """
    根据版本号，对原始数值数据进行特征重构
    numeric_data: (Time, Farm, 5) -> [U10, V10, U100, V100, Power]
    """
    U10, V10 = numeric_data[:, :, 0], numeric_data[:, :, 1]
    U100, V100 = numeric_data[:, :, 2], numeric_data[:, :, 3]
    Power = numeric_data[:, :, 4]
    
    # 基础物理特征 
    WS10 = np.sqrt(U10**2 + V10**2)
    WS100 = np.sqrt(U100**2 + V100**2)

    # 🌟 针对 V1：只有最基础的 6 个气象特征
    if version == 'v1':
        weather_features = np.stack([U10, V10, U100, V100, WS10, WS100], axis=-1)
        return weather_features, Power
    
    # 🌟 针对 V2, V3 及以上：加入高级物理与时间特征 (13维)
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
    
    return weather_features, Power

def build_test_X_Y(weather_features, power_data, version, window_size=24):
    """
    根据版本逻辑构建基础的滑动窗口测试集 X 和 Y
    """
    num_time_steps = weather_features.shape[0]
    num_samples = num_time_steps - window_size * 2 + 1
    num_farms = 10
    
    X_list = [[] for _ in range(num_farms)]
    Y_list = [[] for _ in range(num_farms)]
    
    for i in range(window_size, num_time_steps - window_size + 1):
        # V3 专属：提取全局 10 个风场的历史出力 (240 维)
        if 'v3' in version:
            past_power_global = power_data[i-window_size : i, :].flatten() 
            
        for f in range(num_farms):
            past_power_local = power_data[i-window_size : i, f] 
            p_weather = weather_features[i-window_size : i, f, :].flatten()
            f_weather = weather_features[i : i+window_size, f, :].flatten()
            
            # V1: 24(历史出力) + 144 + 144 = 312 维
            if version == 'v1':
                x_row = np.concatenate([past_power_local, p_weather, f_weather])
                
            # V3, V3_lite, V3_1: 240(全局出力) + 4(统计) + 312 + 312 = 868 维
            elif 'v3' in version:
                p_stats = np.array([np.mean(past_power_local), np.std(past_power_local), 
                                    np.max(past_power_local), np.min(past_power_local)])
                x_row = np.concatenate([past_power_global, p_stats, p_weather, f_weather])
                
            # V2: 24(本场出力) + 4(统计) + 312 + 312 = 652 维
            elif version == 'v2':
                p_stats = np.array([np.mean(past_power_local), np.std(past_power_local), 
                                    np.max(past_power_local), np.min(past_power_local)])
                x_row = np.concatenate([past_power_local, p_stats, p_weather, f_weather])
                
            y_row = power_data[i : i+window_size, f]
            X_list[f].append(x_row)
            Y_list[f].append(y_row)
            
    return [np.array(X) for X in X_list], [np.array(Y) for Y in Y_list]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True, help='评测模型版本：v1, v2, v3, v3_lite, v3_1')
    parser.add_argument('--test-data', type=str, default='data/wind_test_2013-07-14_to_end.npy')
    args = parser.parse_args()

    # ================= 🚀 核心新增：解析路由与瘦身索引 =================
    top_indices = None
    routing_dict = None
    
    # V3_Lite (全局一刀切瘦身)
    if args.version == 'v3_lite':
        index_file = 'data/top_indices_v3.npy'
        if os.path.exists(index_file):
            print(f"🎯 检测到 {args.version}，加载统一特征筛选索引: {index_file}")
            top_indices = np.load(index_file)
        else:
            print(f"❌ 错误：找不到索引文件 {index_file}")
            return
            
    # V3_1 (千人千面路由瘦身)
    elif args.version == 'v3_1':
        dict_file = 'data/v3_1_routing_dict.npy'
        if os.path.exists(dict_file):
            print(f"🎯 检测到 {args.version}，加载千人千面路由字典: {dict_file}")
            routing_dict = np.load(dict_file)
        else:
            print(f"❌ 错误：找不到路由字典 {dict_file}")
            return
    # ===================================================================

    model_dir = os.path.join('saved_models', args.version)
    if not os.path.exists(model_dir):
        print(f"❌ 找不到模型目录: {model_dir}")
        return

    print(f"📦 正在加载测试集: {args.test_data}...")
    raw_test = np.load(args.test_data, allow_pickle=True)
    numeric_test = raw_test[:, :, 1:].astype(np.float32)
    
    # --- 数据清洗防线 ---
    if np.isnan(numeric_test).any():
        print(f"⚠️ 警告：测试集中发现 NaN，正在进行插值处理...")
        for f in range(numeric_test.shape[1]): 
            for c in range(numeric_test.shape[2]): 
                series = pd.Series(numeric_test[:, f, c])
                numeric_test[:, f, c] = series.interpolate().ffill().bfill().values
        print("✅ 空值填充完毕。")
    
    # 构造基础矩阵 X 和 Y
    weather_feat, power_data = feature_engineering_factory(numeric_test, args.version)
    X_farms, Y_farms = build_test_X_Y(weather_feat, power_data, args.version)
    
    results = []
    print(f"\n🚀 开始为版本 [{args.version}] 进行全量评测...")
    
    for f in range(10):
        model_path = os.path.join(model_dir, f'xgb_{args.version}_farm_{f}.pkl')
        if not os.path.exists(model_path):
            print(f"⚠️ 找不到风场 {f} 的模型: {model_path}，跳过")
            continue
            
        model = joblib.load(model_path)
        X_test = X_farms[f]
        
        # --- 🎯 魔法路由切片 (针对瘦身版模型) ---
        if routing_dict is not None:
            # v3_1: 取当前风场的专属 150 维
            X_test = X_test[:, routing_dict[f]]
        elif top_indices is not None:
            # v3_lite: 取全局统一的 150 维
            X_test = X_test[:, top_indices]
            
        # 确保 GPU 内存连续性
        X_test = np.ascontiguousarray(X_test)
        Y_true = Y_farms[f]
        
        print(f"   ⚙️ Farm {f} - 当前推断矩阵特征维度: {X_test.shape[1]} 维")
        
        Y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
        mae = mean_absolute_error(Y_true, Y_pred)
        
        results.append(f"Farm {f}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        print(f"   ✅ Farm {f} 测试完毕: MAE = {mae:.4f}")

    avg_rmse = np.mean([float(r.split('=')[1].split(',')[0]) for r in results])
    avg_mae = np.mean([float(r.split('=')[2]) for r in results])
    
    summary = "\n".join(results)
    header = f"模型版本: {args.version}\n测试数据集: {args.test_data}\n" + "="*30 + "\n"
    footer = "\n" + "="*30 + f"\n全局平均 RMSE: {avg_rmse:.4f}\n全局平均 MAE: {avg_mae:.4f}"
    
    final_report = header + summary + footer
    report_path = os.path.join(model_dir, f"test_results_{args.version}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f_out:
        f_out.write(final_report)
        
    print("\n" + "="*40)
    print(f"🎉 版本 [{args.version}] 评测报告已生成: {report_path}")
    print(f"🏆 全局平均 RMSE: {avg_rmse:.4f}")
    print(f"🏆 全局平均 MAE: {avg_mae:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()