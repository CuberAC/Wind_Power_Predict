# ==============================================================================
# 脚本名称: 04_evaluate_xgboost.py
# 任务目标: 加载指定版本的 XGBoost 模型，在全新测试集上进行 24 小时日前预测评测
# 使用方法: python 04_evaluate_xgboost.py --version v2
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
    
    # 基础物理特征 (V1/V2/V3 通用)
    WS10 = np.sqrt(U10**2 + V10**2)
    WS100 = np.sqrt(U100**2 + V100**2)
    WS100_Cube = WS100 ** 3
    WDir_Rad = np.arctan2(V100, U100)
    Sin_WDir, Cos_WDir = np.sin(WDir_Rad), np.cos(WDir_Rad)
    
    # 时间特征 (这里由于测试集是连续的，我们手动构造 mock 时间循环，
    # 实际应用中建议从时间戳解析。这里为了快速运行采用通用逻辑)
    # 假设测试集从 0 点开始
    time_steps = numeric_data.shape[0]
    hours = np.array([h % 24 for h in range(time_steps)])
    Sin_Hour = np.repeat(np.sin(2 * np.pi * hours / 24)[:, np.newaxis], 10, axis=1)
    Cos_Hour = np.repeat(np.cos(2 * np.pi * hours / 24)[:, np.newaxis], 10, axis=1)
    # 月份特征在短期预测中影响较小，暂设为固定值以匹配维度
    Sin_Month = np.zeros_like(Sin_Hour)
    Cos_Month = np.zeros_like(Cos_Hour)

    # 拼装 13 维核心气象包
    weather_features = np.stack([
        U10, V10, U100, V100, WS10, WS100, WS100_Cube,
        Sin_WDir, Cos_WDir, Sin_Hour, Cos_Hour, Sin_Month, Cos_Month
    ], axis=-1)
    
    return weather_features, Power

def build_test_X_Y(weather_features, power_data, version, window_size=24):
    """
    根据版本逻辑（V2 或 V3）构建滑动窗口测试集
    """
    num_time_steps = weather_features.shape[0]
    num_samples = num_time_steps - window_size * 2 + 1
    num_farms = 10
    
    X_list = [[] for _ in range(num_farms)]
    Y_list = [[] for _ in range(num_farms)]
    
    for i in range(window_size, num_time_steps - window_size + 1):
        # V3 专属：提取全局 10 个风场的历史出力
        if version == 'v3':
            past_power_global = power_data[i-window_size : i, :].flatten() # 240维
            
        for f in range(num_farms):
            past_power_local = power_data[i-window_size : i, f] # 24维
            # 统计量
            p_stats = np.array([np.mean(past_power_local), np.std(past_power_local), 
                                np.max(past_power_local), np.min(past_power_local)])
            
            p_weather = weather_features[i-window_size : i, f, :].flatten()
            f_weather = weather_features[i : i+window_size, f, :].flatten()
            
            if version == 'v3':
                x_row = np.concatenate([past_power_global, p_stats, p_weather, f_weather])
            else: # v2 逻辑
                x_row = np.concatenate([past_power_local, p_stats, p_weather, f_weather])
                
            y_row = power_data[i : i+window_size, f]
            
            X_list[f].append(x_row)
            Y_list[f].append(y_row)
            
    return [np.array(X) for X in X_list], [np.array(Y) for Y in Y_list]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True, help='评测模型版本，如 v2, v3, lite')
    parser.add_argument('--test-data', type=str, default='data/wind_test_2013-07-14_to_end.npy')
    args = parser.parse_args()

    # Lite 版本索引处理：评测时使用与训练一致的特征子集
    top_indices = None
    if 'lite' in args.version.lower():
        index_file = 'data/top_indices_v3.npy'
        if os.path.exists(index_file):
            print(f"🎯 检测到 Lite 版本，正在加载特征筛选索引: {index_file}")
            top_indices = np.load(index_file)
        else:
            print(f"❌ 错误：找不到索引文件 {index_file}，无法运行 Lite 评测！")
            return

    # 1. 路径准备
    model_dir = os.path.join('saved_models', args.version)
    if not os.path.exists(model_dir):
        print(f"❌ 找不到模型目录: {model_dir}")
        return

    # 2. 加载测试数据
    print(f"📦 正在加载测试集: {args.test_data}...")
    raw_test = np.load(args.test_data, allow_pickle=True)
    # 剥离时间列
    numeric_test = raw_test[:, :, 1:].astype(np.float32)
    
     # ================= 🚀 核心新增：数据清洗防线 =================
    # 检查是否有空值
    if np.isnan(numeric_test).any():
        nan_count = np.isnan(numeric_test).sum()
        print(f"⚠️ 警告：测试集中发现 {nan_count} 个空值(NaN)！正在进行插值处理...")
        
        # 使用线性插值填充空值（时序数据最常用的方法，比填0更科学）
        # 如果是 3D 数组，我们需要对每一个维度分别处理
        for f in range(numeric_test.shape[1]): # 遍历10个风场
            for c in range(numeric_test.shape[2]): # 遍历5个特征
                series = pd.Series(numeric_test[:, f, c])
                # 先尝试线性插值，剩下的头尾空值用后向/前向填充
                numeric_test[:, f, c] = series.interpolate().ffill().bfill().values
        
        print("✅ 空值填充完毕。")
    # ===========================================================
    
    # 3. 特征工程处理
    weather_feat, power_data = feature_engineering_factory(numeric_test, args.version)
    X_farms, Y_farms = build_test_X_Y(weather_feat, power_data, args.version)
    
    # 4. 循环评测 10 个风场
    results = []
    print(f"🚀 开始为版本 [{args.version}] 进行全量测试...")
    
    for f in range(10):
        model_path = os.path.join(model_dir, f'xgb_{args.version}_farm_{f}.pkl')
        if not os.path.exists(model_path):
            print(f"⚠️ 找不到风场 {f} 的模型，跳过")
            continue
            
        model = joblib.load(model_path)
        
        # 预测
        X_test = X_farms[f]
        if top_indices is not None:
            X_test = X_test[:, top_indices]
        X_test = np.ascontiguousarray(X_test)
        Y_true = Y_farms[f]
        Y_pred = model.predict(X_test)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
        mae = mean_absolute_error(Y_true, Y_pred)
        
        results.append(f"Farm {f}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        print(f"✅ 风场 {f} 测试完毕: MAE = {mae:.4f}")

    # 5. 计算汇总并保存
    avg_rmse = np.mean([float(r.split('=')[1].split(',')[0]) for r in results])
    avg_mae = np.mean([float(r.split('=')[2]) for r in results])
    
    summary = "\n".join(results)
    header = f"模型版本: {args.version}\n测试数据集: {args.test_data}\n" + "="*30 + "\n"
    footer = "\n" + "="*30 + f"\n全局平均 RMSE: {avg_rmse:.4f}\n全局平均 MAE: {avg_mae:.4f}"
    
    final_report = header + summary + footer
    
    # 保存结果到对应文件夹下的 txt
    report_path = os.path.join(model_dir, f"test_results_{args.version}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
        
    print("\n" + "="*30)
    print(f"🎉 评测报告已生成: {report_path}")
    print(f"🏆 全局平均 MAE: {avg_mae:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()