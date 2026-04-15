import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

def analyze_temporal_dependency(data_path="data/wind_train_val_2012-01-02_to_2013-07-13.npy", node_idx=0, lags=72):
    """
    探究风电功率序列的时序依赖性 (Temporal Dependency)。
    使用 ACF (自相关) 和 PACF (偏自相关) 进行定量分析。
    
    :param data_path: 数据集路径
    :param node_idx: 要分析的风电场节点索引 (0-9)
    :param lags: 要往前看多少个时间步 (默认 72 小时，即 3 天)
    """
    print(f"🚀 正在加载数据，分析风电场 {node_idx} 的时序依赖性...")
    
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件，请检查路径: {data_path}")
        return

    try:
        raw_data = np.load(data_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # =================================================================
    # 1. 提取功率序列并清洗
    # =================================================================
    # Index 5 是 Power
    power_array = raw_data[:, node_idx, 5].astype(float)
    time_array = raw_data[:, node_idx, 0]
    
    # 转换为 pandas Series 方便处理 NaN 和绘图
    power_series = pd.Series(power_array, index=pd.to_datetime(time_array))
    
    # 填补 NaN (如果存在的话，保证序列连续性)
    if power_series.isnull().any():
        print(f"⚠️ 发现 {power_series.isnull().sum()} 个缺失值，正在进行线性插值...")
        power_series.interpolate(method='linear', inplace=True)
        
    print(f"✅ 功率序列提取完毕，总长度: {len(power_series)} 步 (小时)。")

    # =================================================================
    # 2. 绘制 ACF (自相关) 和 PACF (偏自相关) 图
    # =================================================================
    print(f"📊 正在计算并绘制前 {lags} 个小时的 ACF 和 PACF...")
    
    # 设置画布，上下两张图
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=100)
    
    # --- 图 1：ACF (Autocorrelation Function) ---
    # ACF 衡量的是 t 时刻与 t-k 时刻的总体相关性 (包含了中间时刻的传递影响)
    sm.graphics.tsa.plot_acf(
        power_series, 
        lags=lags, 
        ax=axes[0], 
        alpha=0.05, # 95% 置信区间
        title=f"Autocorrelation Function (ACF) - Node {node_idx}"
    )
    axes[0].set_xlabel("Lag (Hours)")
    axes[0].set_ylabel("Correlation")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 画几条参考线，帮助你判断
    axes[0].axvline(x=24, color='red', linestyle=':', alpha=0.5, label='24h (1 Day)')
    axes[0].axvline(x=48, color='green', linestyle=':', alpha=0.5, label='48h (2 Days)')
    axes[0].legend()

    # --- 图 2：PACF (Partial Autocorrelation Function) ---
    # PACF 衡量的是 t 时刻与 t-k 时刻的“纯粹”相关性 (剔除了 t-1 到 t-k+1 的中间影响)
    # 这对判断模型到底需要看多远的历史 (比如 LSTM 的步长) 极其关键！
    sm.graphics.tsa.plot_pacf(
        power_series, 
        lags=lags, 
        ax=axes[1], 
        alpha=0.05, # 95% 置信区间
        method='ywm', # 推荐的计算方法，更稳定
        title=f"Partial Autocorrelation Function (PACF) - Node {node_idx}"
    )
    axes[1].set_xlabel("Lag (Hours)")
    axes[1].set_ylabel("Partial Correlation")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 同样画出周期参考线
    axes[1].axvline(x=24, color='red', linestyle=':', alpha=0.5)
    axes[1].axvline(x=48, color='green', linestyle=':', alpha=0.5)

    plt.tight_layout()
    
    # 保存图片
    save_path = f"temporal_dependency_node_{node_idx}.png"
    plt.savefig(save_path)
    print(f"📸 分析图已保存为: {save_path}")
    
    # 如果在本地运行，可以弹窗显示
    # plt.show()

if __name__ == "__main__":
    # 你可以修改 node_idx (0-9) 看看不同风场的情况
    analyze_temporal_dependency(node_idx=0, lags=72)