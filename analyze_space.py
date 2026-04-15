import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_cross_correlation(data_path="data/wind_train_val_2012-01-02_to_2013-07-13.npy", 
                              target_node=0, 
                              max_lag=12):
    """
    使用互相关函数 (CCF) 分析目标风场与其他风场之间的时空延迟相关性。
    
    :param data_path: 数据集路径
    :param target_node: 我们要分析的目标风场 (如 Node 0)
    :param max_lag: 往前/往后看的最大时间延迟 (小时)，默认 12 小时足够了
    """
    print(f"🚀 正在加载数据，分析风电场 {target_node} 与其他风场的时空延迟相关性...")
    
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件，请检查路径: {data_path}")
        return

    raw_data = np.load(data_path, allow_pickle=True)
    num_nodes = raw_data.shape[1]
    
    # 提取 10 个风场的绝对风速 (Index 1 是 U10, Index 2 是 V10)
    # 我们用风速算相关性比用功率更好，因为功率有硬截断(满发)，会掩盖真实的物理气流传递
    u10_all = raw_data[:, :, 1].astype(float)
    v10_all = raw_data[:, :, 2].astype(float)
    ws_all = np.sqrt(u10_all**2 + v10_all**2)
    
    # 构建一个 DataFrame，每一列是一个风场的风速
    df_ws = pd.DataFrame(ws_all, columns=[f"Node_{i}" for i in range(num_nodes)])
    
    # 处理可能的 NaN
    if df_ws.isnull().any().any():
        df_ws.interpolate(method='linear', inplace=True)
        df_ws.fillna(method='bfill', inplace=True)
        df_ws.fillna(method='ffill', inplace=True)

    print(f"✅ 数据准备完毕。开始计算 CCF (最大延迟: ±{max_lag} 小时)...")

    # =================================================================
    # 核心计算：计算 Target Node 与其他所有 Nodes 的 CCF
    # =================================================================
    lags = np.arange(-max_lag, max_lag + 1)
    ccf_matrix = np.zeros((num_nodes, len(lags)))
    
    target_series = df_ws[f"Node_{target_node}"]
    
    for i in range(num_nodes):
        other_series = df_ws[f"Node_{i}"]
        
        # 计算在不同 lag 下的相关系数
        for j, lag in enumerate(lags):
            # shift(lag): 
            # 正数 lag 代表 other_series 往后推 (也就是 target 当前时刻 vs other 过去时刻)
            # 负数 lag 代表 other_series 往前拉 (也就是 target 当前时刻 vs other 未来时刻)
            shifted_other = other_series.shift(lag)
            
            # 计算皮尔逊相关系数
            corr = target_series.corr(shifted_other)
            ccf_matrix[i, j] = corr

    # =================================================================
    # 可视化：绘制时空延迟相关性热力图 (Heatmap)
    # =================================================================
    print("📊 正在生成时空延迟热力图...")
    plt.figure(figsize=(14, 8), dpi=100)
    
    # 使用 Seaborn 画热力图
    # cmap 选用 coolwarm，红色代表强正相关，蓝色代表强负相关
    ax = sns.heatmap(ccf_matrix, 
                     xticklabels=lags, 
                     yticklabels=[f"Node {i}" for i in range(num_nodes)], 
                     cmap="coolwarm", 
                     center=0,
                     annot=False, # 如果想要看具体数字，可以设为 True
                     cbar_kws={'label': 'Pearson Correlation'})
    
    plt.title(f"Spatiotemporal Cross-Correlation (Target: Node {target_node})", fontsize=16, fontweight='bold')
    plt.xlabel(f"Time Lag (Hours)\n<-- Node i happened IN THE PAST | Node i happens IN THE FUTURE -->", fontsize=12)
    plt.ylabel("Source Node i", fontsize=12)
    
    # 突出显示 Lag=0 的中心线 (纯空间同步相关性)
    plt.axvline(x=max_lag + 0.5, color='black', linestyle='--', linewidth=2, label="Lag 0 (Synchronous)")
    
    # 把 Target Node 自己的那一行标出来 (就是最开始的 ACF)
    ax.add_patch(plt.Rectangle((0, target_node), len(lags), 1, fill=False, edgecolor='gold', lw=3, label="Target Node (ACF)"))
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = f"spatiotemporal_ccf_node_{target_node}.png"
    plt.savefig(save_path)
    print(f"📸 热力图已保存为: {save_path}")

if __name__ == "__main__":
    # 你可以修改 target_node 看看谁是整个风电场群的“上游风口”
    analyze_cross_correlation(target_node=0, max_lag=12)